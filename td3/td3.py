'''
Twin Delayed DDPG (TD3), if no twin no delayed then it's DDPG.
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net, 1 target policy net
original paper: https://arxiv.org/pdf/1802.09477.pdf
'''
import math
import random
from sqlite3 import Timestamp

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, CyclicLR
import torch.nn.functional as F
from torch.distributions import Normal

from .common.buffers import ReplayBuffer, ReplayBufferPER
from .common.value_networks import *
from .common.policy_networks import *

from time import time

# torch.manual_seed(1234)  #Reproducibility

# GPU = True
# device_idx = 0
# if GPU:
#     device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
# else:
#     device = torch.device("cpu")
# print(device)

# parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
# parser.add_argument('--train', dest='train', action='store_true', default=False)
# parser.add_argument('--test', dest='test', action='store_true', default=False)

# args = parser.parse_args()

class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action
        



class TD3_Trainer():
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, action_scale=1.0,out_actf=None, device='cpu', policy_target_update_interval=1,**kwargs):
        self.replay_buffer = replay_buffer
        self.device = device

        self.q_net1 = QNetwork(state_space, action_space, hidden_dim).to(self.device)
        self.q_net2 = QNetwork(state_space, action_space, hidden_dim).to(self.device)
        self.target_q_net1 = QNetwork(state_space, action_space, hidden_dim).to(self.device)
        self.target_q_net2 = QNetwork(state_space, action_space, hidden_dim).to(self.device)
        self.policy_net = PolicyNetwork(state_space, action_space, hidden_dim, device, out_actf, action_scale).to(self.device)
        self.target_policy_net = PolicyNetwork(state_space, action_space, hidden_dim, device, out_actf, action_scale).to(self.device)
        # print('Q Network (1,2): ', self.q_net1)
        # print('Policy Network: ', self.policy_net)

        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)
        

        q_lr = kwargs.get('q_lr',1e-3)
        policy_lr = kwargs.get('policy_lr',1e-4)
        weight_decay = kwargs.get('weight_decay',1e-4)
        t_max = kwargs.get('t_max', 1000)
        self.lr_scheduler = kwargs.get('lr_scheduler', False)
        self.update_cnt = 0
        self.policy_target_update_interval = policy_target_update_interval

        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=q_lr, weight_decay=weight_decay)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=q_lr, weight_decay=weight_decay)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr, weight_decay=weight_decay)

        if self.lr_scheduler:
            # self.scheduler_q1 = CosineAnnealingLR(self.q_optimizer1, T_max=t_max, eta_min=0, last_epoch=-1, verbose=False)
            self.scheduler_q1 = CyclicLR(self.q_optimizer1, base_lr=1e-7, max_lr=q_lr, step_size_up=t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
            # self.scheduler_q2 = CosineAnnealingLR(self.q_optimizer2, T_max=t_max, eta_min=0, last_epoch=-1, verbose=False)
            self.scheduler_q2 = CyclicLR(self.q_optimizer2, base_lr=1e-7, max_lr=q_lr, step_size_up=t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
            # self.scheduler_policy = CosineAnnealingLR(self.policy_optimizer, T_max=t_max, eta_min=0, last_epoch=-1, verbose=False)
            self.scheduler_policy = CyclicLR(self.policy_optimizer, base_lr=1e-7, max_lr=policy_lr, step_size_up=t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')

    def target_ini(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(param.data)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
    # Soft update the target net
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net
    
    def update(self, batch_size, deterministic, eval_noise_scale, gamma=0.99,soft_tau=1e-3):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        if len(state.shape)==len(action.shape)+1:
            action = action.unsqueeze(-1)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        predicted_q_value1 = self.q_net1(state, action)
        predicted_q_value2 = self.q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state, deterministic, eval_noise_scale=0.0)  # no noise, deterministic policy gradients
        new_next_action, _, _, _, _ = self.target_policy_net.evaluate(next_state, deterministic, eval_noise_scale=eval_noise_scale) # clipped normal noise

        # reward = (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

    # Training Q Function
        target_q_1 = self.target_q_net1(next_state, new_next_action)
        target_q_2 = self.target_q_net2(next_state, new_next_action)
        target_q_min = torch.min(target_q_1, target_q_2)

        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        q_value_loss1 = ((predicted_q_value1 - target_q_value.detach())**2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((predicted_q_value2 - target_q_value.detach())**2).mean()  
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        nn.utils.clip_grad_norm_(self.q_net1.parameters(), 1.0)
        self.q_optimizer1.step()
        if self.lr_scheduler:
            self.scheduler_q1.step()
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        nn.utils.clip_grad_norm_(self.q_net2.parameters(), 1.0)
        self.q_optimizer2.step()
        if self.lr_scheduler:
            self.scheduler_q2.step()

        policy_loss = None
        if self.update_cnt%self.policy_target_update_interval==0:
        # This is the **Delayed** update of policy and all targets (for Q and policy). 
        # Training Policy Function
            ''' implementation 1 '''
            # predicted_new_q_value = torch.min(self.q_net1(state, new_action),self.q_net2(state, new_action))
            ''' implementation 2 '''
            predicted_new_q_value = self.q_net1(state, new_action)

            policy_loss = - predicted_new_q_value.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.policy_optimizer.step()
            if self.lr_scheduler:
                self.scheduler_policy.step()
        
        # Soft update the target nets
            self.target_q_net1=self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2=self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        self.update_cnt+=1

        return {"policy_loss": policy_loss.item() if not policy_loss is None else 0, 
                "q_loss_1": q_value_loss1.item(), 
                "q_loss_2": q_value_loss2.item()}

    def save_model(self, path):
        torch.save(self.q_net1.state_dict(), path+'_q1.pt')
        torch.save(self.q_net2.state_dict(), path+'_q2.pt')
        torch.save(self.policy_net.state_dict(), path+'_policy.pt')

    def load_model(self, path):
        self.q_net1.load_state_dict(torch.load(path+'_q1.pt', map_location=self.device))
        self.q_net2.load_state_dict(torch.load(path+'_q2.pt', map_location=self.device))
        self.policy_net.load_state_dict(torch.load(path+'_policy.pt', map_location=self.device))
        self.q_net1.eval()
        self.q_net2.eval()
        self.policy_net.eval()


class TD3RNN_Trainer():
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, rnn_type='RNN', out_actf=None, action_scale=1.0,device='cpu', policy_target_update_interval=1, **kwargs):
        self.replay_buffer = replay_buffer
        self.device = device
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        
        if rnn_type=='RNN':
            self.q_net1 = QNetworkRNN(state_space, action_space, hidden_dim).to(self.device)
            self.q_net2 = QNetworkRNN(state_space, action_space, hidden_dim).to(self.device)
            self.target_q_net1 = QNetworkRNN(state_space, action_space, hidden_dim).to(self.device)
            self.target_q_net2 = QNetworkRNN(state_space, action_space, hidden_dim).to(self.device)
            self.policy_net = PolicyNetworkRNN(state_space, action_space, hidden_dim, device, out_actf, action_scale).to(self.device)
            self.target_policy_net = PolicyNetworkRNN(state_space, action_space, hidden_dim, device, out_actf, action_scale).to(self.device)
        elif rnn_type=='LSTM':
            self.q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(self.device)
            self.q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(self.device)
            self.target_q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(self.device)
            self.target_q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(self.device)
            self.policy_net = PolicyNetworkLSTM(state_space, action_space, hidden_dim, device, out_actf, action_scale).to(self.device)
            self.target_policy_net = PolicyNetworkLSTM(state_space, action_space, hidden_dim, device, out_actf, action_scale).to(self.device)
        elif rnn_type=='GRU':
            self.q_net1 = QNetworkGRU(state_space, action_space, hidden_dim).to(self.device)
            self.q_net2 = QNetworkGRU(state_space, action_space, hidden_dim).to(self.device)
            self.target_q_net1 = QNetworkGRU(state_space, action_space, hidden_dim).to(self.device)
            self.target_q_net2 = QNetworkGRU(state_space, action_space, hidden_dim).to(self.device)
            self.policy_net = PolicyNetworkGRU(state_space, action_space, hidden_dim, device, out_actf, action_scale).to(self.device)
            self.target_policy_net = PolicyNetworkGRU(state_space, action_space, hidden_dim, device, out_actf, action_scale).to(self.device)
        else:
            assert NotImplementedError

        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)
        
        q_lr = kwargs.get('q_lr',1e-3)
        policy_lr = kwargs.get('policy_lr',1e-4)
        weight_decay = kwargs.get('weight_decay',1e-4)
        t_max = kwargs.get('t_max', 1000)
        self.lr_scheduler = kwargs.get('lr_scheduler', False)
        self.update_cnt = 0
        self.policy_target_update_interval = policy_target_update_interval

        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=q_lr, weight_decay=weight_decay)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=q_lr, weight_decay=weight_decay)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr, weight_decay=weight_decay)

        if self.lr_scheduler:
            # self.scheduler_q1 = CosineAnnealingLR(self.q_optimizer1, T_max=t_max, eta_min=0, last_epoch=-1, verbose=False)
            self.scheduler_q1 = CyclicLR(self.q_optimizer1, base_lr=1e-7, max_lr=q_lr, step_size_up=t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
            # self.scheduler_q2 = CosineAnnealingLR(self.q_optimizer2, T_max=t_max, eta_min=0, last_epoch=-1, verbose=False)
            self.scheduler_q2 = CyclicLR(self.q_optimizer2, base_lr=1e-7, max_lr=q_lr, step_size_up=t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
            # self.scheduler_policy = CosineAnnealingLR(self.policy_optimizer, T_max=t_max, eta_min=0, last_epoch=-1, verbose=False)
            self.scheduler_policy = CyclicLR(self.policy_optimizer, base_lr=1e-7, max_lr=policy_lr, step_size_up=t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
    
    def target_ini(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(param.data)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
    # Soft update the target net
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net
    
    def update(self, batch_size, deterministic, eval_noise_scale, gamma=0.99, soft_tau=1e-3):
        time_test = {}

        t_start = time()
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        time_test['sample'] = time()-t_start
        # print('sample:', state, action,  reward, done)

        t_start = time()
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        last_action     = torch.FloatTensor(last_action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)  
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(self.device)
        time_test['tensor'] = time()-t_start
 
        t_start = time()
        predicted_q_value1, _ = self.q_net1(state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.q_net2(state, action, last_action, hidden_in)
        time_test['q'] = time()-t_start
        t_start = time()
        new_action, hidden_out, hidden_all, *_= self.policy_net.evaluate(state, last_action, hidden_in, deterministic, eval_noise_scale=0.0)  # no noise, deterministic policy gradients
        new_next_action, hidden_out, hidden_all, *_ = self.target_policy_net.evaluate(next_state, action, hidden_out, deterministic, eval_noise_scale=eval_noise_scale) # clipped normal noise
        time_test['p'] = time()-t_start
        # reward = (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        t_start = time()
        target_q_1, _ = self.target_q_net1(next_state, new_next_action, action, hidden_out)
        target_q_2, _ = self.target_q_net2(next_state, new_next_action, action, hidden_out)
        target_q_min = torch.min(target_q_1, target_q_2)
        time_test['q_target'] = time()-t_start

        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward 
        

        # print("debug",predicted_q_value1[0,:,0], target_q_value[0,:,0])
        t_start = time()
        q_value_loss1 = ((predicted_q_value1 - target_q_value.detach())**2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((predicted_q_value2 - target_q_value.detach())**2).mean()
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        nn.utils.clip_grad_norm_(self.q_net1.parameters(), 1.0)
        self.q_optimizer1.step()
        if self.lr_scheduler:
            self.scheduler_q1.step()
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        nn.utils.clip_grad_norm_(self.q_net2.parameters(), 1.0)
        self.q_optimizer2.step()
        if self.lr_scheduler:
            self.scheduler_q2.step()

        # print(target_q_value[0][149].item(), predicted_q_value1[0][149].item(), q_value_loss1.item())
        
        policy_loss = None
        if self.update_cnt%self.policy_target_update_interval==0:
            # Training Policy Function
            ''' implementation 1 '''
            # predicted_new_q_value = torch.min(self.q_net1(state, new_action),self.q_net2(state, new_action))
            ''' implementation 2 '''
            predicted_new_q_value, _ = self.q_net1(state, new_action, last_action, hidden_in)

            policy_loss = - predicted_new_q_value.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.policy_optimizer.step()
            if self.lr_scheduler:
                self.scheduler_policy.step()
            # Soft update the target nets
            self.target_q_net1=self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2=self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)
        time_test['backward'] = time()-t_start
        self.update_cnt+=1

        return {"policy_loss": policy_loss.item() if not policy_loss is None else 0, 
                "q_loss_1": q_value_loss1.item(), 
                "q_loss_2": q_value_loss2.item(),
                "time_test":time_test}

    def save_model(self, path):
        torch.save(self.q_net1.state_dict(), path+'_q1.pt')
        torch.save(self.q_net2.state_dict(), path+'_q2.pt')
        torch.save(self.policy_net.state_dict(), path+'_policy.pt')

    def load_model(self, path):
        import os
        print(os.getcwd())
        self.q_net1.load_state_dict(torch.load(path+'_q1.pt', map_location=self.device))
        self.q_net2.load_state_dict(torch.load(path+'_q2.pt', map_location=self.device))
        self.policy_net.load_state_dict(torch.load(path+'_policy.pt', map_location=self.device))
        self.q_net1.eval()
        self.q_net2.eval()
        self.policy_net.eval()

class TD3RNN_Trainer2(TD3RNN_Trainer):
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, param_num, rnn_type='RNN', out_actf=None, action_scale=1.0, device='cpu', policy_target_update_interval=1, **kwargs):
        super().__init__(replay_buffer, state_space, action_space, hidden_dim, rnn_type=rnn_type.strip('2'), out_actf=out_actf, action_scale=action_scale,device=device, policy_target_update_interval=policy_target_update_interval, **kwargs)
        self.q_net1 = QNetworkParam(state_space, action_space, param_num, hidden_dim).to(self.device)
        self.q_net2 = QNetworkParam(state_space, action_space, param_num, hidden_dim).to(self.device)
        self.target_q_net1 = QNetworkParam(state_space, action_space, param_num, hidden_dim).to(self.device)
        self.target_q_net2 = QNetworkParam(state_space, action_space, param_num, hidden_dim).to(self.device)

        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        
        q_lr = kwargs.get('q_lr',1e-3)
        weight_decay = kwargs.get('weight_decay',1e-4)
        t_max = kwargs.get('t_max', 1000)
        self.lr_scheduler = kwargs.get('lr_scheduler', False)

        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=q_lr, weight_decay=weight_decay)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=q_lr, weight_decay=weight_decay)

        if self.lr_scheduler:
            self.scheduler_q1 = CyclicLR(self.q_optimizer1, base_lr=1e-7, max_lr=q_lr, step_size_up=t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
            self.scheduler_q2 = CyclicLR(self.q_optimizer2, base_lr=1e-7, max_lr=q_lr, step_size_up=t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
        
    def update(self, batch_size, deterministic, eval_noise_scale, gamma=0.99, soft_tau=1e-3):
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done, param = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        B,L        = state.shape[:2]
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        last_action     = torch.FloatTensor(last_action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)  
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(self.device)
        param      = torch.FloatTensor(param[:,None,:]).expand(B,L,-1).to(self.device)
 
        predicted_q_value1 = self.q_net1(state, action, param)
        predicted_q_value2 = self.q_net2(state, action, param)
        new_action, hidden_out, hidden_all, *_= self.policy_net.evaluate(state, last_action, hidden_in, deterministic, eval_noise_scale=0.0)  # no noise, deterministic policy gradients
        new_next_action, hidden_out, hidden_all, *_ = self.target_policy_net.evaluate(next_state, action, hidden_out, deterministic, eval_noise_scale=eval_noise_scale) # clipped normal noise

        # Training Q Function
        predicted_target_q1 = self.target_q_net1(next_state, new_next_action, param)
        predicted_target_q2 = self.target_q_net2(next_state, new_next_action, param)
        target_q_min = torch.min(predicted_target_q1, predicted_target_q2)

        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        q_value_loss1 = ((predicted_q_value1 - target_q_value.detach())**2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((predicted_q_value2 - target_q_value.detach())**2).mean()         
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        nn.utils.clip_grad_norm_(self.q_net1.parameters(), 1.0)
        self.q_optimizer1.step()
        if self.lr_scheduler:
            self.scheduler_q1.step()
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        nn.utils.clip_grad_norm_(self.q_net2.parameters(), 1.0)
        self.q_optimizer2.step()
        if self.lr_scheduler:
            self.scheduler_q2.step()
        
        policy_loss = None
        if self.update_cnt%self.policy_target_update_interval==0:
            # Training Policy Function
            ''' implementation 1 '''
            # predicted_new_q_value = torch.min(self.q_net1(state, new_action),self.q_net2(state, new_action))
            ''' implementation 2 '''
            predicted_new_q_value = self.q_net1(state, new_action, param)

            policy_loss = - predicted_new_q_value.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.policy_optimizer.step()
            if self.lr_scheduler:
                self.scheduler_policy.step()
            # Soft update the target nets
            self.target_q_net1=self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2=self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        self.update_cnt+=1
        return {"policy_loss": policy_loss.item() if not policy_loss is None else 0, 
                "q_loss_1": q_value_loss1.item(), 
                "q_loss_2": q_value_loss2.item()
                }

class TD3RNN_Trainer3(TD3RNN_Trainer):
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, param_num, rnn_type='RNN', out_actf=None, action_scale=1.0, device='cpu', policy_target_update_interval=1, **kwargs):
        super(TD3RNN_Trainer3, self).__init__(replay_buffer, state_space, action_space, hidden_dim, rnn_type=rnn_type.strip('3'), out_actf=out_actf, action_scale=action_scale,device=device, policy_target_update_interval=policy_target_update_interval, **kwargs)
        if rnn_type=='RNN3':
            self.q_net1 = QNetworkRNNParam(state_space, action_space, hidden_dim, param_num).to(self.device)
            self.q_net2 = QNetworkRNNParam(state_space, action_space, hidden_dim, param_num).to(self.device)
            self.target_q_net1 = QNetworkRNNParam(state_space, action_space, hidden_dim, param_num).to(self.device)
            self.target_q_net2 = QNetworkRNNParam(state_space, action_space, hidden_dim, param_num).to(self.device)
        elif rnn_type=='LSTM3':
            self.q_net1 = QNetworkLSTMParam(state_space, action_space, hidden_dim, param_num).to(self.device)
            self.q_net2 = QNetworkLSTMParam(state_space, action_space, hidden_dim, param_num).to(self.device)
            self.target_q_net1 = QNetworkLSTMParam(state_space, action_space, hidden_dim, param_num).to(self.device)
            self.target_q_net2 = QNetworkLSTMParam(state_space, action_space, hidden_dim, param_num).to(self.device)
        elif rnn_type=='GRU3':
            self.q_net1 = QNetworkGRUParam(state_space, action_space, hidden_dim, param_num).to(self.device)
            self.q_net2 = QNetworkGRUParam(state_space, action_space, hidden_dim, param_num).to(self.device)
            self.target_q_net1 = QNetworkGRUParam(state_space, action_space, hidden_dim, param_num).to(self.device)
            self.target_q_net2 = QNetworkGRUParam(state_space, action_space, hidden_dim, param_num).to(self.device)
        else:
            assert NotImplementedError


    def update(self, batch_size, deterministic, eval_noise_scale, gamma=0.99, soft_tau=1e-3):
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done, param = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        B,L        = state.shape[:2]
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        last_action     = torch.FloatTensor(last_action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)  
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(self.device)
        param      = torch.FloatTensor(param[:,None,:]).expand(B,L,-1).to(self.device)
 
        predicted_q_value1, _ = self.q_net1(state, action, last_action, hidden_in, param)
        predicted_q_value2, _ = self.q_net1(state, action, last_action, hidden_in, param)
        new_action, hidden_out, hidden_all, *_= self.policy_net.evaluate(state, last_action, hidden_in, deterministic, eval_noise_scale=0.0)  # no noise, deterministic policy gradients
        new_next_action, hidden_out, hidden_all, *_ = self.target_policy_net.evaluate(next_state, action, hidden_out, deterministic, eval_noise_scale=eval_noise_scale) # clipped normal noise

        # Training Q Function
        predicted_target_q1, _ = self.target_q_net1(next_state, new_next_action, action, hidden_out, param)
        predicted_target_q2, _ = self.target_q_net2(next_state, new_next_action, action, hidden_out, param)
        target_q_min = torch.min(predicted_target_q1, predicted_target_q2)

        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        q_value_loss1 = ((predicted_q_value1 - target_q_value.detach())**2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((predicted_q_value2 - target_q_value.detach())**2).mean()         
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        nn.utils.clip_grad_norm_(self.q_net1.parameters(), 1.0)
        self.q_optimizer1.step()
        if self.lr_scheduler:
            self.scheduler_q1.step()
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        nn.utils.clip_grad_norm_(self.q_net2.parameters(), 1.0)
        self.q_optimizer2.step()
        if self.lr_scheduler:
            self.scheduler_q2.step()
        
        policy_loss = None
        if self.update_cnt%self.policy_target_update_interval==0:
            # Training Policy Function
            ''' implementation 1 '''
            # predicted_new_q_value = torch.min(self.q_net1(state, new_action),self.q_net2(state, new_action))
            ''' implementation 2 '''
            predicted_new_q_value, _ = self.q_net1(state, new_action, last_action, hidden_in, param)

            policy_loss = - predicted_new_q_value.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.policy_optimizer.step()
            if self.lr_scheduler:
                self.scheduler_policy.step()
            # Soft update the target nets
            self.target_q_net1=self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2=self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        self.update_cnt+=1
        return {"policy_loss": policy_loss.item() if not policy_loss is None else 0, 
                "q_loss_1": q_value_loss1.item(), 
                "q_loss_2": q_value_loss2.item()
                }

class TD3HERRNN_Trainer(TD3RNN_Trainer):
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, param_num, goal_dim, rnn_type='RNN', out_actf=None, action_scale=1.0, device='cpu', policy_target_update_interval=1, **kwargs):
        super().__init__(replay_buffer, state_space, action_space, hidden_dim, rnn_type=rnn_type.replace('HER',''), out_actf=out_actf, action_scale=action_scale,device=device, policy_target_update_interval=policy_target_update_interval, **kwargs)
        self.state_space, self.action_space, self.param_num, self.hidden_dim, self.goal_dim = \
            state_space, action_space, param_num, hidden_dim, goal_dim
        self.q_net1 = QNetworkGoalParam(state_space, action_space, param_num, hidden_dim, goal_dim).to(self.device)
        self.q_net2 = QNetworkGoalParam(state_space, action_space, param_num, hidden_dim, goal_dim).to(self.device)
        self.target_q_net1 = QNetworkGoalParam(state_space, action_space, param_num, hidden_dim, goal_dim).to(self.device)
        self.target_q_net2 = QNetworkGoalParam(state_space, action_space, param_num, hidden_dim, goal_dim).to(self.device)
        if rnn_type=='RNNHER':
            self.policy_net = PolicyNetworkGoalRNN(state_space, action_space, hidden_dim, goal_dim, device, out_actf, action_scale).to(self.device)
            self.target_policy_net = PolicyNetworkGoalRNN(state_space, action_space, hidden_dim, goal_dim, device, out_actf, action_scale).to(self.device)
        elif rnn_type=='LSTMHER':
            self.policy_net = PolicyNetworkGoalLSTM(state_space, action_space, hidden_dim, goal_dim, device, out_actf, action_scale).to(self.device)
            self.target_policy_net = PolicyNetworkGoalLSTM(state_space, action_space, hidden_dim, goal_dim, device, out_actf, action_scale).to(self.device)
        elif rnn_type=='GRUHER':
            self.policy_net = PolicyNetworkGoalGRU(state_space, action_space, hidden_dim, goal_dim, device, out_actf, action_scale).to(self.device)
            self.target_policy_net = PolicyNetworkGoalGRU(state_space, action_space, hidden_dim, goal_dim, device, out_actf, action_scale).to(self.device)

        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)
        
        self.q_lr = kwargs.get('q_lr',1e-3)
        self.policy_lr = kwargs.get('policy_lr',1e-4)
        self.weight_decay = kwargs.get('weight_decay',1e-4)
        self.t_max = kwargs.get('t_max', 1000)
        self.lr_scheduler = kwargs.get('lr_scheduler', False)

        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=self.q_lr, weight_decay=self.weight_decay)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=self.q_lr, weight_decay=self.weight_decay)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr, weight_decay=self.weight_decay)

        if self.lr_scheduler:
            self.scheduler_q1 = CyclicLR(self.q_optimizer1, base_lr=1e-7, max_lr=self.q_lr, step_size_up=self.t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
            self.scheduler_q2 = CyclicLR(self.q_optimizer2, base_lr=1e-7, max_lr=self.q_lr, step_size_up=self.t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
            self.scheduler_policy = CyclicLR(self.policy_optimizer, base_lr=1e-7, max_lr=self.policy_lr, step_size_up=self.t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
            # self.scheduler_policy = CyclicLR(self.policy_optimizer, base_lr=1e-7, max_lr=self.policy_lr, step_size_up=self.t_max//self.policy_target_update_interval, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')

        self.use_her = True
    
    def reset_critic(self):
        self.q_net1 = QNetworkGoalParam(self.state_space, self.action_space, self.param_num, self.hidden_dim, self.goal_dim).to(self.device)
        self.q_net2 = QNetworkGoalParam(self.state_space, self.action_space, self.param_num, self.hidden_dim, self.goal_dim).to(self.device)
        self.target_q_net1 = QNetworkGoalParam(self.state_space, self.action_space, self.param_num, self.hidden_dim, self.goal_dim).to(self.device)
        self.target_q_net2 = QNetworkGoalParam(self.state_space, self.action_space, self.param_num, self.hidden_dim, self.goal_dim).to(self.device)
        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=self.q_lr, weight_decay=self.weight_decay)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=self.q_lr, weight_decay=self.weight_decay)
        if self.lr_scheduler:
            self.scheduler_q1 = CyclicLR(self.q_optimizer1, base_lr=1e-7, max_lr=self.q_lr, step_size_up=self.t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
            self.scheduler_q2 = CyclicLR(self.q_optimizer2, base_lr=1e-7, max_lr=self.q_lr, step_size_up=self.t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')

    def update(self, batch_size, deterministic, eval_noise_scale, gamma=0.99, soft_tau=1e-3):
        if self.use_her:
            hidden_in, hidden_out, state, action, last_action, reward, next_state, done, param, goal = self.replay_buffer.sample(batch_size)
        else:
            hidden_in, hidden_out, state, action, last_action, reward, next_state, done, param, goal = self.replay_buffer.sample_original(batch_size)
        # print('sample:', state, action,  reward, done)

        B,L        = state.shape[:2]
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        last_action     = torch.FloatTensor(last_action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)  
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(self.device)
        param      = torch.FloatTensor(param[:,None,:]).expand(B,L,-1).to(self.device)
        goal       = torch.FloatTensor(goal).expand(B,L,-1).to(self.device)
 
        predicted_q_value1 = self.q_net1(state, action, param, goal)
        predicted_q_value2 = self.q_net2(state, action, param, goal)
        new_action, hidden_out, hidden_all, *_= self.policy_net.evaluate(state, last_action, hidden_in, goal, deterministic, eval_noise_scale=0.0)  # no noise, deterministic policy gradients
        new_next_action, hidden_out, hidden_all, *_ = self.target_policy_net.evaluate(next_state, action, hidden_out, goal, deterministic, eval_noise_scale=eval_noise_scale) # clipped normal noise

        # Training Q Function
        predicted_target_q1 = self.target_q_net1(next_state, new_next_action, param, goal)
        predicted_target_q2 = self.target_q_net2(next_state, new_next_action, param, goal)
        target_q_min = torch.min(predicted_target_q1, predicted_target_q2)

        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        q_value_loss1 = ((predicted_q_value1 - target_q_value.detach())**2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((predicted_q_value2 - target_q_value.detach())**2).mean()         
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        nn.utils.clip_grad_norm_(self.q_net1.parameters(), 1.0)
        self.q_optimizer1.step()
        if self.lr_scheduler:
            self.scheduler_q1.step()
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        nn.utils.clip_grad_norm_(self.q_net2.parameters(), 1.0)
        self.q_optimizer2.step()
        if self.lr_scheduler:
            self.scheduler_q2.step()
        
        policy_loss = None
        if self.update_cnt%self.policy_target_update_interval==0:
            # Training Policy Function
            ''' implementation 1 '''
            # predicted_new_q_value = torch.min(self.q_net1(state, new_action),self.q_net2(state, new_action))
            ''' implementation 2 '''
            predicted_new_q_value = self.q_net1(state, new_action, param, goal)

            policy_loss = - predicted_new_q_value.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.policy_optimizer.step()
            if self.lr_scheduler:
                self.scheduler_policy.step()
            # Soft update the target nets
            self.target_q_net1=self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2=self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        self.update_cnt+=1
        return {"policy_loss": policy_loss.item() if not policy_loss is None else 0, 
                "q_loss_1": q_value_loss1.item(), 
                "q_loss_2": q_value_loss2.item()
                }

class ParamPredictorNetwork(nn.Module):
    """ Base network class for value function approximation """
    def __init__(self, input_dim, param_num, activation=nn.Tanh):
        super(ParamPredictorNetwork, self).__init__()
        self.input_dim = input_dim
        self.param_num = param_num
        self.activation = activation() if activation is not None else None

        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, param_num),
        )

    def forward(self,x):
        if len(x.shape)>2:
            orig_shape = x.shape
            x = x.contiguous().view(-1, orig_shape[-1])
        out = self.net(x)
        out = out.view(*orig_shape[:-1], self.param_num)
        if self.activation is not None:
            out = self.activation(out)
        
        return out

class TD3FastAdaptRNN_Trainer(TD3RNN_Trainer):
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, param_num, rnn_type='fastRNN', out_actf=None, action_scale=1.0, device='cpu', policy_target_update_interval=1, **kwargs):
        super(TD3FastAdaptRNN_Trainer, self).__init__(replay_buffer, state_space, action_space, hidden_dim, rnn_type=rnn_type.strip('fast'), out_actf=out_actf, action_scale=action_scale,device=device, policy_target_update_interval=policy_target_update_interval, **kwargs)
        param_lr = kwargs.get('param_lr',1e-4)
        weight_decay = kwargs.get('weight_decay',1e-4)
        t_max = kwargs.get('t_max',1000)
        self.param_net = ParamPredictorNetwork(hidden_dim, param_num).to(self.device)
        self.param_optimizer = optim.Adam(self.param_net.parameters(), lr=param_lr, weight_decay=weight_decay)
        
        if self.lr_scheduler:
            # self.scheduler_param = CosineAnnealingLR(self.param_optimizer, T_max=t_max, eta_min=0, last_epoch=-1, verbose=False)
            self.scheduler_param = CyclicLR(self.param_optimizer, base_lr=1e-7, max_lr=param_lr, step_size_up=t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')

    def update(self, batch_size, deterministic, eval_noise_scale, gamma=0.99, soft_tau=1e-3):
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done, param = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        last_action     = torch.FloatTensor(last_action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)  
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(self.device)
 
        predicted_q_value1, _ = self.q_net1(state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.q_net2(state, action, last_action, hidden_in)
        new_action, hidden_out, hidden_all, *_= self.policy_net.evaluate(state, last_action, hidden_in, deterministic, eval_noise_scale=0.0)  # no noise, deterministic policy gradients
        new_next_action, hidden_out, hidden_all, *_ = self.target_policy_net.evaluate(next_state, action, hidden_out, deterministic, eval_noise_scale=eval_noise_scale) # clipped normal noise

        # Intrinsic rewards (Dynamic parameter convergence)
        target_param = torch.FloatTensor(param[:,None,:]).to(self.device)
        predicted_params = self.param_net(hidden_all.detach()) # detach: no gradients for policy (TODO gradient 유무에 따라 결과가 바뀔 수 있을 듯)
        d_param = ((predicted_params - target_param)**2).mean(dim=-1, keepdim=True)
        reward = reward - d_param.detach() # Intrinsic reward, detach: no gradients
        param_loss = d_param.mean()
        self.param_optimizer.zero_grad()
        param_loss.backward()
        nn.utils.clip_grad_norm_(self.param_net.parameters(), 1.0)
        self.param_optimizer.step()
        if self.lr_scheduler:
            self.scheduler_param.step()
        # reward = (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        predicted_target_q1, _ = self.target_q_net1(next_state, new_next_action, action, hidden_out)
        predicted_target_q2, _ = self.target_q_net2(next_state, new_next_action, action, hidden_out)
        target_q_min = torch.min(predicted_target_q1, predicted_target_q2)

        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        q_value_loss1 = ((predicted_q_value1 - target_q_value.detach())**2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((predicted_q_value2 - target_q_value.detach())**2).mean()         
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        nn.utils.clip_grad_norm_(self.q_net1.parameters(), 1.0)
        self.q_optimizer1.step()
        if self.lr_scheduler:
            self.scheduler_q1.step()
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        nn.utils.clip_grad_norm_(self.q_net2.parameters(), 1.0)
        self.q_optimizer2.step()
        if self.lr_scheduler:
            self.scheduler_q2.step()
        
        policy_loss = None
        if self.update_cnt%self.policy_target_update_interval==0:
            # Training Policy Function
            ''' implementation 1 '''
            # predicted_new_q_value = torch.min(self.q_net1(state, new_action),self.q_net2(state, new_action))
            ''' implementation 2 '''
            predicted_new_q_value, _ = self.q_net1(state, new_action, last_action, hidden_in)

            policy_loss = - predicted_new_q_value.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.policy_optimizer.step()
            if self.lr_scheduler:
                self.scheduler_policy.step()
            # Soft update the target nets
            self.target_q_net1=self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2=self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        self.update_cnt+=1
        return {"policy_loss": policy_loss.item() if not policy_loss is None else 0, 
                "q_loss_1": q_value_loss1.item(), 
                "q_loss_2": q_value_loss2.item(),
                "param_loss": param_loss.item()}

    def predict_param(self, hidden):
        with torch.no_grad():
            hidden = hidden.to(self.device)
            predicted_params = self.param_net(hidden) # detach: no gradients for policy (TODO gradient 유무에 따라 결과가 바뀔 수 있을 듯)

        return predicted_params.cpu().numpy()
        
    def save_model(self, path):
        super().save_model(path)
        torch.save(self.param_net.state_dict(), path+'_param.pt')

    def load_model(self, path):
        super().load_model(path)
        self.param_net.load_state_dict(torch.load(path+'_param.pt', map_location=self.device))
        self.param_net.eval()