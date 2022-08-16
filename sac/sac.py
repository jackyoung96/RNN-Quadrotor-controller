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

from .common.buffers import *
from .common.value_networks import *
from .common.policy_networks import *

from time import time


class SAC_Trainer():
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, action_scale=1.0, out_actf=F.tanh, device='cpu', target_update_interval=1,**kwargs):
        self.replay_buffer = replay_buffer
        self.device = device
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_dim = hidden_dim

        self.q_net1 = QNetwork(state_space, action_space, 128).to(self.device)
        self.q_net2 = QNetwork(state_space, action_space, 128).to(self.device)
        self.target_q_net1 = QNetwork(state_space, action_space, 128).to(self.device)
        self.target_q_net2 = QNetwork(state_space, action_space, 128).to(self.device)
        policy_actf = kwargs.get('activation', F.relu)
        self.policy_net = PolicyNetwork(state_space, action_space, hidden_dim, device, policy_actf, out_actf, action_scale).to(self.device)
        self.target_policy_net = PolicyNetwork(state_space, action_space, hidden_dim, device, policy_actf, out_actf, action_scale).to(self.device)
        self.behavior_net = PolicyNetwork(state_space, action_space, hidden_dim, device, policy_actf, out_actf, action_scale).to(self.device)
        self.is_behavior = False

        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)
        

        lr = kwargs.get('learning_rate',1e-4)
        self.lr = lr
        # weight_decay = kwargs.get('weight_decay',1e-4)
        weight_decay = kwargs.get('weight_decay',0.0)

        # Entropy coeff
        self._setup_entropy()
        
        self.lr_scheduler = kwargs.get('lr_scheduler', False)
        self.update_cnt = 0
        self.target_update_interval = target_update_interval

        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=lr, weight_decay=weight_decay)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=lr, weight_decay=weight_decay)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=weight_decay)

        if self.lr_scheduler:
            t_max = kwargs.get('t_max', 1000)
            self.scheduler_q1 = CyclicLR(self.q_optimizer1, base_lr=1e-7, max_lr=lr, step_size_up=t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
            self.scheduler_q2 = CyclicLR(self.q_optimizer2, base_lr=1e-7, max_lr=lr, step_size_up=t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
            self.scheduler_policy = CyclicLR(self.policy_optimizer, base_lr=1e-7, max_lr=lr, step_size_up=t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')

        self.reward_norm = kwargs.get('reward_norm', False)

    def _setup_entropy(self):
        self.target_entropy = -np.prod(self.action_space.shape).astype(np.float32)

        init_value = 1.0
        self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.lr)

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

    def get_action(self, state, last_action, **kwargs):
        return self.policy_net.get_action(state)
    
    def update(self, batch_size, gamma=0.99,soft_tau=5e-3):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).to(self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).to(self.device)
        if len(state.shape)==len(action.shape)+1:
            action = action.unsqueeze(-1)
        if len(state.shape)==len(reward.shape)+1:
            reward = reward.unsqueeze(-1)
        if len(state.shape)==len(done.shape)+1:
            done = done.unsqueeze(-1)

        new_action, log_prob, *_ = self.policy_net.evaluate(state)

        # Entropy loss update
        ent_coef = torch.exp(self.log_ent_coef.detach())
        ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

        with torch.no_grad():
            new_next_action, next_log_prob, *_ = self.target_policy_net.evaluate(next_state) 
            target_q_1 = self.target_q_net1(next_state, new_next_action)
            target_q_2 = self.target_q_net2(next_state, new_next_action)
            target_q_min = torch.min(target_q_1, target_q_2)
            target_q_min = target_q_min - ent_coef * next_log_prob.reshape(-1,1)
            target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        predicted_q_value1 = self.q_net1(state, action)
        predicted_q_value2 = self.q_net2(state, action)

        # Critic update
        q_value_loss1 = F.mse_loss(predicted_q_value1, target_q_value.detach()) / 2  # detach: no gradients for the variable
        q_value_loss2 = F.mse_loss(predicted_q_value2, target_q_value.detach()) / 2  
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.q_optimizer1.step()
        if self.lr_scheduler:
            self.scheduler_q1.step()
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.q_optimizer2.step()
        if self.lr_scheduler:
            self.scheduler_q2.step()

        # Actor updaate
        predicted_new_q_value = self.q_net1(state, new_action)
        policy_loss = (ent_coef * log_prob - predicted_new_q_value).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        if self.lr_scheduler:
            self.scheduler_policy.step()

        if self.update_cnt%self.target_update_interval==0:
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
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, rnn_type='RNN', out_actf=None, action_scale=1.0, device='cpu', policy_target_update_interval=2, **kwargs):
        self.replay_buffer = replay_buffer
        self.device = device
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        
        if 'RNN' in self.rnn_type:
            qnet = QNetworkRNN
            policy = PolicyNetworkRNN
        elif 'LSTM' in self.rnn_type:
            qnet = QNetworkLSTM
            policy = PolicyNetworkLSTM
        elif 'GRU' in self.rnn_type:
            qnet = QNetworkGRU
            policy = PolicyNetworkGRU
        else:
            assert NotImplementedError

        self.q_net1 = qnet(state_space, action_space, 128).to(self.device)
        self.q_net2 = qnet(state_space, action_space, 128).to(self.device)
        self.target_q_net1 = qnet(state_space, action_space, 128).to(self.device)
        self.target_q_net2 = qnet(state_space, action_space, 128).to(self.device)
        
        policy_actf = kwargs.get('policy_actf', F.tanh)
        rnn_dropout = kwargs.get('rnn_dropout', 0.5)
        self.policy_net = policy(state_space, action_space, hidden_dim, device, policy_actf, out_actf, action_scale, rnn_dropout=rnn_dropout).to(self.device)
        self.target_policy_net = policy(state_space, action_space, hidden_dim, device, policy_actf, out_actf, action_scale, rnn_dropout=rnn_dropout).to(self.device)
        self.behavior_net = policy(state_space, action_space, hidden_dim, device, policy_actf, out_actf, action_scale, rnn_dropout=rnn_dropout).to(self.device)
        self.is_behavior = False

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
    
        self.reward_norm = kwargs.get('reward_norm', False)

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

    def load_behavior(self, path):
        self.behavior_net.load_state_dict(torch.load(path+'_policy.pt', map_location=self.device))
        self.is_behavior = True
    
    def get_action(self, state, last_action, hidden_in, **kwargs):
        with torch.no_grad():
            if not self.is_behavior:
                return self.policy_net.get_action(state, last_action, hidden_in, **kwargs)
            else:
                # kwargs['explore_noise_scale'] = 0.0
                return self.behavior_net.get_action(state, last_action, hidden_in, **kwargs)

    def update(self, batch_size, norm_ftn, deterministic, eval_noise_scale, gamma=0.99, soft_tau=1e-3):
        state, action, last_action, reward, next_state, done, param = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state, next_state = map(norm_ftn, [state, next_state])

        B,L        = state.shape[:2]
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        last_action     = torch.FloatTensor(last_action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)  
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(self.device)

        if self.reward_norm:
            # Normalize regard to episode axis (clipped by 10.0)
            reward = torch.clamp((reward - reward.mean(dim=1, keepdim=True)) / (reward.std(dim=1, keepdim=True) + 1e-8), -10.0, 10.0)

        if "LSTM" in self.rnn_type:
            hidden_in = (torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device), \
                        torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
        else:
            hidden_in = torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device)

        predicted_q_value1, _ = self.q_net1(state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.q_net2(state, action, last_action, hidden_in)
        new_next_action, *_ = self.target_policy_net.evaluate(next_state, action, hidden_in, deterministic, eval_noise_scale=eval_noise_scale) # clipped normal noise

        # Training Q Function
        target_q_1, _ = self.target_q_net1(next_state, new_next_action, action, hidden_in)
        target_q_2, _ = self.target_q_net2(next_state, new_next_action, action, hidden_in)
        target_q_min = torch.min(target_q_1, target_q_2)

        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward 

        q_value_loss1 = ((predicted_q_value1 - target_q_value.detach())**2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((predicted_q_value2 - target_q_value.detach())**2).mean()
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        nn.utils.clip_grad_norm_(self.q_net1.parameters(), 0.5)
        self.q_optimizer1.step()
        if self.lr_scheduler:
            self.scheduler_q1.step()
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        nn.utils.clip_grad_norm_(self.q_net2.parameters(), 0.5)
        self.q_optimizer2.step()
        if self.lr_scheduler:
            self.scheduler_q2.step()

        # print(target_q_value[0][149].item(), predicted_q_value1[0][149].item(), q_value_loss1.item())
        
        policy_loss = None
        if self.update_cnt%self.policy_target_update_interval==0:

            new_action, *_= self.policy_net.evaluate(state, last_action, hidden_in, deterministic, eval_noise_scale=0.0)  # no noise, deterministic policy gradients
            predicted_new_q_value, _ = self.q_net1(state, new_action, last_action, hidden_in)

            policy_loss = - predicted_new_q_value.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
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
        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)


class TD3RNN_Trainer2(TD3RNN_Trainer):
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, param_num, rnn_type='RNN2', out_actf=None, action_scale=1.0, device='cpu', policy_target_update_interval=2, **kwargs):
        super().__init__(replay_buffer, state_space, action_space, hidden_dim, rnn_type=rnn_type.strip('2'), out_actf=out_actf, action_scale=action_scale,device=device, policy_target_update_interval=policy_target_update_interval, **kwargs)
        # self.q_net1 = QNetworkParam(state_space, action_space, param_num, hidden_dim).to(self.device)
        # self.q_net2 = QNetworkParam(state_space, action_space, param_num, hidden_dim).to(self.device)
        # self.target_q_net1 = QNetworkParam(state_space, action_space, param_num, hidden_dim).to(self.device)
        # self.target_q_net2 = QNetworkParam(state_space, action_space, param_num, hidden_dim).to(self.device)
        self.q_net1 = QNetworkParam(state_space, action_space, param_num, 128).to(self.device)
        self.q_net2 = QNetworkParam(state_space, action_space, param_num, 128).to(self.device)
        self.target_q_net1 = QNetworkParam(state_space, action_space, param_num, 128).to(self.device)
        self.target_q_net2 = QNetworkParam(state_space, action_space, param_num, 128).to(self.device)

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
        
        self.reward_norm = kwargs.get('reward_norm', False)

    def update(self, batch_size, norm_ftn, deterministic, eval_noise_scale, gamma=0.99, soft_tau=1e-3):
        state, action, last_action, reward, next_state, done, param = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state, next_state = norm_ftn(state), norm_ftn(next_state)

        B,L        = state.shape[:2]
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        last_action     = torch.FloatTensor(last_action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)  
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(self.device)
        param      = torch.FloatTensor(param[:,None,:]).expand(B,L,-1).to(self.device)
 
        if self.reward_norm:
            reward = (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)

        if "LSTM" in self.rnn_type:
            hidden_in = (torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device), \
                        torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
        else:
            hidden_in = torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device)

        new_next_action, *_ = self.target_policy_net.evaluate(next_state, action, hidden_in, deterministic, eval_noise_scale=eval_noise_scale) # clipped normal noise

        # I.I.D condition for non-recurrent critic
        iid = [[i for i in range(B)],list(np.random.randint(0,L,B))]
        
        predicted_q_value1 = self.q_net1(state[iid], action[iid], param[iid])
        predicted_q_value2 = self.q_net2(state[iid], action[iid], param[iid])

        # Training Q Function
        predicted_target_q1 = self.target_q_net1(next_state[iid], new_next_action[iid], param[iid])
        predicted_target_q2 = self.target_q_net2(next_state[iid], new_next_action[iid], param[iid])

        target_q_min = torch.min(predicted_target_q1, predicted_target_q2)

        target_q_value = reward[iid] + (1 - done[iid]) * gamma * target_q_min # if done==1, only reward

        q_value_loss1 = ((predicted_q_value1 - target_q_value.detach())**2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((predicted_q_value2 - target_q_value.detach())**2).mean()         
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        # nn.utils.clip_grad_norm_(self.q_net1.parameters(), 0.5)
        self.q_optimizer1.step()
        if self.lr_scheduler:
            self.scheduler_q1.step()
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        # nn.utils.clip_grad_norm_(self.q_net2.parameters(), 0.5)
        self.q_optimizer2.step()
        if self.lr_scheduler:
            self.scheduler_q2.step()
        
        policy_loss = None
        if self.update_cnt%self.policy_target_update_interval==0:
            
            new_action, *_= self.policy_net.evaluate(state, last_action, hidden_in, deterministic, eval_noise_scale=0.0)  # no noise, deterministic policy gradients
            predicted_new_q_value = self.q_net1(state, new_action, param)

            policy_loss = - predicted_new_q_value.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            # nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
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
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, param_num, rnn_type='RNN3', out_actf=None, action_scale=1.0, device='cpu', policy_target_update_interval=2, **kwargs):
        super(TD3RNN_Trainer3, self).__init__(replay_buffer, state_space, action_space, hidden_dim, rnn_type=rnn_type.strip('3'), out_actf=out_actf, action_scale=action_scale,device=device, policy_target_update_interval=policy_target_update_interval, **kwargs)
        if 'RNN' in self.rnn_type:
            qnet = QNetworkRNNParam
        elif 'LSTM' in self.rnn_type:
            qnet = QNetworkLSTMParam
        elif 'GRU' in self.rnn_type:
            qnet = QNetworkGRUParam
        else:
            assert NotImplementedError

        rnn_dropout = kwargs('rnn_dropout', 0.5)
        self.q_net1 = qnet(state_space, action_space, 128, param_num, rnn_dropout=rnn_dropout).to(self.device)
        self.q_net2 = qnet(state_space, action_space, 128, param_num, rnn_dropout=rnn_dropout).to(self.device)
        self.target_q_net1 = qnet(state_space, action_space, 128, param_num, rnn_dropout=rnn_dropout).to(self.device)
        self.target_q_net2 = qnet(state_space, action_space, 128, param_num, rnn_dropout=rnn_dropout).to(self.device)
        
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

        self.reward_norm = kwargs.get('reward_norm', False)

    def update(self, batch_size, norm_ftn, deterministic, eval_noise_scale, gamma=0.99, soft_tau=1e-3):
        state, action, last_action, reward, next_state, done, param = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        B,L        = state.shape[:2]
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        last_action     = torch.FloatTensor(last_action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)  
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(self.device)
        param      = torch.FloatTensor(param[:,None,:]).expand(B,L,-1).to(self.device)
 
        if self.reward_norm:
            # Normalize regard to episode axis (clipped by 10.0)
            reward = torch.clamp((reward - reward.mean(dim=1, keepdim=True)) / (reward.std(dim=1, keepdim=True) + 1e-8), -10.0, 10.0)

        if "LSTM" in self.rnn_type:
            hidden_in = (torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device), \
                        torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
            hidden_in_q = (torch.zeros([1, B, 128], dtype=torch.float).to(self.device), \
                        torch.zeros([1, B, 128], dtype=torch.float).to(self.device))
        else:
            hidden_in = torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device)
            hidden_in_q = torch.zeros([1, B, 128], dtype=torch.float).to(self.device)

        predicted_q_value1, _ = self.q_net1(state, action, last_action, hidden_in_q, param)
        predicted_q_value2, _ = self.q_net2(state, action, last_action, hidden_in_q, param)
        new_next_action, *_ = self.target_policy_net.evaluate(next_state, action, hidden_in, deterministic, eval_noise_scale=eval_noise_scale) # clipped normal noise

        # Training Q Function
        predicted_target_q1, _ = self.target_q_net1(next_state, new_next_action, action, hidden_in_q, param)
        predicted_target_q2, _ = self.target_q_net2(next_state, new_next_action, action, hidden_in_q, param)
        target_q_min = torch.min(predicted_target_q1, predicted_target_q2)

        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        q_value_loss1 = ((predicted_q_value1 - target_q_value.detach())**2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((predicted_q_value2 - target_q_value.detach())**2).mean()         
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        # nn.utils.clip_grad_norm_(self.q_net1.parameters(), 0.5)
        self.q_optimizer1.step()
        if self.lr_scheduler:
            self.scheduler_q1.step()
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        # nn.utils.clip_grad_norm_(self.q_net2.parameters(), 0.5)
        self.q_optimizer2.step()
        if self.lr_scheduler:
            self.scheduler_q2.step()
        
        policy_loss = None
        if self.update_cnt%self.policy_target_update_interval==0:

            new_action, *_= self.policy_net.evaluate(state, last_action, hidden_in, deterministic, eval_noise_scale=0.0)  # no noise, deterministic policy gradients
            predicted_new_q_value, _ = self.q_net1(state, new_action, last_action, hidden_in_q, param)

            policy_loss = - predicted_new_q_value.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            # nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
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
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, param_num, goal_dim, rnn_type='RNN', out_actf=None, action_scale=1.0, device='cpu', policy_target_update_interval=2, **kwargs):
        super().__init__(replay_buffer, state_space, action_space, hidden_dim, rnn_type=rnn_type.replace('sHER','').replace('HER',''), out_actf=out_actf, action_scale=action_scale,device=device, policy_target_update_interval=policy_target_update_interval, **kwargs)
        self.state_space, self.action_space, self.param_num, self.hidden_dim, self.goal_dim = \
            state_space, action_space, param_num, hidden_dim, goal_dim

        batchnorm = kwargs.get('batch_norm', False)
        # self.q_net1 = QNetworkGoalParam(state_space, action_space, param_num, hidden_dim, goal_dim, batchnorm=batchnorm).to(self.device)
        # self.q_net2 = QNetworkGoalParam(state_space, action_space, param_num, hidden_dim, goal_dim, batchnorm=batchnorm).to(self.device)
        # self.target_q_net1 = QNetworkGoalParam(state_space, action_space, param_num, hidden_dim, goal_dim, batchnorm=batchnorm).to(self.device)
        # self.target_q_net2 = QNetworkGoalParam(state_space, action_space, param_num, hidden_dim, goal_dim, batchnorm=batchnorm).to(self.device)
        self.q_net1 = QNetworkGoalParam(state_space, action_space, param_num, 128, goal_dim, batchnorm=batchnorm).to(self.device)
        self.q_net2 = QNetworkGoalParam(state_space, action_space, param_num, 128, goal_dim, batchnorm=batchnorm).to(self.device)
        self.target_q_net1 = QNetworkGoalParam(state_space, action_space, param_num, 128, goal_dim, batchnorm=batchnorm).to(self.device)
        self.target_q_net2 = QNetworkGoalParam(state_space, action_space, param_num, 128, goal_dim, batchnorm=batchnorm).to(self.device)
        if 'RNN' in self.rnn_type:
            policy = PolicyNetworkGoalRNN
        elif 'LSTM' in self.rnn_type:
            policy = PolicyNetworkGoalLSTM
        elif 'GRU' in self.rnn_type:
            policy = PolicyNetworkGoalGRU
        policy_actf = kwargs.get('policy_actf', F.tanh)
        rnn_dropout = kwargs.get('rnn_dropout', 0.5)
        self.policy_net = policy(state_space, action_space, hidden_dim, goal_dim, device, batchnorm=batchnorm, actf=policy_actf, out_actf=out_actf, action_scale=action_scale, rnn_dropout=rnn_dropout).to(self.device)
        self.target_policy_net = policy(state_space, action_space, hidden_dim, goal_dim, device, batchnorm=batchnorm, actf=policy_actf, out_actf=out_actf, action_scale=action_scale, rnn_dropout=rnn_dropout).to(self.device)
        self.behavior_net = policy(state_space, action_space, hidden_dim, goal_dim, device, batchnorm=batchnorm, actf=policy_actf, out_actf=out_actf, action_scale=action_scale, rnn_dropout=rnn_dropout).to(self.device)
        self.is_behavior = False

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

        self.use_her = kwargs.get('use_her', True)
        self.reward_norm = kwargs.get('reward_norm', False)

    def update(self, batch_size, norm_ftn, deterministic, eval_noise_scale, gamma=0.99, soft_tau=5e-3):
        if self.use_her:
            state, action, last_action, reward, next_state, done, param, goal = self.replay_buffer.sample(batch_size)
        else:
            state, action, last_action, reward, next_state, done, param, goal = self.replay_buffer.sample_original(batch_size)
        # print('sample:', state, action,  reward, done)
        state, next_state, goal = map(norm_ftn, [state, next_state, goal])


        B,L        = state.shape[:2]
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        last_action     = torch.FloatTensor(last_action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)  
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(self.device)
        param      = torch.FloatTensor(param[:,None,:]).expand(B,L,-1).to(self.device)
        goal       = torch.FloatTensor(goal).expand(B,L,-1).to(self.device)

        if self.reward_norm:
            # Normalize regard to episode axis (clipped by 10.0)
            reward = torch.clamp((reward - reward.mean(dim=1, keepdim=True)) / (reward.std(dim=1, keepdim=True) + 1e-8), -10.0, 10.0)
 
        if "LSTM" in self.rnn_type:
            hidden_in = (torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device), \
                        torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
        else:
            hidden_in = torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device)

        new_next_action, *_ = self.target_policy_net.evaluate(next_state, action, hidden_in, goal, deterministic, eval_noise_scale=eval_noise_scale) # clipped normal noise

        # I.I.D condition for non-recurrent critic
        # iid = [[i for i in range(B)],list(np.random.randint(0,L,B))]
        
        # predicted_q_value1 = self.q_net1(state[iid], action[iid], param[iid], goal[iid])
        # predicted_q_value2 = self.q_net2(state[iid], action[iid], param[iid], goal[iid])

        # # Training Q Function
        # predicted_target_q1 = self.target_q_net1(next_state[iid], new_next_action[iid], param[iid], goal[iid])
        # predicted_target_q2 = self.target_q_net2(next_state[iid], new_next_action[iid], param[iid], goal[iid])

        # target_q_min = torch.min(predicted_target_q1, predicted_target_q2)

        # target_q_value = reward[iid] + (1 - done[iid]) * gamma * target_q_min # if done==1, only reward

        predicted_q_value1 = self.q_net1(state, action, param, goal)
        predicted_q_value2 = self.q_net2(state, action, param, goal)

        # Training Q Function
        predicted_target_q1 = self.target_q_net1(next_state, new_next_action, param, goal)
        predicted_target_q2 = self.target_q_net2(next_state, new_next_action, param, goal)

        target_q_min = torch.min(predicted_target_q1, predicted_target_q2)

        target_q_value = reward + (1 - done) * gamma * target_q_min

        q_value_loss1 = ((predicted_q_value1 - target_q_value.detach())**2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((predicted_q_value2 - target_q_value.detach())**2).mean()         
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        # nn.utils.clip_grad_norm_(self.q_net1.parameters(), 0.5)
        self.q_optimizer1.step()
        if self.lr_scheduler:
            self.scheduler_q1.step()
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        # nn.utils.clip_grad_norm_(self.q_net2.parameters(), 0.5)
        self.q_optimizer2.step()
        if self.lr_scheduler:
            self.scheduler_q2.step()
        
        # Soft update the target nets
        self.target_q_net1=self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
        self.target_q_net2=self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)

        policy_loss = None
        if self.update_cnt%self.policy_target_update_interval==0:

            new_action, *_= self.policy_net.evaluate(state, last_action, hidden_in, goal, deterministic, eval_noise_scale=0.0)  # no noise, deterministic policy gradients
            predicted_new_q_value = self.q_net1(state, new_action, param, goal)

            policy_loss = - predicted_new_q_value.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()
            if self.lr_scheduler:
                self.scheduler_policy.step()
                
            # Soft update the target nets
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        self.update_cnt+=1
        return {"policy_loss": policy_loss.item() if not policy_loss is None else 0, 
                "q_loss_1": q_value_loss1.item(), 
                "q_loss_2": q_value_loss2.item()
                }

class TD3sHERRNN_Trainer(TD3RNN_Trainer):
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, param_num, rnn_type='RNN', out_actf=None, action_scale=1.0, device='cpu', policy_target_update_interval=2, **kwargs):
        super().__init__(replay_buffer, state_space, action_space, hidden_dim, rnn_type=rnn_type.replace('sHER',''), out_actf=out_actf, action_scale=action_scale,device=device, policy_target_update_interval=policy_target_update_interval, **kwargs)
        self.state_space, self.action_space, self.param_num, self.hidden_dim = \
            state_space, action_space, param_num, hidden_dim

        # self.q_net1 = QNetworkParam(state_space, action_space, param_num, hidden_dim).to(self.device)
        # self.q_net2 = QNetworkParam(state_space, action_space, param_num, hidden_dim).to(self.device)
        # self.target_q_net1 = QNetworkParam(state_space, action_space, param_num, hidden_dim).to(self.device)
        # self.target_q_net2 = QNetworkParam(state_space, action_space, param_num, hidden_dim).to(self.device)
        self.q_net1 = QNetworkParam(state_space, action_space, param_num, 128).to(self.device)
        self.q_net2 = QNetworkParam(state_space, action_space, param_num, 128).to(self.device)
        self.target_q_net1 = QNetworkParam(state_space, action_space, param_num, 128).to(self.device)
        self.target_q_net2 = QNetworkParam(state_space, action_space, param_num, 128).to(self.device)
        if 'RNN' in self.rnn_type:
            policy = PolicyNetworkRNN
        elif 'LSTM' in self.rnn_type:
            policy = PolicyNetworkLSTM
        elif 'GRU' in self.rnn_type:
            policy = PolicyNetworkGRU
        policy_actf = kwargs.get('policy_actf', F.tanh)
        self.policy_net = policy(state_space, action_space, hidden_dim,device,actf=policy_actf, out_actf=out_actf, action_scale=action_scale).to(self.device)
        self.target_policy_net = policy(state_space, action_space, hidden_dim, device, actf=policy_actf, out_actf=out_actf, action_scale=action_scale).to(self.device)
        self.behavior_net = policy(state_space, action_space, hidden_dim,device, actf=policy_actf, out_actf=out_actf, action_scale=action_scale).to(self.device)
        self.is_behavior = False

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

        self.use_her = kwargs.get('use_her', True)
        self.reward_norm = kwargs.get('reward_norm', False)

    def update(self, batch_size, norm_ftn, deterministic, eval_noise_scale, gamma=0.99, soft_tau=5e-3):
        if self.use_her:
            state, action, last_action, reward, next_state, done, param = self.replay_buffer.sample(batch_size)
        else:
            state, action, last_action, reward, next_state, done, param = self.replay_buffer.sample_original(batch_size)
        # print('sample:', state, action,  reward, done)
        state, next_state = map(norm_ftn, [state, next_state])


        B,L        = state.shape[:2]
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        last_action     = torch.FloatTensor(last_action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)  
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(self.device)
        param      = torch.FloatTensor(param[:,None,:]).expand(B,L,-1).to(self.device)

        if self.reward_norm:
            # Normalize regard to episode axis (clipped by 10.0)
            reward = torch.clamp((reward - reward.mean(dim=1, keepdim=True)) / (reward.std(dim=1, keepdim=True) + 1e-8), -10.0, 10.0)
 
        if "LSTM" in self.rnn_type:
            hidden_in = (torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device), \
                        torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
        else:
            hidden_in = torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device)

        predicted_q_value1 = self.q_net1(state, action, param)
        predicted_q_value2 = self.q_net2(state, action, param)
        new_next_action, *_ = self.target_policy_net.evaluate(next_state, action, hidden_in, deterministic, eval_noise_scale=eval_noise_scale) # clipped normal noise

        # Training Q Function
        predicted_target_q1 = self.target_q_net1(next_state, new_next_action, param)
        predicted_target_q2 = self.target_q_net2(next_state, new_next_action, param)
        target_q_min = torch.min(predicted_target_q1, predicted_target_q2)

        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        q_value_loss1 = ((predicted_q_value1 - target_q_value.detach())**2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((predicted_q_value2 - target_q_value.detach())**2).mean()         
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        # nn.utils.clip_grad_norm_(self.q_net1.parameters(), 0.5)
        self.q_optimizer1.step()
        if self.lr_scheduler:
            self.scheduler_q1.step()
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        # nn.utils.clip_grad_norm_(self.q_net2.parameters(), 0.5)
        self.q_optimizer2.step()
        if self.lr_scheduler:
            self.scheduler_q2.step()
        
        # Soft update the target nets
        self.target_q_net1=self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
        self.target_q_net2=self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)

        policy_loss = None
        if self.update_cnt%self.policy_target_update_interval==0:

            new_action, *_= self.policy_net.evaluate(state, last_action, hidden_in, deterministic, eval_noise_scale=0.0)  # no noise, deterministic policy gradients
            predicted_new_q_value = self.q_net1(state, new_action, param)

            policy_loss = - predicted_new_q_value.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()
            if self.lr_scheduler:
                self.scheduler_policy.step()
                
            # Soft update the target nets
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        self.update_cnt+=1
        return {"policy_loss": policy_loss.item() if not policy_loss is None else 0, 
                "q_loss_1": q_value_loss1.item(), 
                "q_loss_2": q_value_loss2.item()
                }








#####################################
## FastAdapt td3 ####################
#####################################

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