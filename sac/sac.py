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
from .common.param_networks import *

from time import time


class SAC_Trainer():
    def __init__(self, replay_buffer, 
                        state_space, 
                        action_space, 
                        hidden_dim, 
                        action_scale=1.0, 
                        out_actf=F.tanh, 
                        device='cpu', 
                        target_update_interval=1,
                        **kwargs):
        self.replay_buffer = replay_buffer
        self.device = device
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_dim = hidden_dim

        self.q_net1 = QNetwork(state_space, action_space, 128).to(self.device)
        self.q_net2 = QNetwork(state_space, action_space, 128).to(self.device)
        self.target_q_net1 = QNetwork(state_space, action_space, 128).to(self.device)
        self.target_q_net2 = QNetwork(state_space, action_space, 128).to(self.device)
        self.policy_actf = kwargs.get('activation', F.relu)
        self.policy_net = PolicyNetwork(state_space, action_space, hidden_dim, device, self.policy_actf, out_actf, action_scale).to(self.device)
        self.target_policy_net = PolicyNetwork(state_space, action_space, hidden_dim, device, self.policy_actf, out_actf, action_scale).to(self.device)
        self.is_behavior = False

        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)
        

        self.lr = kwargs.get('learning_rate',1e-4)
        self.weight_decay = kwargs.get('weight_decay',0.0)

        # Entropy coeff
        self._setup_entropy()
        
        self.lr_scheduler = kwargs.get('lr_scheduler', False)
        self.update_cnt = 0
        self.target_update_interval = target_update_interval

        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.lr_scheduler:
            self.t_max = kwargs.get('t_max', 1000)
            self.scheduler_q1 = CyclicLR(self.q_optimizer1, base_lr=1e-7, max_lr=self.lr, step_size_up=self.t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
            self.scheduler_q2 = CyclicLR(self.q_optimizer2, base_lr=1e-7, max_lr=self.lr, step_size_up=self.t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
            self.scheduler_policy = CyclicLR(self.policy_optimizer, base_lr=1e-7, max_lr=self.lr, step_size_up=self.t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')

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

    def get_action(self, state, last_action, *args, **kwargs):
        return self.policy_net.get_action(state)
    
    def update(self, batch_size, gamma=0.99,soft_tau=5e-3):
        sample = self.replay_buffer.sample(batch_size)
        if len(sample)==5:
            state, action, reward, next_state, done = sample
        elif len(sample)==7:
            state, action, last_action, reward, next_state, done, param = sample
        else:
            raise "Wrong buffer sampling"

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


class SACparam_Trainer(SAC_Trainer):
    def __init__(self, replay_buffer, 
                        state_space, 
                        action_space, 
                        hidden_dim, 
                        param_dim,
                        rnn_type='RNNparam', 
                        out_actf=F.tanh, 
                        action_scale=1.0, 
                        device='cpu', 
                        target_update_interval=1, 
                        **kwargs):
        super(SACparam_Trainer, self).__init__(
            replay_buffer=replay_buffer, 
            state_space=state_space, 
            action_space=action_space, 
            hidden_dim=hidden_dim, 
            out_actf=out_actf, 
            action_scale=action_scale, 
            device=device, 
            target_update_interval=target_update_interval, 
            **kwargs
        )
        self.rnn_type = rnn_type.strip('param')
        self.param_dim = param_dim
        self.rnn_dropout = kwargs.get('rnn_dropout', 0.5)
        
        self.param_net = ParamNetwork(
            self.policy_net._state_dim-self.param_dim,
            self.policy_net._action_dim,
            self.param_dim,
            self.hidden_dim,
            self.rnn_type,
            rnn_dropout=self.rnn_dropout,
        ).to(self.device)
        self.param_optimizer = optim.Adam(self.param_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def get_action(self, state, last_action, *args, **kwargs):
        if kwargs.get('param',True):
            return self.policy_net.get_action(state)
        else:
            # param net -> policy net
            hidden_in = args[0]
            deterministic = kwargs.get('deterministic',False)
            state_torch = torch.FloatTensor(state[:-self.param_dim]).view(1,1,-1).to(self.device)
            last_action_torch = torch.FloatTensor(last_action).view(1,1,-1).to(self.device)
            param, hidden_out = self.param_net(state_torch, last_action_torch, hidden_in)
            state[-self.param_dim:] = param.detach().cpu().numpy().squeeze()
            return self.policy_net.get_action(state, deterministic=deterministic), hidden_out

    def update(self, batch_size, gamma=0.99, soft_tau=1e-3):
        loss_dict = super().update(batch_size, gamma=gamma, soft_tau=soft_tau)

        # Update param
        param_loss = self.update_param(batch_size)
        loss_dict.update({"param_loss": param_loss})

        return loss_dict

    def update_param(self, batch_size):
        state, action, last_action, reward, next_state, done, param = self.replay_buffer.sample_sequence(batch_size)
        
        B,L                = state.shape[:2]
        state_noparam      = torch.FloatTensor(state[:,:,:-self.param_dim]).to(self.device)
        param              = torch.FloatTensor(param).to(self.device)
        last_action        = torch.FloatTensor(last_action).to(self.device)

        if len(state.shape)==len(last_action.shape)+1:
            last_action = last_action.unsqueeze(-1)

        if "LSTM" in self.rnn_type:
            hidden_in = (torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device), \
                        torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
        else:
            hidden_in = torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device)

        predicted_param, hidden_out = self.param_net(state_noparam, last_action, hidden_in)
        # Many-to-One RNN
        # param_loss = F.mse_loss(predicted_param[:,-1], param.mean(dim=1))
        param_loss = F.mse_loss(predicted_param, param)
        
        self.param_optimizer.zero_grad()
        param_loss.backward()
        self.param_optimizer.step()

        return param_loss.item()

    def save_model(self, path):
        super().save_model(path)
        torch.save(self.param_net.state_dict(), path+'_param.pt')

    def load_model(self, path):
        super().load_model(path)
        self.param_net.load_state_dict(torch.load(path+'_param.pt', map_location=self.device))
        self.param_net.eval()


class SACRNN_Trainer(SAC_Trainer):
    def __init__(self, replay_buffer, 
                        state_space, 
                        action_space, 
                        hidden_dim, 
                        param_dim,
                        rnn_type='RNN', 
                        out_actf=F.tanh, 
                        action_scale=1.0, 
                        device='cpu', 
                        target_update_interval=1, 
                        **kwargs):
        super(SACRNN_Trainer, self).__init__(
            replay_buffer=replay_buffer, 
            state_space=state_space, 
            action_space=action_space, 
            hidden_dim=hidden_dim, 
            out_actf=out_actf, 
            action_scale=action_scale, 
            device=device, 
            target_update_interval=target_update_interval, 
            **kwargs
        )
        self.rnn_type = rnn_type
        self.param_dim = param_dim
        self.rnn_dropout = kwargs.get('rnn_dropout', 0.5)
        
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

        
        self.q_net1 = qnet(state_space, action_space, 128, self.param_dim, rnn_dropout=self.rnn_dropout).to(self.device)
        self.q_net2 = qnet(state_space, action_space, 128, self.param_dim, rnn_dropout=self.rnn_dropout).to(self.device)
        self.target_q_net1 = qnet(state_space, action_space, 128, self.param_dim, rnn_dropout=self.rnn_dropout).to(self.device)
        self.target_q_net2 = qnet(state_space, action_space, 128, self.param_dim, rnn_dropout=self.rnn_dropout).to(self.device)
        
        self.policy_net = policy(state_space, action_space, hidden_dim, param_dim, device, self.policy_actf, out_actf, action_scale, rnn_dropout=self.rnn_dropout).to(self.device)
        self.target_policy_net = policy(state_space, action_space, hidden_dim, param_dim, device, self.policy_actf, out_actf, action_scale, rnn_dropout=self.rnn_dropout).to(self.device)

        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)

        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.lr_scheduler:
            self.scheduler_q1 = CyclicLR(self.q_optimizer1, base_lr=1e-7, max_lr=self.lr, step_size_up=self.t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
            self.scheduler_q2 = CyclicLR(self.q_optimizer2, base_lr=1e-7, max_lr=self.lr, step_size_up=self.t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
            self.scheduler_policy = CyclicLR(self.policy_optimizer, base_lr=1e-7, max_lr=self.lr, step_size_up=self.t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
    
    def get_action(self, state, last_action, hidden_in, **kwargs):
        state = state[:-self.param_dim]
        deterministic = kwargs.get('deterministic',False)
        return self.policy_net.get_action(state, last_action, hidden_in, deterministic=deterministic)

    def update(self, batch_size, gamma=0.99, soft_tau=1e-3):
        state, action, last_action, reward, next_state, done, param = self.replay_buffer.sample_sequence(batch_size)

        B,L        = state.shape[:2]
        state      = torch.FloatTensor(state[:,:,:-self.param_dim]).to(self.device)
        next_state = torch.FloatTensor(next_state[:,:,:-self.param_dim]).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        last_action     = torch.FloatTensor(last_action).to(self.device)
        reward     = torch.FloatTensor(reward).to(self.device)  
        done       = torch.FloatTensor(np.float32(done)).to(self.device)
        param      = torch.FloatTensor(param).to(self.device)

        if len(state.shape)==len(action.shape)+1:
            action = action.unsqueeze(-1)
        if len(state.shape)==len(reward.shape)+1:
            reward = reward.unsqueeze(-1)
        if len(state.shape)==len(done.shape)+1:
            done = done.unsqueeze(-1)
        reward, done = reward[:,-1], done[:,-1]

        if "LSTM" in self.rnn_type:
            hidden_in_q = (torch.zeros([1, B, 128], dtype=torch.float).to(self.device), \
                        torch.zeros([1, B, 128], dtype=torch.float).to(self.device))
            hidden_in_p = (torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device), \
                        torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
        else:
            hidden_in_q = torch.zeros([1, B, 128], dtype=torch.float).to(self.device)
            hidden_in_p = torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device)

        new_action, log_prob, *_ = self.policy_net.evaluate(state, last_action, hidden_in_p)

        # Entropy loss update
        ent_coef = torch.exp(self.log_ent_coef.detach())
        ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

        with torch.no_grad():
            new_next_action, next_log_prob, *_ = self.target_policy_net.evaluate(next_state, action, hidden_in_p) 
            target_q_1, _ = self.target_q_net1(next_state, new_next_action, param, action, hidden_in_q)
            target_q_2, _ = self.target_q_net2(next_state, new_next_action, param, action, hidden_in_q)
            target_q_min = torch.min(target_q_1, target_q_2)
            target_q_min = target_q_min - ent_coef * next_log_prob.reshape(-1,1)
            target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        predicted_q_value1, _ = self.q_net1(state, action, param, last_action, hidden_in_q)
        predicted_q_value2, _ = self.q_net2(state, action, param, last_action, hidden_in_q)

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

        # Actor update
        predicted_new_q_value, _ = self.q_net1(state, new_action, param, last_action, hidden_in_q)
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

class SACRNNfull_Trainer(SACRNN_Trainer):
    def __init__(self, replay_buffer, 
                        state_space, 
                        action_space, 
                        hidden_dim, 
                        param_dim,
                        rnn_type='RNNfull', 
                        out_actf=F.tanh, 
                        action_scale=1.0, 
                        device='cpu', 
                        target_update_interval=1, 
                        **kwargs):
        super(SACRNNfull_Trainer, self).__init__(
            replay_buffer=replay_buffer, 
            state_space=state_space, 
            action_space=action_space, 
            hidden_dim=hidden_dim, 
            param_dim=param_dim,
            rnn_type=rnn_type.strip('full'),
            out_actf=out_actf, 
            action_scale=action_scale, 
            device=device, 
            target_update_interval=target_update_interval, 
            **kwargs
        )
        if 'RNN' in self.rnn_type:
            policy = PolicyNetworkRNNfull
            qnet = QNetworkRNNfull
        elif 'LSTM' in self.rnn_type:
            policy = PolicyNetworkLSTMfull
            qnet = QNetworkLSTMfull
        elif 'GRU' in self.rnn_type:
            policy = PolicyNetworkGRUfull
            qnet = QNetworkGRUfull
        else:
            assert NotImplementedError

        self.q_net1 = qnet(state_space, action_space, 128, self.param_dim, rnn_dropout=self.rnn_dropout).to(self.device)
        self.q_net2 = qnet(state_space, action_space, 128, self.param_dim, rnn_dropout=self.rnn_dropout).to(self.device)
        self.target_q_net1 = qnet(state_space, action_space, 128, self.param_dim, rnn_dropout=self.rnn_dropout).to(self.device)
        self.target_q_net2 = qnet(state_space, action_space, 128, self.param_dim, rnn_dropout=self.rnn_dropout).to(self.device)
        
        self.policy_net = policy(state_space, action_space, hidden_dim, param_dim, device, self.policy_actf, out_actf, action_scale, rnn_dropout=self.rnn_dropout).to(self.device)
        self.target_policy_net = policy(state_space, action_space, hidden_dim, param_dim, device, self.policy_actf, out_actf, action_scale, rnn_dropout=self.rnn_dropout).to(self.device)

        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)

        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.lr_scheduler:
            self.scheduler_q1 = CyclicLR(self.q_optimizer1, base_lr=1e-7, max_lr=self.lr, step_size_up=self.t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
            self.scheduler_q2 = CyclicLR(self.q_optimizer2, base_lr=1e-7, max_lr=self.lr, step_size_up=self.t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
            self.scheduler_policy = CyclicLR(self.policy_optimizer, base_lr=1e-7, max_lr=self.lr, step_size_up=self.t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
    
    def update(self, batch_size, gamma=0.99, soft_tau=1e-3):
        state, action, last_action, reward, next_state, done, param = self.replay_buffer.sample_sequence(batch_size)

        B,L        = state.shape[:2]
        state      = torch.FloatTensor(state[:,:,:-self.param_dim]).to(self.device)
        next_state = torch.FloatTensor(next_state[:,:,:-self.param_dim]).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        last_action     = torch.FloatTensor(last_action).to(self.device)
        reward     = torch.FloatTensor(reward).to(self.device)  
        done       = torch.FloatTensor(np.float32(done)).to(self.device)
        param      = torch.FloatTensor(param).to(self.device)

        if len(state.shape)==len(action.shape)+1:
            action = action.unsqueeze(-1)
        if len(state.shape)==len(reward.shape)+1:
            reward = reward.unsqueeze(-1)
        if len(state.shape)==len(done.shape)+1:
            done = done.unsqueeze(-1)

        if "LSTM" in self.rnn_type:
            hidden_in_q = (torch.zeros([1, B, 128], dtype=torch.float).to(self.device), \
                        torch.zeros([1, B, 128], dtype=torch.float).to(self.device))
            hidden_in_p = (torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device), \
                        torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
        else:
            hidden_in_q = torch.zeros([1, B, 128], dtype=torch.float).to(self.device)
            hidden_in_p = torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device)

        new_action, log_prob, *_ = self.policy_net.evaluate(state, last_action, hidden_in_p)

        # Entropy loss update
        ent_coef = torch.exp(self.log_ent_coef.detach())
        ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

        with torch.no_grad():
            new_next_action, next_log_prob, *_ = self.target_policy_net.evaluate(next_state, action, hidden_in_p) 
            target_q_1, _ = self.target_q_net1(next_state, new_next_action, param, action, hidden_in_q)
            target_q_2, _ = self.target_q_net2(next_state, new_next_action, param, action, hidden_in_q)
            target_q_min = torch.min(target_q_1, target_q_2)
            target_q_min = target_q_min - ent_coef * next_log_prob
            target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        predicted_q_value1, _ = self.q_net1(state, action, param, last_action, hidden_in_q)
        predicted_q_value2, _ = self.q_net2(state, action, param, last_action, hidden_in_q)

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

        # Actor update
        predicted_new_q_value, _ = self.q_net1(state, new_action, param, last_action, hidden_in_q)
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

class SACRNNpolicy_Trainer(SACRNN_Trainer):
    def __init__(self, replay_buffer, 
                        state_space, 
                        action_space, 
                        hidden_dim, 
                        param_dim, 
                        rnn_type='RNNpolicy', 
                        out_actf=None, 
                        action_scale=1.0, 
                        device='cpu', 
                        policy_target_update_interval=2, 
                        **kwargs):
        super().__init__(replay_buffer, 
                            state_space, 
                            action_space, 
                            hidden_dim, 
                            param_dim, 
                            rnn_type=rnn_type.strip('policy'), 
                            out_actf=out_actf, 
                            action_scale=action_scale,
                            device=device, 
                            policy_target_update_interval=policy_target_update_interval, 
                            **kwargs)

        self.q_net1 = QNetwork(state_space, action_space, 128).to(self.device)
        self.q_net2 = QNetwork(state_space, action_space, 128).to(self.device)
        self.target_q_net1 = QNetwork(state_space, action_space, 128).to(self.device)
        self.target_q_net2 = QNetwork(state_space, action_space, 128).to(self.device)

        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)

        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.lr_scheduler:
            self.scheduler_q1 = CyclicLR(self.q_optimizer1, base_lr=1e-7, max_lr=self.lr, step_size_up=self.t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')
            self.scheduler_q2 = CyclicLR(self.q_optimizer2, base_lr=1e-7, max_lr=self.lr, step_size_up=self.t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')

    def update(self, batch_size, gamma=0.99, soft_tau=1e-3):
        state, action, last_action, reward, next_state, done, param = self.replay_buffer.sample_sequence(batch_size)

        B,L        = state.shape[:2]
        state_nop  = torch.FloatTensor(state[:,:,:-self.param_dim]).to(self.device)
        next_state_nop = torch.FloatTensor(next_state[:,:,:-self.param_dim]).to(self.device)
        state      = torch.FloatTensor(state[:,-1]).to(self.device)
        next_state = torch.FloatTensor(next_state[:,-1]).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        last_action     = torch.FloatTensor(last_action).to(self.device)
        reward     = torch.FloatTensor(reward[:,-1]).to(self.device)  
        done       = torch.FloatTensor(np.float32(done)[:,-1]).to(self.device)
        param      = torch.FloatTensor(param).to(self.device)

        if len(state_nop.shape)==len(action.shape)+1:
            action = action.unsqueeze(-1)
        if len(state_nop.shape)==len(reward.shape)+1:
            reward = reward.unsqueeze(-1)
        if len(state_nop.shape)==len(done.shape)+1:
            done = done.unsqueeze(-1)

        if "LSTM" in self.rnn_type:
            hidden_in_p = (torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device), \
                        torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
        else:
            hidden_in_p = torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device)

        new_action, log_prob, *_ = self.policy_net.evaluate(state_nop, last_action, hidden_in_p)

        # Entropy loss update
        ent_coef = torch.exp(self.log_ent_coef.detach())
        ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

        with torch.no_grad():
            new_next_action, next_log_prob, *_ = self.target_policy_net.evaluate(next_state_nop, action, hidden_in_p) 
            target_q_1 = self.target_q_net1(next_state, new_next_action)
            target_q_2 = self.target_q_net2(next_state, new_next_action)
            target_q_min = torch.min(target_q_1, target_q_2)
            target_q_min = target_q_min - ent_coef * next_log_prob.reshape(-1,1)
            target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        predicted_q_value1 = self.q_net1(state, action[:,-1])
        predicted_q_value2 = self.q_net2(state, action[:,-1])

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

        # Actor update
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

class SACRNNpolicyfull_Trainer(SACRNNpolicy_Trainer):
    def __init__(self, replay_buffer, 
                        state_space, 
                        action_space, 
                        hidden_dim, 
                        param_dim, 
                        rnn_type='RNNpolicyfull', 
                        out_actf=None, 
                        action_scale=1.0, 
                        device='cpu', 
                        policy_target_update_interval=2, 
                        **kwargs):
        super().__init__(replay_buffer, 
                            state_space, 
                            action_space, 
                            hidden_dim, 
                            param_dim, 
                            rnn_type=rnn_type.strip('full'), 
                            out_actf=out_actf, 
                            action_scale=action_scale,
                            device=device, 
                            policy_target_update_interval=policy_target_update_interval, 
                            **kwargs)

        if 'RNN' in self.rnn_type:
            policy = PolicyNetworkRNNfull
        elif 'LSTM' in self.rnn_type:
            policy = PolicyNetworkLSTMfull
        elif 'GRU' in self.rnn_type:
            policy = PolicyNetworkGRUfull
        else:
            assert NotImplementedError

        self.policy_net = policy(state_space, action_space, hidden_dim, param_dim, device, self.policy_actf, out_actf, action_scale, rnn_dropout=self.rnn_dropout).to(self.device)
        self.target_policy_net = policy(state_space, action_space, hidden_dim, param_dim, device, self.policy_actf, out_actf, action_scale, rnn_dropout=self.rnn_dropout).to(self.device)

        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.lr_scheduler:
            self.scheduler_policy = CyclicLR(self.policy_optimizer, base_lr=1e-7, max_lr=self.lr, step_size_up=self.t_max, step_size_down=None, verbose=False, cycle_momentum=False, mode='triangular2')

    def update(self, batch_size, gamma=0.99, soft_tau=1e-3):
        state, action, last_action, reward, next_state, done, param = self.replay_buffer.sample_sequence(batch_size)

        B,L        = state.shape[:2]
        state_nop  = torch.FloatTensor(state[:,:,:-self.param_dim]).to(self.device)
        next_state_nop = torch.FloatTensor(next_state[:,:,:-self.param_dim]).to(self.device)
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        last_action     = torch.FloatTensor(last_action).to(self.device)
        reward     = torch.FloatTensor(reward).to(self.device)  
        done       = torch.FloatTensor(np.float32(done)).to(self.device)
        param      = torch.FloatTensor(param).to(self.device)

        if len(state_nop.shape)==len(action.shape)+1:
            action = action.unsqueeze(-1)
        if len(state_nop.shape)==len(reward.shape)+1:
            reward = reward.unsqueeze(-1)
        if len(state_nop.shape)==len(done.shape)+1:
            done = done.unsqueeze(-1)

        if "LSTM" in self.rnn_type:
            hidden_in_p = (torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device), \
                        torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
        else:
            hidden_in_p = torch.zeros([1, B, self.hidden_dim], dtype=torch.float).to(self.device)

        new_action, log_prob, *_ = self.policy_net.evaluate(state_nop, last_action, hidden_in_p)

        # Entropy loss update
        ent_coef = torch.exp(self.log_ent_coef.detach())
        ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

        with torch.no_grad():
            new_next_action, next_log_prob, *_ = self.target_policy_net.evaluate(next_state_nop, action, hidden_in_p) 
            target_q_1 = self.target_q_net1(next_state, new_next_action)
            target_q_2 = self.target_q_net2(next_state, new_next_action)
            target_q_min = torch.min(target_q_1, target_q_2)
            target_q_min = target_q_min - ent_coef * next_log_prob
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

        # Actor update
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