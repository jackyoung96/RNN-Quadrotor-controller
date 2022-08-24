import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
from .initialize import *

class PolicyNetworkBase(nn.Module):
    """ Base network class for policy function """
    def __init__(self, state_space, action_space, device):
        super(PolicyNetworkBase, self).__init__()
        self._state_space = state_space
        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            self._state_dim = self._state_shape[0]
        else:  # high-dim state 
            pass  
        self._action_space = action_space
        self._action_shape = action_space.shape
        if len(self._action_shape) < 1:  # Discrete space
            self._action_dim = action_space.n
        else:
            self._action_dim = self._action_shape[0]
        

        self.device = device

    def forward(self):
        pass
    
    def evaluate(self):
        pass 
    
    def get_action(self):
        pass

    def sample_action(self,):
        a=torch.FloatTensor(self._action_dim).uniform_(self._action_space.low, self._action_space.high)
        return a.numpy()
        
class PolicyNetwork(PolicyNetworkBase):
    def __init__(self, state_space, action_space, hidden_size, device, actf=F.relu, out_actf=F.tanh, action_scale=1.0, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super().__init__(state_space, action_space, device)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(self._state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, self._action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, self._action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.actf = actf
        self.out_actf = out_actf
        self.action_scale = action_scale
        
    def forward(self, state):
        x = self.actf(self.linear1(state))
        x = self.actf(self.linear2(x))
        x = self.actf(self.linear3(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return self.action_scale * mean, self.action_scale * log_std
    
    def evaluate(self, state):
        '''
        generate action with state as input wrt the policy network, for calculating gradients
        '''
        if len(state.shape)==1:
            state = state[None,:]
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        ''' add noise '''
        normal = Normal(0, 1)
        z = normal.sample(log_std.shape)
        gaussian_action = mean + std * z.to(self.device)
        action = self.out_actf(gaussian_action)
        
        log_prob = Normal(mean, std).log_prob(gaussian_action)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        # Squash correction
        log_prob -= torch.sum(torch.log(1-action**2 + 1e-6), dim=1, keepdim=True)

        return action, log_prob, z, mean, std
        
    
    def get_action(self, state, deterministic=False):
        '''
        generate action for interaction with env
        '''
        if len(state.shape)==1:
            state = state[None,:]
        state = torch.FloatTensor(state).to(self.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(0,1)
        z = normal.sample(log_std.shape).to(self.device)

        gaussian_action = mean if deterministic else (mean + std*z)
        action = self.out_actf(gaussian_action).detach().cpu().numpy()

        return action

class PolicyNetworkRNN(PolicyNetworkBase):
    def __init__(self, state_space, action_space, hidden_size, param_dim, device, actf=F.relu, out_actf=F.tanh, action_scale=1.0, init_w=3e-3, log_std_min=-20, log_std_max=2, rnn_dropout=0.5):
        super().__init__(state_space, action_space, device)
        self.param_dim = param_dim
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(self._state_dim-self.param_dim, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear_rnn = nn.Linear(self._state_dim+self._action_dim-self.param_dim, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.rnn_dropout = nn.Dropout(p=rnn_dropout)
        self.linear2 = nn.Linear(2*hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, self._action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, self._action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.actf = actf
        self.out_actf = out_actf
        self.action_scale = action_scale
        
    def forward(self, state, last_action, hidden_in):
        if len(state.shape)+1==len(last_action.shape):
            state = state.unsqueeze(0)
        elif len(state.shape)==len(last_action.shape)+1:
            last_action = last_action.unsqueeze(-1)
        else:
            assert True, "Something wrong"
        
        if len(state.shape)==2:
            B,L=state.shape[0],1
        elif len(state.shape)==3:
            B,L = state.shape[:2]
        else:
            assert True, "Something wrong"

        if len(state.shape)==2:
            fc_x = self.actf(self.linear1(state))
        else:
            fc_x = self.actf(self.linear1(state[:,-1]))
        # fc_x = F.relu(self.linear2(fc_x)) 
        sa_cat = torch.cat([state,last_action], dim=-1)
        rnn_x = self.actf(self.linear_rnn(sa_cat)).view(B,L,-1)
        rnn_out, rnn_hidden = self.rnn(rnn_x, hidden_in)
        # Use only last result
        rnn_x = self.rnn_dropout(rnn_out[:,-1].contiguous().view(*fc_x.shape)) # Dropout for make RNN weaker
        merged_x = torch.cat([fc_x, rnn_x],dim=-1)
        x = self.actf(self.linear2(merged_x))
        x = self.actf(self.linear3(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return self.action_scale * mean, self.action_scale * log_std, rnn_out, rnn_hidden
    
    def evaluate(self, state, last_action, hidden_in):
        '''
        generate action with state as input wrt the policy network, for calculating gradients
        '''
        if len(state.shape)==1:
            state = state[None,:]
        mean, log_std, hidden_all, hidden_out = self.forward(state, last_action, hidden_in)
        
        ''' add noise '''
        normal = Normal(0, 1)
        z = normal.sample(log_std.shape).to(self.device)
        std = log_std.exp()
        gaussian_action = mean + std * z
        action = self.out_actf(gaussian_action)
        
        log_prob = Normal(mean, std).log_prob(gaussian_action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # Squash correction
        log_prob -= torch.sum(torch.log(1-action**2 + 1e-6), dim=-1, keepdim=True)

        return action, log_prob, hidden_out, hidden_all, z, mean, std
        
    
    def get_action(self, state, last_action, hidden_in, deterministic=False):
        '''
        generate action for interaction with env
        '''
        if len(state.shape)==1:
            state = state[None,:]
        state = torch.FloatTensor(state).to(self.device)
        last_action = torch.FloatTensor(last_action).to(self.device)
        mean, log_std, hidden_all, hidden_out = self.forward(state, last_action, hidden_in)

        # print("debug",state[0],hidden_in[0], mean[0], std[0])

        normal = Normal(0,1)
        z = normal.sample(log_std.shape).to(self.device)
        std = log_std.exp()
        gaussian_action = mean if deterministic else mean + std * z
        action = self.out_actf(gaussian_action).detach().cpu().numpy()

        return action, hidden_out

class PolicyNetworkLSTM(PolicyNetworkRNN):
    def __init__(self, state_space, action_space, hidden_size, device, actf=F.relu, out_actf=F.tanh, action_scale=1.0, init_w=3e-3, log_std_min=-20, log_std_max=2, rnn_dropout=0.5):
        super().__init__(state_space, action_space, hidden_size, device, actf=actf, out_actf=out_actf, action_scale=action_scale, init_w=init_w, log_std_min=log_std_min, log_std_max=log_std_max, rnn_dropout=rnn_dropout)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)

class PolicyNetworkGRU(PolicyNetworkRNN):
    def __init__(self, state_space, action_space, hidden_size, device, actf=F.tanh, out_actf=F.tanh, action_scale=1.0, init_w=3e-3, log_std_min=-20, log_std_max=2, rnn_dropout=0.5):
        super().__init__(state_space, action_space, hidden_size, device, actf=actf, out_actf=out_actf, action_scale=action_scale, init_w=init_w, log_std_min=log_std_min, log_std_max=log_std_max, rnn_dropout=rnn_dropout)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)