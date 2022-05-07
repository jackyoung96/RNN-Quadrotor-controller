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
    def __init__(self, state_space, action_space, hidden_size, device, out_actf=None, action_scale=1.0, init_w=3e-3, log_std_min=np.exp(-20), log_std_max=np.exp(2)):
        super().__init__(state_space, action_space, device)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(self._state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, self._action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, self._action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.out_actf = out_actf
        self.action_scale = action_scale
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mean = self.mean_linear(x)
        if not self.out_actf is None:
            mean = self.out_actf(mean)
        std = F.softplus(self.log_std_linear(x))
        std = torch.clamp(std, self.log_std_min, self.log_std_max)

        return self.action_scale * mean, self.action_scale * std
    
    def evaluate(self, state, deterministic, eval_noise_scale):
        '''
        generate action with state as input wrt the policy network, for calculating gradients
        '''
        num_batch = 1 if len(state.shape) < 2 else state.shape[0]
        mean, std = self.forward(state)
        
        ''' add noise '''
        normal = Normal(0, 1)
        z = normal.sample((num_batch,1))
        action_0 = mean + std * z.to(self.device)
        action = mean if deterministic else action_0
        
        log_prob = Normal(mean, std).log_prob(action_0)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        eval_noise_clip = 2*eval_noise_scale
        noise = normal.sample(action.shape) * eval_noise_scale
        noise = torch.clamp(
        noise,
        -eval_noise_clip,
        eval_noise_clip)
        action = action + noise.to(self.device)

        return action, log_prob, z, mean, std
        
    
    def get_action(self, state, deterministic, explore_noise_scale):
        '''
        generate action for interaction with env
        '''
        num_batch = 1 if len(state.shape) < 2 else state.shape[0]
        state = torch.FloatTensor(state).to(self.device)
        mean, std = self.forward(state)
        normal = Normal(0,1)
        z = normal.sample((num_batch,1)).to(self.device)

        action = mean.detach().cpu().numpy().squeeze() if deterministic\
                else (mean + std*z).detach().cpu().numpy().squeeze()

        ''' add noise '''
        noise = normal.sample(action.shape) * explore_noise_scale
        action = action + noise.numpy()

        return action

class PolicyNetworkRNN(PolicyNetworkBase):
    def __init__(self, state_space, action_space, hidden_size, device, out_actf=None, action_scale=1.0, init_w=3e-3, log_std_min=np.exp(-20), log_std_max=np.exp(2)):
        super().__init__(state_space, action_space, device)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(self._state_dim, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear_rnn = nn.Linear(self._state_dim+self._action_dim, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.linear3 = nn.Linear(2*hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, self._action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, self._action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.out_actf = out_actf
        self.action_scale = action_scale
        
    def forward(self, state, last_action, hidden_in):
        assert state.shape[0]==last_action.shape[0], "Batch dimension is not matched, {}, {}".format(state.shape, last_action.shape)
        if len(state.shape)==len(last_action.shape)+1:
            last_action = last_action.unsqueeze(-1)
        
        if len(state.shape)==2:
            B,L=state.shape[0],1
        elif len(state.shape)==3:
            B,L = state.shape[:2]
        else:
            assert True, "Something wrong"

        fc_x = F.relu(self.linear1(state))  
        # fc_x = F.relu(self.linear2(fc_x)) 
        sa_cat = torch.cat([state,last_action], dim=-1)
        rnn_x = F.relu(self.linear_rnn(sa_cat)).view(B,L,-1)
        rnn_out, rnn_hidden = self.rnn(rnn_x, hidden_in)
        rnn_x = rnn_out.contiguous().view(*fc_x.shape)
        merged_x = torch.cat([fc_x, rnn_x],dim=-1)
        x = F.relu(self.linear3(merged_x))
        x = F.relu(self.linear4(x))
        mean = self.mean_linear(x)
        if not self.out_actf is None:
            mean = self.out_actf(mean)
        std = F.softplus(self.log_std_linear(x))
        std = torch.clamp(std, self.log_std_min, self.log_std_max)

        if not L==1:
            mean = mean.view(B,L,-1)
            std = std.view(B,L,-1)

        return self.action_scale * mean, self.action_scale * std, rnn_hidden, rnn_out
    
    def evaluate(self, state, last_action, hidden_in, \
                deterministic, eval_noise_scale):
        '''
        generate action with state as input wrt the policy network, for calculating gradients
        '''
        num_batch = 1 if len(state.shape) < 2 else state.shape[0]
        mean, std, hidden_out, hidden_all = self.forward(state, last_action, hidden_in)
        
        ''' add noise '''
        normal = Normal(0, 1)
        z = normal.sample(std.shape)
        action_0 = mean + std * z.to(self.device)
        action = mean if deterministic else action_0
        
        log_prob = Normal(mean, std).log_prob(action_0)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        eval_noise_clip = 2*eval_noise_scale
        noise = normal.sample(action.shape) * eval_noise_scale
        noise = torch.clamp(
        noise,
        -eval_noise_clip,
        eval_noise_clip)
        action = action + noise.to(self.device)

        return action, hidden_out, hidden_all, log_prob, z, mean, std
        
    
    def get_action(self, state, last_action, hidden_in,\
                    deterministic, explore_noise_scale):
        '''
        generate action for interaction with env
        '''
        num_batch = 1 if len(state.shape) < 2 else state.shape[0]
        state = torch.FloatTensor(state).to(self.device)
        last_action = torch.FloatTensor(last_action).to(self.device)
        mean, std, hidden_out, hidden_all = self.forward(state, last_action, hidden_in)

        # print("debug",state[0],hidden_in[0], mean[0], std[0])

        normal = Normal(0,1)
        z = normal.sample(std.shape).to(self.device)

        action = mean.detach().cpu().numpy().squeeze() if deterministic\
                else (mean + std*z).detach().cpu().numpy().squeeze()

        # print("debug get action", mean.squeeze(), action)

        ''' add noise '''
        noise = normal.sample(action.shape) * explore_noise_scale
        action = action + noise.numpy()

        return action, hidden_out

class PolicyNetworkLSTM(PolicyNetworkRNN):
    def __init__(self, state_space, action_space, hidden_size, device, out_actf=None, action_scale=1.0, init_w=3e-3, log_std_min=np.exp(-20), log_std_max=np.exp(2)):
        super().__init__(state_space, action_space, hidden_size, device, out_actf=out_actf, action_scale=action_scale, init_w=init_w, log_std_min=log_std_min, log_std_max=log_std_max)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)

class PolicyNetworkGRU(PolicyNetworkRNN):
    def __init__(self, state_space, action_space, hidden_size, device, out_actf=None, action_scale=1.0, init_w=3e-3, log_std_min=np.exp(-20), log_std_max=np.exp(2)):
        super().__init__(state_space, action_space, hidden_size, device, out_actf=out_actf, action_scale=action_scale, init_w=init_w, log_std_min=log_std_min, log_std_max=log_std_max)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)


class PolicyNetworkGoalRNN(PolicyNetworkBase):
    def __init__(self, state_space, action_space, hidden_size, goal_dim, device, out_actf=None, action_scale=1.0, init_w=3e-3, log_std_min=np.exp(-20), log_std_max=np.exp(2)):
        super().__init__(state_space, action_space, device)
        self._goal_dim = goal_dim
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(self._state_dim+self._goal_dim, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear_rnn = nn.Linear(self._state_dim+self._action_dim, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.linear3 = nn.Linear(2*hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, self._action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, self._action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.out_actf = out_actf
        self.action_scale = action_scale
        
    def forward(self, state, last_action, hidden_in, goal):
        assert state.shape[0]==last_action.shape[0], "Batch dimension is not matched, {}, {}".format(state.shape, last_action.shape)
        if len(state.shape)==len(last_action.shape)+1:
            last_action = last_action.unsqueeze(-1)
        
        if len(state.shape)==2:
            B,L=state.shape[0],1
        elif len(state.shape)==3:
            B,L = state.shape[:2]
        else:
            assert True, "Something wrong"

        sg_cat = torch.cat([state,goal], dim=-1)
        fc_x = F.relu(self.linear1(sg_cat))  
        # fc_x = F.relu(self.linear2(fc_x)) 
        sa_cat = torch.cat([state,last_action], dim=-1)
        rnn_x = F.relu(self.linear_rnn(sa_cat)).view(B,L,-1)
        rnn_out, rnn_hidden = self.rnn(rnn_x, hidden_in)
        rnn_x = rnn_out.contiguous().view(*fc_x.shape)
        merged_x = torch.cat([fc_x, rnn_x],dim=-1)
        x = F.relu(self.linear3(merged_x))
        x = F.relu(self.linear4(x))
        mean = self.mean_linear(x)
        if not self.out_actf is None:
            mean = self.out_actf(mean)
        std = F.softplus(self.log_std_linear(x))
        std = torch.clamp(std, self.log_std_min, self.log_std_max)

        if not L==1:
            mean = mean.view(B,L,-1)
            std = std.view(B,L,-1)

        return self.action_scale * mean, self.action_scale * std, rnn_hidden, rnn_out
    
    def evaluate(self, state, last_action, hidden_in, goal,\
                deterministic, eval_noise_scale):
        '''
        generate action with state as input wrt the policy network, for calculating gradients
        '''
        num_batch = 1 if len(state.shape) < 2 else state.shape[0]
        mean, std, hidden_out, hidden_all = self.forward(state, last_action, hidden_in, goal)
        
        ''' add noise '''
        normal = Normal(0, 1)
        z = normal.sample(std.shape)
        action_0 = mean + std * z.to(self.device)
        action = mean if deterministic else action_0
        
        log_prob = Normal(mean, std).log_prob(action_0)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        eval_noise_clip = 2*eval_noise_scale
        noise = normal.sample(action.shape) * eval_noise_scale
        noise = torch.clamp(
        noise,
        -eval_noise_clip,
        eval_noise_clip)
        action = action + noise.to(self.device)

        return action, hidden_out, hidden_all, log_prob, z, mean, std
        
    
    def get_action(self, state, last_action, hidden_in, goal,\
                    deterministic, explore_noise_scale):
        '''
        generate action for interaction with env
        '''
        num_batch = 1 if len(state.shape) < 2 else state.shape[0]
        state = torch.FloatTensor(state).to(self.device)
        last_action = torch.FloatTensor(last_action).to(self.device)
        goal = torch.FloatTensor(goal).to(self.device)
        mean, std, hidden_out, hidden_all = self.forward(state, last_action, hidden_in, goal)

        # print("debug",state[0],hidden_in[0], mean[0], std[0])

        normal = Normal(0,1)
        z = normal.sample(std.shape).to(self.device)

        action = mean.detach().cpu().numpy().squeeze() if deterministic\
                else (mean + std*z).detach().cpu().numpy().squeeze()

        # print("debug get action", mean.squeeze(), action)

        ''' add noise '''
        noise = normal.sample(action.shape) * explore_noise_scale
        action = action + noise.numpy()

        return action, hidden_out

class PolicyNetworkLSTM(PolicyNetworkRNN):
    def __init__(self, state_space, action_space, hidden_size, goal_dim, device, out_actf=None, action_scale=1.0, init_w=3e-3, log_std_min=np.exp(-20), log_std_max=np.exp(2)):
        super().__init__(state_space, action_space, hidden_size, goal_dim, device, out_actf=out_actf, action_scale=action_scale, init_w=init_w, log_std_min=log_std_min, log_std_max=log_std_max)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)

class PolicyNetworkGRU(PolicyNetworkRNN):
    def __init__(self, state_space, action_space, hidden_size, goal_dim, device, out_actf=None, action_scale=1.0, init_w=3e-3, log_std_min=np.exp(-20), log_std_max=np.exp(2)):
        super().__init__(state_space, action_space, hidden_size, goal_dim, device, out_actf=out_actf, action_scale=action_scale, init_w=init_w, log_std_min=log_std_min, log_std_max=log_std_max)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)