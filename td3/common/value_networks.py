import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from .initialize import *


class ValueNetworkBase(nn.Module):
    """ Base network class for value function approximation """
    def __init__(self, state_space, activation, output_activation):
        super(ValueNetworkBase, self).__init__()
        self._state_space = state_space
        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            self._state_dim = self._state_shape[0]
        else:  # high-dim state
            pass  

        self.activation = activation
        self.output_activation = output_activation

    def forward(self):
        pass

class QNetworkBase(ValueNetworkBase):
    def __init__(self, state_space, action_space, activation, output_activation=None):
        super().__init__(state_space, activation, output_activation)
        self._action_space = action_space
        self._action_shape = action_space.shape
        self._action_dim = self._action_shape[0]

class ValueNetwork(ValueNetworkBase):
    def __init__(self, state_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, activation)

        self.linear1 = nn.Linear(self._state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.apply(linear_weights_init)
        self.output_activation = output_activation
        
    def forward(self, state):
        x = self.activation(self.linear1(state))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        if self.output_activation is None:
            x = self.linear4(x)
        else:
            x = self.output_activation(self.linear4(x))
        return x


class QNetwork(QNetworkBase):
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)

        self.linear1 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.apply(linear_weights_init)

        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        if self.output_activation is None:
            x = self.linear4(x)
        else:
            x = self.output_activation(self.linear4(x))
        return x    

class QNetworkParam(QNetworkBase):
    def __init__(self, state_space, action_space, param_dim, hidden_dim,  activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)
        self._param_dim = param_dim

        self.linear1 = nn.Linear(self._state_dim+self._action_dim+self._param_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.apply(linear_weights_init)

        
    def forward(self, state, action, param):
        if len(state.shape)==len(action.shape)+1:
            action = action.unsqueeze(-1)
        x = torch.cat([state, action, param], -1) # the dim 0 is number of samples
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        if self.output_activation is None:
            x = self.linear4(x)
        else:
            x = self.output_activation(self.linear4(x))
        return x       

class QNetworkRNN(QNetworkBase):
    """
    Q network with LSTM structure.
    The network follows two-branch structure as in paper: 
    Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
    """
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_rnn = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.linear3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.apply(linear_weights_init)
        
    def forward(self, state, action, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        assert state.shape[0]==last_action.shape[0], "Batch dimension is not matched, {}, {}".format(state.shape, last_action.shape)
        if len(state.shape)==len(last_action.shape)+1:
            last_action = last_action.unsqueeze(-1)
        if len(state.shape)==len(action.shape)+1:
            action = action.unsqueeze(-1)
        
        if len(state.shape)==2:
            B,L=state.shape[0],1
        elif len(state.shape)==3:
            B,L = state.shape[:2]
        else:
            assert True, "Something wrong"

        # branch 1
        fc_branch = torch.cat([state, action], -1)
        fc_branch = self.activation(self.linear1(fc_branch))
        # fc_branch = self.activation(self.linear2(fc_branch))
        # branch 2
        rnn_branch = torch.cat([state, last_action], -1)
        rnn_branch = self.activation(self.linear_rnn(rnn_branch)).view(B,L,-1)  # linear layer for 3d input only applied on the last dim
        rnn_branch, rnn_hidden = self.rnn(rnn_branch, hidden_in)  # no activation after lstm
        # merged
        rnn_branch = rnn_branch.contiguous().view(*fc_branch.shape)
        merged_branch=torch.cat([fc_branch, rnn_branch], -1) 

        x = self.activation(self.linear3(merged_branch))
        x = self.linear4(x)

        if not L==1:
            x = x.view(B,L,-1)

        return x, rnn_hidden    # lstm_hidden is actually tuple: (hidden, cell)   

class QNetworkLSTM(QNetworkRNN):
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, hidden_dim, activation=activation, output_activation=output_activation)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
class QNetworkGRU(QNetworkRNN):
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, hidden_dim, activation=activation, output_activation=output_activation)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

class QNetworkRNNParam(QNetworkBase):
    """
    Q network with LSTM structure.
    The network follows two-branch structure as in paper: 
    Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
    """
    def __init__(self, state_space, action_space, hidden_dim, param_num, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)
        self.hidden_dim = hidden_dim
        self._param_num = param_num

        self.linear1 = nn.Linear(self._state_dim+self._action_dim+self._param_num, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_rnn = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.linear3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.apply(linear_weights_init)
        
    def forward(self, state, action, last_action, hidden_in, param):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        assert state.shape[0]==last_action.shape[0], "Batch dimension is not matched, {}, {}".format(state.shape, last_action.shape)
        if len(state.shape)==len(last_action.shape)+1:
            last_action = last_action.unsqueeze(-1)
        if len(state.shape)==len(action.shape)+1:
            action = action.unsqueeze(-1)
        
        if len(state.shape)==2:
            B,L=state.shape[0],1
        elif len(state.shape)==3:
            B,L = state.shape[:2]
        else:
            assert True, "Something wrong"

        # branch 1
        fc_branch = torch.cat([state, action, param], -1)
        fc_branch = self.activation(self.linear1(fc_branch))
        # fc_branch = self.activation(self.linear2(fc_branch))
        # branch 2
        rnn_branch = torch.cat([state, last_action], -1)
        rnn_branch = self.activation(self.linear_rnn(rnn_branch)).view(B,L,-1)  # linear layer for 3d input only applied on the last dim
        rnn_branch, rnn_hidden = self.rnn(rnn_branch, hidden_in)  # no activation after lstm
        # merged
        rnn_branch = rnn_branch.contiguous().view(*fc_branch.shape)
        merged_branch=torch.cat([fc_branch, rnn_branch], -1) 

        x = self.activation(self.linear3(merged_branch))
        x = self.linear4(x)

        if not L==1:
            x = x.view(B,L,-1)

        return x, rnn_hidden    # lstm_hidden is actually tuple: (hidden, cell)   

class QNetworkLSTMParam(QNetworkRNNParam):
    def __init__(self, state_space, action_space, hidden_dim, param_num, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, hidden_dim, param_num, activation=activation, output_activation=output_activation)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
class QNetworkGRUParam(QNetworkRNNParam):
    def __init__(self, state_space, action_space, hidden_dim, param_num, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, hidden_dim, param_num, activation=activation, output_activation=output_activation)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)