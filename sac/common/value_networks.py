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
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.linearout = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linearout.apply(linear_weights_init)
        self.output_activation = output_activation
        
    def forward(self, state):
        x = self.activation(self.linear1(state))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))
        if self.output_activation is None:
            x = self.linearout(x)
        else:
            x = self.output_activation(self.linearout(x))
        return x


class QNetwork(QNetworkBase):
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)

        self.linear1 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.linearout = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linearout.apply(linear_weights_init)

        
    def forward(self, state, action):
        x = torch.cat([state, action], -1) # the dim 0 is number of samples
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))
        if self.output_activation is None:
            x = self.linearout(x)
        else:
            x = self.output_activation(self.linearout(x))
        return x    

class QNetworkParam(QNetworkBase):
    def __init__(self, state_space, action_space, param_dim, hidden_dim,  activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)
        self._param_dim = param_dim

        self.linear1 = nn.Linear(self._state_dim+self._action_dim+self._param_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.linearout = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linearout.apply(linear_weights_init)

        
    def forward(self, state, action, param):
        if len(state.shape)==len(action.shape)+1:
            action = action.unsqueeze(-1)
        x = torch.cat([state, action, param], -1) # the dim 0 is number of samples
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))
        if self.output_activation is None:
            x = self.linearout(x)
        else:
            x = self.output_activation(self.linearout(x))
        return x    

class QNetworkRNN(QNetworkBase):
    """
    Q network with LSTM structure.
    The network follows two-branch structure as in paper: 
    Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
    """
    def __init__(self, state_space, action_space, hidden_dim, param_dim, activation=F.relu, output_activation=None, rnn_dropout=0.5):
        super().__init__(state_space, action_space, activation)
        self.hidden_dim = hidden_dim
        self.param_dim = param_dim

        self.linear1 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_rnn = nn.Linear(self._state_dim+self._action_dim-self.param_dim, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.rnn_dropout = nn.Dropout(p=rnn_dropout)
        self.linear2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)

        self.linearout = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linearout.apply(linear_weights_init)
        
    def forward(self, state, action, param, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        if len(state.shape)==len(last_action.shape)+1:
            last_action = last_action.unsqueeze(-1)
        if len(state.shape)==len(action.shape)+1:
            action = action.unsqueeze(1)
        
        if len(state.shape)==2:
            B,L=state.shape[0],1
        elif len(state.shape)==3:
            B,L = state.shape[:2]
        else:
            assert True, "Something wrong"

        # branch 1 -> Only use last state, action, param
        fc_branch = torch.cat([state[:,-1], action[:,-1], param[:,-1]], -1)
        fc_branch = self.activation(self.linear1(fc_branch))

        # branch 2
        rnn_branch = torch.cat([state, last_action], -1)
        rnn_branch = self.activation(self.linear_rnn(rnn_branch)).view(B,L,-1)  # linear layer for 3d input only applied on the last dim
        rnn_branch, rnn_hidden = self.rnn(rnn_branch, hidden_in)  # no activation after lstm
        # merged
        rnn_branch = self.rnn_dropout(rnn_branch[:,-1].contiguous().view(*fc_branch.shape))
        merged_branch=torch.cat([fc_branch, rnn_branch], -1) 

        x = self.activation(self.linear2(merged_branch))
        x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))
        x = self.linearout(x)

        return x, rnn_hidden    # lstm_hidden is actually tuple: (hidden, cell)   

class QNetworkLSTM(QNetworkRNN):
    def __init__(self, state_space, action_space, hidden_dim, param_dim, activation=F.relu, output_activation=None, rnn_dropout=0.5):
        super().__init__(state_space, action_space, hidden_dim, param_dim, activation=activation, output_activation=output_activation, rnn_dropout=rnn_dropout)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
class QNetworkGRU(QNetworkRNN):
    def __init__(self, state_space, action_space, hidden_dim, param_dim, activation=F.relu, output_activation=None, rnn_dropout=0.5):
        super().__init__(state_space, action_space, hidden_dim, param_dim, activation=activation, output_activation=output_activation, rnn_dropout=rnn_dropout)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)


class QNetworkRNNfull(QNetworkRNN):
    """
    Q network with LSTM structure.
    The network follows two-branch structure as in paper: 
    Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
    """
    def __init__(self, state_space, action_space, hidden_dim, param_dim, activation=F.relu, output_activation=None, rnn_dropout=0.5):
        super().__init__(state_space, action_space, hidden_dim, param_dim, activation=activation, output_activation=output_activation, rnn_dropout=rnn_dropout)
        
    def forward(self, state, action, param, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """

        if len(state.shape)==1:
            B,L=1,1
        elif len(state.shape)==2:
            B,L=state.shape[0],1
        elif len(state.shape)==3:
            B,L = state.shape[:2]
        else:
            assert True, "Something wrong"

        # branch 1 -> Only use last state, action, param
        fc_branch = torch.cat([state, action, param], -1)
        fc_branch = self.activation(self.linear1(fc_branch))

        # branch 2
        rnn_branch = torch.cat([state, last_action], -1)
        rnn_branch = self.activation(self.linear_rnn(rnn_branch)).view(B,L,-1)  # linear layer for 3d input only applied on the last dim
        rnn_branch, rnn_hidden = self.rnn(rnn_branch, hidden_in)  # no activation after lstm
        # merged
        rnn_branch = self.rnn_dropout(rnn_branch.contiguous().view(*fc_branch.shape))
        merged_branch=torch.cat([fc_branch, rnn_branch], -1) 

        x = self.activation(self.linear2(merged_branch))
        x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))
        x = self.linearout(x)

        return x, rnn_hidden    # lstm_hidden is actually tuple: (hidden, cell)   

class QNetworkLSTMfull(QNetworkRNNfull):
    def __init__(self, state_space, action_space, hidden_dim, param_dim, activation=F.relu, output_activation=None,rnn_dropout=0.5):
        super().__init__(state_space, action_space, hidden_dim, param_dim, activation=activation, output_activation=output_activation, rnn_dropout=rnn_dropout)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
class QNetworkGRUfull(QNetworkRNNfull):
    def __init__(self, state_space, action_space, hidden_dim, param_dim, activation=F.relu, output_activation=None,rnn_dropout=0.5):
        super().__init__(state_space, action_space, hidden_dim, param_dim, activation=activation, output_activation=output_activation, rnn_dropout=rnn_dropout)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)