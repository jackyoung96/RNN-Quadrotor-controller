import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
from .initialize import *

class ParamNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, param_dim, hidden_dim, rnn_type='RNN', actf=F.relu, output_actf=F.tanh, rnn_dropout=0.5):
        super(ParamNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.actf = actf
        self.output_actf = output_actf

        self.linear1 = nn.Linear(self.state_dim+self.action_dim, self.hidden_dim)
        self.rnn = getattr(nn, self.rnn_type)(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.rnn_dropout = nn.Dropout(p=rnn_dropout)
        self.linear2 = nn.Linear(self.hidden_dim, self.param_dim)

    def forward(self, state, action, hidden_in=None):
        # shape 확인할 것
        # raise NotImplementedError

        sa = torch.cat([state, action], -1)
        x = self.actf(self.linear1(sa))
        if hidden_in is None:
            x, hidden_out = self.rnn(x, hidden_in)
        else:
            x, hidden_out = self.rnn(x)
        x = self.rnn_dropout(x.contiguous())
        out = self.output_actf(self.linear2(x))

        return out, hidden_out
