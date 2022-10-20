import math
import random
import numpy as np
from pkg_resources import Environment
import torch

from .utils import rot_matrix_similarity, rot_matrix_z_similarity


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, last_action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def push_batch(self, state, action, last_action, reward, next_state, done):
        state, action, last_action, reward, next_state, done = \
            map(lambda x: np.stack(x, axis=0),[state, action, last_action, reward, next_state, done])
        assert all(state.shape[0] == x.shape[0] for x in [action, last_action, reward, next_state, done]), "Somethings wrong on dimension"
        for s,a,la,r,ns,d in zip(state, action, last_action, reward, next_state, done):
            self.push(s,a,la,r,ns,d)

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst=[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            state, action, last_action, reward, next_state, done = sample
            s_lst.append(state) 
            a_lst.append(action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
        
        s_lst, a_lst, r_lst, ns_lst, d_lst = map(lambda x: np.stack(x,axis=0), [s_lst, a_lst, r_lst, ns_lst, d_lst])

        return s_lst, a_lst, r_lst, ns_lst, d_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)


class ReplayBufferRNN:
    def __init__(self, capacity, param_dim, **kwargs):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.seq_length = kwargs.get("seq_length", 100)
        self.capacity = self.capacity // self.seq_length
        self.param_dim = param_dim

    def push(self, state, action, last_action, reward, next_state, done, param):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, last_action, reward, next_state, done, param)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def push_batch(self, state, action, last_action, reward, next_state, done):
        B,L = state[0].shape[0], len(state)
        state, action, last_action, reward, next_state, done = \
            map(lambda x: np.stack(x, axis=0),[state, action, last_action, reward, next_state, done])
        assert all(state.shape[0] == x.shape[0] for x in [action, last_action, reward, next_state, done]), "Somethings wrong on dimension"
        
        # Extract param from observation 
        param = state[:,-self.param_dim:]
        
        for i in range(0,L+1-self.seq_length,self.seq_length):
            self.push(*map(lambda x:x[i:i+self.seq_length],[state, action, last_action, reward, next_state, done, param]))

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst=[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            state, action, last_action, reward, next_state, done, param = sample
            idx = random.randint(0,self.seq_length-1)
            s_lst.append(state[idx]) 
            a_lst.append(action[idx])
            la_lst.append(last_action[idx])
            r_lst.append(reward[idx])
            ns_lst.append(next_state[idx])
            d_lst.append(done[idx])
            p_lst.append(param[idx])
        
        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst = map(lambda x: np.stack(x,axis=0), [s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst])

        return s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst
    
    def sample_sequence(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst=[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            state, action, last_action, reward, next_state, done, param = sample
            s_lst.append(state) 
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            p_lst.append(param)
        
        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst = map(lambda x: np.stack(x,axis=0), [s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst])

        return s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)
