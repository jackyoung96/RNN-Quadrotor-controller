import math
import random
import numpy as np
from pkg_resources import Environment
import torch

from .utils import rot_matrix_similarity



class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def push_batch(self, state, action, reward, next_state, done):
        assert all(state.shape[0] == x.shape[0] for x in [action, reward, next_state, done]), "Somethings wrong on dimension"
        num_batch = state.shape[0]
        self.buffer.extend([None]*int(min(num_batch,self.capacity-len(self.buffer))))
        for s,a,r,ns,d in zip(state,action,reward,next_state,done):
            self.buffer[self.position] = (s,a,r,ns,d)
            self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack,
                                                      zip(*batch))  # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)

class ReplayBufferPER:
    """ 
    Replay buffer with Prioritized Experience Replay (PER),
    TD error as sampling weights. This is a simple version without sumtree.

    Reference:
    https://github.com/Felhof/DiscreteSAC/blob/main/utilities/ReplayBuffer.py
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.weights = np.zeros(int(capacity))
        self.max_weight = 10**-2
        self.delta = 10**-4

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.weights[self.position] = self.max_weight  # new sample has max weights

        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def push_batch(self, state, action, reward, next_state, done):
        assert all(state.shape[0] == x.shape[0] for x in [action, reward, next_state, done]), "Somethings wrong on dimension"
        num_batch = state.shape[0]
        self.buffer.extend([None]*int(min(num_batch,self.capacity-len(self.buffer))))
        for s,a,r,ns,d in zip(state,action,reward,next_state,done):
            self.buffer[self.position] = (s,a,r,ns,d)
            self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        set_weights = self.weights[:self.position] + self.delta
        probabilities = set_weights / sum(set_weights)
        self.indices = np.random.choice(range(self.position), batch_size, p=probabilities, replace=False)
        batch = np.array(self.buffer)[list(self.indices)]
        state, action, reward, next_state, done = map(np.stack,
                                                      zip(*batch))  # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done

    def update_weights(self, prediction_errors):
        max_error = max(prediction_errors)
        self.max_weight = max(self.max_weight, max_error)
        self.weights[self.indices] = prediction_errors

    def __len__(
            self):  # this is a stupid func! cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)


class ReplayBufferLSTM:
    """ 
    Replay buffer for agent with LSTM network additionally storing previous action, 
    initial input hidden state and output hidden state of LSTM.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for LSTM initialization.

    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (hidden_in, hidden_out, state, action, last_action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def push_batch(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done):
        state, action, last_action, reward, next_state, done = \
            map(lambda x: np.stack(x, axis=1),[state, action, last_action, reward, next_state, done])
        assert all(state.shape[0] == x.shape[0] for x in [action, last_action, reward, next_state, done]), "Somethings wrong on dimension"
        num_batch = state.shape[0]
        hidden_in = (hidden_in[0].view(num_batch,1,1,-1), hidden_in[1].view(num_batch,1,1,-1))
        hidden_out = (hidden_out[0].view(num_batch,1,1,-1), hidden_out[1].view(num_batch,1,1,-1))
        self.buffer.extend([None]*int(min(num_batch,self.capacity-len(self.buffer))))
        for hin_h,hin_c, hout_h,hout_c, s,a,la,r,ns,d in zip(*hidden_in, *hidden_out, state, action, last_action, reward, next_state, done):
            self.buffer[self.position] = ((hin_h,hin_c),(hout_h,hout_c),s,a,la,r,ns,d)
            self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ci_lst, ho_lst, co_lst, d_lst=[],[],[],[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            (h_in, c_in), (h_out, c_out), state, action, last_action, reward, next_state, done = sample
            s_lst.append(state) 
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, hidden_size)
            ci_lst.append(c_in)
            ho_lst.append(h_out)
            co_lst.append(c_out)
        hi_lst = torch.cat(hi_lst, dim=-2).detach() # cat along the batch dim
        ho_lst = torch.cat(ho_lst, dim=-2).detach()
        ci_lst = torch.cat(ci_lst, dim=-2).detach()
        co_lst = torch.cat(co_lst, dim=-2).detach()

        hidden_in = (hi_lst, ci_lst)
        hidden_out = (ho_lst, co_lst)
        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst = map(lambda x: np.stack(x,axis=0), [s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst])
        return hidden_in, hidden_out, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)

class ReplayBufferGRU:
    """ 
    Replay buffer for agent with LSTM network additionally storing previous action, 
    initial input hidden state and output hidden state of LSTM.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for LSTM initialization.

    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (hidden_in, hidden_out, state, action, last_action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def push_batch(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done):
        state, action, last_action, reward, next_state, done = \
            map(lambda x: np.stack(x, axis=1),[state, action, last_action, reward, next_state, done])
        assert all(state.shape[0] == x.shape[0] for x in [action, last_action, reward, next_state, done]), "Somethings wrong on dimension"
        num_batch = state.shape[0]
        hidden_in = hidden_in.view(num_batch,1,1,-1)
        hidden_out = hidden_out.view(num_batch,1,1,-1)
        self.buffer.extend([None]*int(min(num_batch,self.capacity-len(self.buffer))))
        for hin, hout, s,a,la,r,ns,d in zip(hidden_in, hidden_out, state, action, last_action, reward, next_state, done):
            self.buffer[self.position] = (hin,hout,s,a,la,r,ns,d)
            self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ho_lst, d_lst=[],[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            h_in, h_out, state, action, last_action, reward, next_state, done = sample
            s_lst.append(state) 
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, hidden_size)
            ho_lst.append(h_out)
        hidden_in = torch.cat(hi_lst, dim=-2).detach() # cat along the batch dim
        hidden_out = torch.cat(ho_lst, dim=-2).detach()

        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst = map(lambda x: np.stack(x,axis=0), [s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst])
        return hidden_in, hidden_out, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)


class ReplayBufferFastAdaptLSTM:
    """ 
    Replay buffer for agent with LSTM network additionally storing previous action, 
    initial input hidden state and output hidden state of LSTM.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for LSTM initialization.

    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done, param):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (hidden_in, hidden_out, state, action, last_action, reward, next_state, done, param)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def push_batch(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done, param):
        np.stack(last_action,axis=1)
        state, action, last_action, reward, next_state, done = \
            map(lambda x: np.stack(x, axis=1),[state, action, last_action, reward, next_state, done])
        assert all(state.shape[0] == x.shape[0] for x in [action, last_action, reward, next_state, done]), "Somethings wrong on dimension"
        num_batch = state.shape[0]
        hidden_in = (hidden_in[0].view(num_batch,1,1,-1), hidden_in[1].view(num_batch,1,1,-1))
        hidden_out = (hidden_out[0].view(num_batch,1,1,-1), hidden_out[1].view(num_batch,1,1,-1))
        self.buffer.extend([None]*int(min(num_batch,self.capacity-len(self.buffer))))
        for hin_h,hin_c, hout_h,hout_c, s,a,la,r,ns,d,p in zip(*hidden_in, *hidden_out, state, action, last_action, reward, next_state, done,param):
            self.buffer[self.position] = ((hin_h,hin_c),(hout_h,hout_c),s,a,la,r,ns,d,p)
            self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ci_lst, ho_lst, co_lst, d_lst, p_lst=[],[],[],[],[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            (h_in, c_in), (h_out, c_out), state, action, last_action, reward, next_state, done, param = sample
            s_lst.append(state) 
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, hidden_size)
            ci_lst.append(c_in)
            ho_lst.append(h_out)
            co_lst.append(c_out)
            p_lst.append(param)
        hi_lst = torch.cat(hi_lst, dim=-2).detach() # cat along the batch dim
        ho_lst = torch.cat(ho_lst, dim=-2).detach()
        ci_lst = torch.cat(ci_lst, dim=-2).detach()
        co_lst = torch.cat(co_lst, dim=-2).detach()

        hidden_in = (hi_lst, ci_lst)
        hidden_out = (ho_lst, co_lst)
        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst = map(lambda x: np.stack(x,axis=0), [s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst])

        return hidden_in, hidden_out, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)



class ReplayBufferFastAdaptGRU:
    """ 
    Replay buffer for agent with LSTM network additionally storing previous action, 
    initial input hidden state and output hidden state of LSTM.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for LSTM initialization.

    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done, param):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (hidden_in, hidden_out, state, action, last_action, reward, next_state, done, param)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def push_batch(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done, param):
        np.stack(last_action,axis=1)
        state, action, last_action, reward, next_state, done = \
            map(lambda x: np.stack(x, axis=1),[state, action, last_action, reward, next_state, done])
        assert all(state.shape[0] == x.shape[0] for x in [action, last_action, reward, next_state, done]), "Somethings wrong on dimension"
        num_batch = state.shape[0]
        hidden_in = hidden_in.view(num_batch,1,1,-1)
        hidden_out = hidden_out.view(num_batch,1,1,-1)
        self.buffer.extend([None]*int(min(num_batch,self.capacity-len(self.buffer))))
        for hin, hout, s,a,la,r,ns,d,p in zip(hidden_in, hidden_out, state, action, last_action, reward, next_state, done,param):
            self.buffer[self.position] = (hin,hout,s,a,la,r,ns,d,p)
            self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ho_lst, d_lst, p_lst=[],[],[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            h_in, h_out, state, action, last_action, reward, next_state, done, param = sample
            s_lst.append(state) 
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, hidden_size)
            ho_lst.append(h_out)
            p_lst.append(param)
        hidden_in = torch.cat(hi_lst, dim=-2).detach() # cat along the batch dim
        hidden_out = torch.cat(ho_lst, dim=-2).detach()

        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst = map(lambda x: np.stack(x,axis=0), [s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst])

        return hidden_in, hidden_out, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)


class HindsightReplayBufferLSTM(ReplayBufferFastAdaptLSTM):
    """ 
    Replay buffer for agent with LSTM network additionally storing previous action, 
    initial input hidden state and output hidden state of LSTM.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for LSTM initialization.

    """
    def __init__(self, capacity, gamma, epsilon_pos, epsilon_ang, history_length=50, mode='end',env='takeoff-aviary-v0'):
        super().__init__(capacity)
        self.gamma = gamma
        self.history_length = history_length
        self.epsilon_pos = epsilon_pos
        self.epsilon_ang = epsilon_ang
        self.mode = mode
        self.env = env
        if not self.mode in ['end','episode','future']:
            assert "mode should be choosen among [end, episode, future]"
        if not self.env in ['takeoff-aviary-v0', 'Pendulum-v0']:
            assert "env should be choosen among ['takeoff-aviary-v0', 'Pendulum-v0']"
    
    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done, param, goal):
        gs = [goal[None,:]]
        if np.random.random()<0.8:
            if self.mode=='end':
                gs.append(next_state[-1:,:])
            elif self.mode=='episode':
                idxs = list(np.random.randint(0,state.shape[0], size=4))
                gs.extend([next_state[idx:idx+1,:] for idx in idxs])

        for i,goal in enumerate(gs):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            if self.env == 'takeoff-aviary-v0':
                if i!=0:
                    ####### Distribution mean -> 0 #################
                    state[:,:3] = state[:,:3] - goal[:,:3]
                    next_state[:,:3] = next_state[:,:3] - goal[:,:3]
                    goal[:,:3] = 0
                    ################################################
                pos_achieve = np.linalg.norm(next_state[:,:3]-goal[:,:3],axis=-1)<self.epsilon_pos
                ang_achieve = rot_matrix_similarity(next_state[:,3:12],goal[:,3:12])<self.epsilon_ang
                reward = (1-self.gamma)*np.where(np.logical_and(pos_achieve, ang_achieve) ,0.0, -1.0)+self.gamma*reward
                done = np.where(np.logical_and(pos_achieve, ang_achieve) , 1.0, 0.0)
                done[:-1] = 0.0
            elif self.env == 'Pendulum-v0':
                pos_achieve = np.linalg.norm(next_state[:,:2]-goal[:,:2],axis=-1)<self.epsilon_pos
                reward = (1-self.gamma)*np.where(np.logical_and(pos_achieve, ang_achieve) ,0.0, -1.0)+self.gamma*reward
                done = np.where(pos_achieve , 1.0, 0.0)
            # ang_achieve = np.linalg.norm(next_state[:,15:18]-goal[:,15:18],axis=-1)<self.epsilon_ang
            self.buffer[self.position] = (hidden_in, hidden_out, state, action, last_action, reward, next_state, done, param, goal)
            self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def push_batch(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done, param, goal):
        B,L = state[0].shape[0], len(state)
        state, action, last_action, reward, next_state, done = \
            map(lambda x: np.stack(x, axis=1),[state, action, last_action, reward, next_state, done])
        assert all(state.shape[0] == x.shape[0] for x in [action, last_action, reward, next_state, done]), "Somethings wrong on dimension"
        hidden_in = (hidden_in[0].view(B,1,1,-1), hidden_in[1].view(B,1,1,-1))
        hidden_out = (hidden_out[0].view(B,1,1,-1), hidden_out[1].view(B,1,1,-1))

        for hin_h,hin_c, hout_h,hout_c, s,a,la,r,ns,d,p,g in zip(*hidden_in, *hidden_out, state, action, last_action, reward, next_state, done,param, goal):
            hin, hout = (hin_h,hin_c), (hout_h,hout_c)
            for i in range(0,L+1-self.history_length,self.history_length):
                self.push(hin,hout,*map(lambda x:x[i:i+self.history_length],[s,a,la,r,ns,d]),p,g)

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ci_lst, ho_lst, co_lst, d_lst, p_lst, g_lst=[],[],[],[],[],[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            (h_in, c_in), (h_out, c_out), state, action, last_action, reward, next_state, done, param, goal = sample
            # t0 = np.random.randint(0,state.shape[0]-self.sample_length)
            s_lst.append(state) 
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, hidden_size)
            ci_lst.append(c_in)
            ho_lst.append(h_out)
            co_lst.append(c_out)
            p_lst.append(param)
            g_lst.append(goal)
        hi_lst = torch.cat(hi_lst, dim=-2).detach() # cat along the batch dim
        ho_lst = torch.cat(ho_lst, dim=-2).detach()
        ci_lst = torch.cat(ci_lst, dim=-2).detach()
        co_lst = torch.cat(co_lst, dim=-2).detach()

        hidden_in = (hi_lst, ci_lst)
        hidden_out = (ho_lst, co_lst)
        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, g_lst, p_lst = map(lambda x: np.stack(x,axis=0), [s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, g_lst, p_lst])

        return hidden_in, hidden_out, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst, g_lst

    # def sample(self, batch_size):
    #     s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ci_lst, ho_lst, co_lst, d_lst, p_lst, g_lst=[],[],[],[],[],[],[],[],[],[],[],[]
    #     batch = random.sample(self.buffer, batch_size)
    #     for sample in batch:
    #         (h_in, c_in), (h_out, c_out), state, action, last_action, reward, next_state, done, param, goal = sample
    #         t = np.random.randint(0,state.shape[0]-self.sample_length)
    #         t0 = 0
    #         # t0 = np.random.randint(0,state.shape[0]-self.sample_length)
    #         s_lst.extend([state[t0:t0+self.sample_length], state[t:t+self.sample_length]]) 
    #         a_lst.extend([action[t0:t0+self.sample_length],action[t:t+self.sample_length]])
    #         la_lst.extend([last_action[t0:t0+self.sample_length],last_action[t:t+self.sample_length]])
    #         r = np.zeros_like(reward[t:t+self.sample_length])
    #         r[-1] = 1
    #         r_lst.extend([reward[t0:t0+self.sample_length]+r,reward[t:t+self.sample_length]+r])
    #         ns_lst.extend([next_state[t0:t0+self.sample_length],next_state[t:t+self.sample_length]])
    #         d_lst.extend([done[t0:t0+self.sample_length],done[t:t+self.sample_length]])
    #         hi_lst.extend([h_in,h_in])  # h_in: (1, batch_size=1, hidden_size)
    #         ci_lst.extend([c_in,c_in])
    #         ho_lst.extend([h_out, h_out])
    #         co_lst.extend([c_out,c_out])
    #         p_lst.extend([param,param])
    #         g_lst.extend([next_state[t0+self.sample_length-1:t0+self.sample_length],next_state[t+self.sample_length-1:t+self.sample_length]])
    #     hi_lst = torch.cat(hi_lst, dim=-2).detach() # cat along the batch dim
    #     ho_lst = torch.cat(ho_lst, dim=-2).detach()
    #     ci_lst = torch.cat(ci_lst, dim=-2).detach()
    #     co_lst = torch.cat(co_lst, dim=-2).detach()

    #     hidden_in = (hi_lst, ci_lst)
    #     hidden_out = (ho_lst, co_lst)
    #     s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, g_lst, p_lst = map(lambda x: np.stack(x,axis=0), [s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, g_lst, p_lst])

    #     return hidden_in, hidden_out, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst, g_lst

    def sample_original(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ci_lst, ho_lst, co_lst, d_lst, p_lst, g_lst=[],[],[],[],[],[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, 2*batch_size)
        for sample in batch:
            (h_in, c_in), (h_out, c_out), state, action, last_action, reward, next_state, done, param, goal = sample
            s_lst.append(state) 
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, hidden_size)
            ci_lst.append(c_in)
            ho_lst.append(h_out)
            co_lst.append(c_out)
            p_lst.append(param)
            g_lst.append(np.tile(goal, (state.shape[0],1)))
        hi_lst = torch.cat(hi_lst, dim=-2).detach() # cat along the batch dim
        ho_lst = torch.cat(ho_lst, dim=-2).detach()
        ci_lst = torch.cat(ci_lst, dim=-2).detach()
        co_lst = torch.cat(co_lst, dim=-2).detach()

        hidden_in = (hi_lst, ci_lst)
        hidden_out = (ho_lst, co_lst)
        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, g_lst, p_lst = map(lambda x: np.stack(x,axis=0), [s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, g_lst, p_lst])

        return hidden_in, hidden_out, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst, g_lst

class HindsightReplayBufferGRU(ReplayBufferFastAdaptGRU):
    """ 
    Replay buffer for agent with LSTM network additionally storing previous action, 
    initial input hidden state and output hidden state of LSTM.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for LSTM initialization.

    """
    def __init__(self, capacity, gamma, epsilon_pos, epsilon_ang, history_length=50, mode='end', env='takeoff-aviary-v0'):
        super().__init__(capacity)
        self.gamma = gamma
        self.history_length = history_length
        self.epsilon_pos = epsilon_pos
        self.epsilon_ang = epsilon_ang
        self.mode = mode
        self.env = env
        if not self.mode in ['end','episode','future']:
            assert "mode should be choosen among [end, episode, future]"
        if not self.env in ['takeoff-aviary-v0', 'Pendulum-v0']:
            assert "env should be choosen among ['takeoff-aviary-v0', 'Pendulum-v0']"

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done, param, goal):
        gs = [goal[None,:]]
        if np.random.random()<0.8:
            if self.mode=='end':
                gs.append(next_state[-1:,:])
            elif self.mode=='episode':
                idxs = list(np.random.randint(0,state.shape[0], size=4))
                gs.extend([next_state[idx:idx+1,:] for idx in idxs])

        for i,goal in enumerate(gs):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            if self.env == 'takeoff-aviary-v0':
                if i!=0:
                    ####### Distribution mean -> 0 #################
                    state[:,:3] = state[:,:3] - goal[:,:3]
                    next_state[:,:3] = next_state[:,:3] - goal[:,:3]
                    goal[:,:3] = 0
                    ################################################
                pos_achieve = np.linalg.norm(next_state[:,:3]-goal[:,:3],axis=-1)<self.epsilon_pos
                ang_achieve = rot_matrix_similarity(next_state[:,3:12],goal[:,3:12])<self.epsilon_ang
                reward = (1-self.gamma)*np.where(np.logical_and(pos_achieve, ang_achieve) ,0.0, -1.0)+self.gamma*reward
                done = np.where(np.logical_and(pos_achieve, ang_achieve) , 1.0, 0.0)
                done[:-1] = 0.0
            elif self.env == 'Pendulum-v0':
                pos_achieve = np.linalg.norm(next_state[:,:2]-goal[:,:2],axis=-1)<self.epsilon_pos
                reward = (1-self.gamma)*np.where(np.logical_and(pos_achieve, ang_achieve) ,0.0, -1.0)+self.gamma*reward
                done = np.where(pos_achieve , 1.0, 0.0)
            # ang_achieve = np.linalg.norm(next_state[:,15:18]-goal[:,15:18],axis=-1)<self.epsilon_ang
            
            
            self.buffer[self.position] = (hidden_in, hidden_out, state, action, last_action, reward, next_state, done, param, goal)
            self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def push_batch(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done, param, goal):
        B,L = state[0].shape[0], len(state)
        state, action, last_action, reward, next_state, done = \
            map(lambda x: np.stack(x, axis=1),[state, action, last_action, reward, next_state, done])
        assert all(state.shape[0] == x.shape[0] for x in [action, last_action, reward, next_state, done]), "Somethings wrong on dimension"
        hidden_in = hidden_in.view(B,1,1,-1)
        hidden_out = hidden_out.view(B,1,1,-1)

        for hin,hout,s,a,la,r,ns,d,p,g in zip(hidden_in, hidden_out, state, action, last_action, reward, next_state, done,param, goal):
            for i in range(0,L+1-self.history_length,self.history_length):
                self.push(hin,hout,*map(lambda x:x[i:i+self.history_length],[s,a,la,r,ns,d]),p,g)

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ho_lst, d_lst, p_lst, g_lst=[],[],[],[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            h_in, h_out, state, action, last_action, reward, next_state, done, param, goal = sample
            s_lst.append(state) 
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, hidden_size)
            ho_lst.append(h_out)
            p_lst.append(param)
            g_lst.append(goal)
        hidden_in = torch.cat(hi_lst, dim=-2).detach() # cat along the batch dim
        hidden_out = torch.cat(ho_lst, dim=-2).detach()

        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, g_lst, p_lst = map(lambda x: np.stack(x,axis=0), [s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, g_lst, p_lst])

        return hidden_in, hidden_out, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst, g_lst

    # def sample(self, batch_size):
    #     s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ho_lst, d_lst, p_lst, g_lst=[],[],[],[],[],[],[],[],[],[]
    #     batch = random.sample(self.buffer, batch_size)
    #     for sample in batch:
    #         h_in, h_out, state, action, last_action, reward, next_state, done, param, _ = sample
    #         t = np.random.randint(0,state.shape[0]-self.sample_length)
    #         t0 = 0
    #         # t0 = np.random.randint(0,state.shape[0]-self.sample_length)
    #         s_lst.extend([state[t0:t0+self.sample_length], state[t:t+self.sample_length]]) 
    #         a_lst.extend([action[t0:t0+self.sample_length],action[t:t+self.sample_length]])
    #         la_lst.extend([last_action[t0:t0+self.sample_length],last_action[t:t+self.sample_length]])
    #         r = -np.ones_like(reward[t:t+self.sample_length])
    #         r[-1] = 0
    #         r_lst.extend([self.gamma*reward[t0:t0+self.sample_length]+(1-self.gamma)*r,
    #                     self.gamma*reward[t:t+self.sample_length]+(1-self.gamma)*r])
    #         ns_lst.extend([next_state[t0:t0+self.sample_length],next_state[t:t+self.sample_length]])
    #         d_lst.extend([done[t0:t0+self.sample_length],done[t:t+self.sample_length]])
    #         hi_lst.extend([h_in,h_in])  # h_in: (1, batch_size=1, hidden_size)
    #         ho_lst.extend([h_out, h_out])
    #         p_lst.extend([param,param])
    #         g_lst.extend([next_state[t0+self.sample_length-1:t0+self.sample_length],next_state[t+self.sample_length-1:t+self.sample_length]])
    #     hidden_in = torch.cat(hi_lst, dim=-2).detach() # cat along the batch dim
    #     hidden_out = torch.cat(ho_lst, dim=-2).detach()

    #     s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, g_lst, p_lst = map(lambda x: np.stack(x,axis=0), [s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, g_lst, p_lst])

    #     return hidden_in, hidden_out, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst, g_lst

    def sample_original(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ho_lst, d_lst, p_lst, g_lst=[],[],[],[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, 2*batch_size)
        for sample in batch:
            h_in, h_out, state, action, last_action, reward, next_state, done, param, goal = sample
            s_lst.append(state) 
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, hidden_size)
            ho_lst.append(h_out)
            p_lst.append(param)
            g_lst.append(np.tile(goal, (state.shape[0],1)))
        hidden_in = torch.cat(hi_lst, dim=-2).detach() # cat along the batch dim
        hidden_out = torch.cat(ho_lst, dim=-2).detach()

        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, g_lst, p_lst = map(lambda x: np.stack(x,axis=0), [s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, g_lst, p_lst])

        return hidden_in, hidden_out, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst, g_lst

class HindsightReplayBuffer:
    """ 
    Replay buffer for agent with LSTM network additionally storing previous action, 
    initial input hidden state and output hidden state of LSTM.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for LSTM initialization.

    """
    def __init__(self, capacity, sample_length):
        self.capacity = capacity
        self.sample_length = sample_length
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, param):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, param)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def push_batch(self, state, action, reward, next_state, done, param):
        state, action, reward, next_state, done = \
            map(lambda x: np.stack(x, axis=1),[state, action, reward, next_state, done])
        assert all(state.shape[0] == x.shape[0] for x in [action, reward, next_state, done]), "Somethings wrong on dimension"
        num_batch = state.shape[0]
        self.buffer.extend([None]*int(min(num_batch,self.capacity-len(self.buffer))))
        for s,a,la,r,ns,d,p in zip(state, action, reward, next_state, done,param):
            self.buffer[self.position] = (s,a,la,r,ns,d,p)
            self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, r_lst, ns_lst, d_lst, g_lst, p_lst=[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            state, action, reward, next_state, done, param = sample
            t = np.random.randint(0,state.shape[0]-self.sample_length)
            s_lst.append(state[t:t+self.sample_length]) 
            a_lst.append(action[t:t+self.sample_length])
            new_reward = -np.ones_like(reward[t:t+self.sample_length])
            new_reward[-1] = 0
            r_lst.append(new_reward)
            ns_lst.append(next_state[t:t+self.sample_length])
            d_lst.append(done[t:t+self.sample_length])
            p_lst.append(param)
            goal = np.zeros_like(state[t:t+self.sample_length]) + state[t+self.sample_length-1:t+self.sample_length]
            g_lst.append(goal)

        s_lst, a_lst, r_lst, ns_lst, d_lst, g_lst, p_lst = map(lambda x: np.concatenate(x,axis=0), [s_lst, a_lst, r_lst, ns_lst, d_lst, p_lst])

        return s_lst, a_lst, r_lst, ns_lst, d_lst, g_lst, p_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)
