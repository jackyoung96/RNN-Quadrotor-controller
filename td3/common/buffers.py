import math
import random
import numpy as np
from pkg_resources import Environment
import torch

from .utils import rot_matrix_similarity, rot_matrix_z_similarity


class ReplayBuffer:
    def __init__(self, capacity, **kwargs):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, last_action, reward, next_state, done, param):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, last_action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def push_batch(self, state, action, last_action, reward, next_state, done, param):
        np.stack(last_action,axis=1)
        state, action, last_action, reward, next_state, done = \
            map(lambda x: np.concatenate(x, axis=0),[state, action, last_action, reward, next_state, done])
        assert all(state.shape[0] == x.shape[0] for x in [action, last_action, reward, next_state, done]), "Somethings wrong on dimension"
        for s,a,la,r,ns,d in zip(state, action, last_action, reward, next_state, done):
            self.push(s,a,la,r,ns,d,None)

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
    def __init__(self, capacity, **kwargs):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, last_action, reward, next_state, done, param):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, last_action, reward, next_state, done, param)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def push_batch(self, state, action, last_action, reward, next_state, done, param):
        np.stack(last_action,axis=1)
        state, action, last_action, reward, next_state, done = \
            map(lambda x: np.stack(x, axis=1),[state, action, last_action, reward, next_state, done])
        assert all(state.shape[0] == x.shape[0] for x in [action, last_action, reward, next_state, done]), "Somethings wrong on dimension"
        for s,a,la,r,ns,d,p in zip(state, action, last_action, reward, next_state, done,param):
            self.push(s,a,la,r,ns,d,p)

    def sample(self, batch_size):
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


class HindsightReplayBufferRNN(ReplayBufferRNN):
    """ 
    Replay buffer for agent with LSTM network additionally storing previous action, 
    initial input hidden state and output hidden state of LSTM.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for LSTM initialization.

    """
    def __init__(self, capacity, env_name='takeoff-aviary-v0', **kwargs):
        super().__init__(capacity, **kwargs)
        self.positive_rew = kwargs.get("positive_rew", False)
        self.angvel_goal = kwargs.get("angvel_goal", False)
        self.gamma = kwargs.get("her_gamma", 0.0)
        self.her = True if self.gamma != 1.0 else False
        self.history_length = kwargs.get("her_length", 100)
        self.epsilon_pos = kwargs.get("epsilon_pos", 0.1/6)
        self.epsilon_ang = kwargs.get("epsilon_ang", np.deg2rad(10))
        self.single_pos = kwargs.get("single_pos", False)
        self.env_name = env_name
        if not self.env_name in ['takeoff-aviary-v0', 'Pendulum-v0']:
            assert "env should be choosen among ['takeoff-aviary-v0', 'Pendulum-v0']"
    
    def push(self, state, action, last_action, reward, next_state, done, param, goal):
        gs = [goal[None,:]]
        if np.random.random()<0.8:
            goal = next_state[-1:,:].copy() # Achieved goal
            if self.single_pos:
                goal[:,:3] = 0 # Single goal
            gs.append(goal)

        for i,goal in enumerate(gs):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            if self.env_name == 'takeoff-aviary-v0':
                if i!=0 and self.single_pos:
                    ####### Relative observation ###################
                    state_m = state[:,3:12].reshape((-1,3,3))
                    next_state_m = next_state[:,3:12].reshape((-1,3,3))
                    state_w = np.matmul(state_m, state[:,:3].reshape((-1,3,1)))
                    next_state_w = np.matmul(next_state_m, next_state[:,:3].reshape((-1,3,1)))

                    state_w = state_w - next_state_w[-1:].copy()
                    next_state_w = next_state_w - next_state_w[-1:].copy()

                    state[:,:3] = np.matmul(np.swapaxes(state_m,1,2), state_w[:,:3]).reshape((-1,3))
                    next_state[:,:3] = np.matmul(np.swapaxes(next_state_m,1,2), next_state_w[:,:3]).reshape((-1,3))
                    ################################################
                pos_achieve = np.linalg.norm(next_state[:,:3]-goal[:,:3],axis=-1)<self.epsilon_pos
                ang_value = rot_matrix_similarity(next_state[:,3:12],goal[:,3:12]) # 1: 0deg, 0: >90 deg, from vertical z-axis
                ang_achieve = ang_value < self.epsilon_ang
                angvel_achieve = 2*180*np.linalg.norm(next_state[:,15:18]-goal[:,15:18],axis=-1)< 18*self.epsilon_ang # 180 deg/s 

                reward = np.where(np.logical_and(pos_achieve, ang_achieve, angvel_achieve), 0.0, -1.0)
                
                # done = np.where(np.logical_and(pos_achieve, ang_achieve, angvel_achieve) , 1.0, 0.0)

            elif self.env_name == 'Pendulum-v0':
                theta = np.arctan2(next_state[:,1:2],next_state[:,0:1])
                goal_theta = np.arctan2(goal[:,1:2],goal[:,0:1])
                ang_achieve = (theta-goal_theta)%(2*np.pi)<self.epsilon_ang
                reward = (1-self.gamma)*np.where(ang_achieve ,0.0, -1.0)+self.gamma*reward
                # done = np.where(pos_achieve , 1.0, 0.0)
            # ang_achieve = np.linalg.norm(next_state[:,15:18]-goal[:,15:18],axis=-1)<self.epsilon_ang
            self.buffer[self.position] = (state, action, last_action, reward, next_state, done, param, goal)
            self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def push_batch(self, state, action, last_action, reward, next_state, done, param, goal):
        B,L = state[0].shape[0], len(state)
        state, action, last_action, reward, next_state, done = \
            map(lambda x: np.stack(x, axis=1),[state, action, last_action, reward, next_state, done])
        assert all(state.shape[0] == x.shape[0] for x in [action, last_action, reward, next_state, done]), "Somethings wrong on dimension"
        for s,a,la,r,ns,d,p,g in zip(state, action, last_action, reward, next_state, done,param, goal):
            for i in range(0,L+1-self.history_length,self.history_length):
                self.push(*map(lambda x:x[i:i+self.history_length],[s,a,la,r,ns,d]),p,g)

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst, g_lst=[],[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        
        for sample in batch:
            state, action, last_action, reward, next_state, done, param, goal = sample
            # t0 = np.random.randint(0,state.shape[0]-self.sample_length)
            s_lst.append(state) 
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            p_lst.append(param)
            g_lst.append(goal)
        
        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst, g_lst = map(lambda x: np.stack(x,axis=0), [s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst, g_lst])

        return s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, p_lst, g_lst

class SingleHindsightReplayBufferRNN(ReplayBufferRNN):
    """ 
    Replay buffer for agent with LSTM network additionally storing previous action, 
    initial input hidden state and output hidden state of LSTM.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for LSTM initialization.

    """
    def __init__(self, capacity, env_name='takeoff-aviary-v0', **kwargs):
        super().__init__(capacity, **kwargs)
        self.positive_rew = kwargs.get("positive_rew", False)
        self.angvel_goal = kwargs.get("angvel_goal", False)
        self.gamma = kwargs.get("her_gamma", 0.0)
        self.reward_scale = kwargs.get("reward_scale", 1.0)
        self.maintain_length = kwargs.get("maintain_length", 1)
        self.her = True if self.gamma != 1.0 else False
        self.history_length = kwargs.get("her_length", 100)
        self.epsilon_pos = kwargs.get("epsilon_pos", 0.1/6)
        self.epsilon_ang = kwargs.get("epsilon_ang", np.deg2rad(10))
        self.env_name = env_name
        if not self.env_name in ['takeoff-aviary-v0', 'Pendulum-v0']:
            assert "env should be choosen among ['takeoff-aviary-v0', 'Pendulum-v0']"
    
    def push(self, state, action, last_action, reward, next_state, done, param):
        gs = [np.zeros(3).reshape((1,3,1))]
        if np.random.random()<0.8:
            # last position
            new_goal = np.matmul(next_state[-1,3:12].reshape((3,3)), next_state[-1,:3].reshape((3,1))).reshape((1,3,1))
            gs.append(new_goal)
            # best angle
            idx = np.argmax(next_state[:,11])
            new_goal = np.matmul(next_state[idx,3:12].reshape((3,3)), next_state[idx,:3].reshape((3,1))).reshape((1,3,1))
            gs.append(new_goal)
            # random position
            for _ in range(2):
                idx = np.random.randint(next_state.shape[0])
                new_goal = np.matmul(next_state[idx,3:12].reshape((3,3)), next_state[idx,:3].reshape((3,1))).reshape((1,3,1))
                gs.append(new_goal)

        ###### Relative position ################
        state_m = state[:,3:12].reshape((-1,3,3))
        next_state_m = next_state[:,3:12].reshape((-1,3,3))
        state_w = np.matmul(state_m, state[:,:3].reshape((-1,3,1)))
        next_state_w = np.matmul(next_state_m, next_state[:,:3].reshape((-1,3,1)))

        for i,goal in enumerate(gs):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            if self.env_name == 'takeoff-aviary-v0':
                state_new = state.copy()
                next_state_new = next_state.copy()
                if i!=0:
                    ####### Relative observation ###################
                    state_new = state_new[:,:,None]
                    next_state_new = next_state_new[:,:,None]
                    state_new[:,:3] = state_w - goal
                    next_state_new[:,:3] = next_state_w - goal

                    state_new[:,:3] = np.matmul(np.swapaxes(state_m,1,2), state_new[:,:3])
                    next_state_new[:,:3] = np.matmul(np.swapaxes(next_state_m,1,2), next_state_new[:,:3])
                    state_new = state_new[:,:,0]
                    next_state_new = next_state_new[:,:,0]
                    ################################################
                pos_achieve = np.linalg.norm(next_state_new[:,:3],axis=-1) < self.epsilon_pos
                ang_value = next_state_new[:,11] # 1: 0deg, -1: 180 deg, from vertical z-axis
                ang_achieve = np.arccos(ang_value) < self.epsilon_ang

                # Cumulative rewards
                cum_ang_value = np.cumsum(ang_value)
                for j in range(ang_value.shape[0]-1, 0, -1):
                    cum_ang_value[j] = max(1,self.maintain_length/j) * (cum_ang_value[j]-cum_ang_value[max(j-self.maintain_length,0)])
                    # cum_ang_value[j] = cum_ang_value[j]-cum_ang_value[max(j-self.maintain_length,0)]
                reward_new = np.where(pos_achieve, self.reward_scale*cum_ang_value -1, -1.0)
                
                done_new = np.where(pos_achieve , 1.0, 0.0)

            elif self.env_name == 'Pendulum-v0':
                theta = np.arctan2(next_state[:,1:2],next_state[:,0:1])
                goal_theta = np.arctan2(goal[:,1:2],goal[:,0:1])
                ang_achieve = (theta-goal_theta)%(2*np.pi)<self.epsilon_ang
                reward = (1-self.gamma)*np.where(ang_achieve ,0.0, -1.0)+self.gamma*reward
                # done = np.where(pos_achieve , 1.0, 0.0)
            # ang_achieve = np.linalg.norm(next_state[:,15:18]-goal[:,15:18],axis=-1)<self.epsilon_ang
            self.buffer[self.position] = (state_new, action, last_action, reward_new, next_state_new, done_new, param)
            self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def push_batch(self, state, action, last_action, reward, next_state, done, param):
        B,L = state[0].shape[0], len(state)
        state, action, last_action, reward, next_state, done = \
            map(lambda x: np.stack(x, axis=1),[state, action, last_action, reward, next_state, done])
        assert all(state.shape[0] == x.shape[0] for x in [action, last_action, reward, next_state, done]), "Somethings wrong on dimension"
        for s,a,la,r,ns,d,p in zip(state, action, last_action, reward, next_state, done,param):
            for i in range(0,L+1-self.history_length,self.history_length):
                self.push(*map(lambda x:x[i:i+self.history_length],[s,a,la,r,ns,d]),p)