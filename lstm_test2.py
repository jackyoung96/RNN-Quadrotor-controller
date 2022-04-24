from email import policy
from turtle import write_docstringdict
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F

from td3.td3 import TD3_Trainer, TD3LSTM_Trainer
from td3.common.buffers import ReplayBuffer, ReplayBufferLSTM2, ReplayBufferPER, ReplayBufferFastAdaptLSTM, ReplayBufferFastAdaptGRU

from utils import collect_trajectories, random_sample
from envs.customEnv import domainRandeEnv

import argparse
import gym

import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from evaluation import evaluation
import random
import pickle as pkl

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyvirtualdisplay import Display


class lstm_model(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(lstm_model,self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3+1,hidden_dim),
            torch.nn.ReLU())
        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.net2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 3),
            torch.nn.Tanh())
    def forward(self,x):
        x = self.net(x)
        x,_ = self.lstm(x)
        out = self.net2(x)
        return out

class rnn_model(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(rnn_model,self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3+1,hidden_dim),
            torch.nn.ReLU())
        self.rnn = torch.nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.net2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 3),
            torch.nn.Tanh())
    def forward(self,x):
        x = self.net(x)
        x,_ = self.rnn(x)
        out = self.net2(x)
        return out

class gru_model(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(gru_model,self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3+1,hidden_dim),
            torch.nn.ReLU())
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.net2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 3),
            torch.nn.Tanh())
    def forward(self,x):
        x = self.net(x)
        x,_ = self.gru(x)
        out = self.net2(x)
        return out

def train(args, hparam):
    print("hyperparam set:",hparam)
    algorithm_name = 'TD3'
    env_name = 'Pendulum-v0'
    tag = "randomize"

    islstm = "%stest"%args.rnn

    savepath = "save/%s/%s/%s/%s"%(algorithm_name,tag,islstm,env_name)

    if not os.path.isdir(savepath):
        os.makedirs(savepath) 
    
    writer = None
    if args.tb_log:
        if not os.path.isdir('tb_log'):
            os.mkdir('tb_log')
        dtime = datetime.now()
        tbpath = "%s/%s/%s"%(algorithm_name, tag, islstm)
        tbpath = os.path.join('tb_log', env_name, tbpath, dtime.strftime("%y%b%d%H%M%S"))
        writer = SummaryWriter(tbpath)
    dyn_range = {
        # cartpole
        'masscart': 3, # 1/n ~ n
        'masspole': 3, # 1/n ~ n
        'length': 3, # 1/n ~ n
        'force_mag': 3, # 1/n ~ n

        # pendulum
        'max_torque': 3, # 1/n ~ n
        'm': 3, # 1/n ~ n
        'l': 3, # 1/n ~ n

        # drones
        'mass_range': 0.3, # (1-n) ~ (1+n)
        'cm_range': 0.3, # (1-n) ~ (1+n)
        'kf_range': 0.1, # (1-n) ~ (1+n)
        'km_range': 0.1, # (1-n) ~ (1+n)
        'battery_range': 0.3 # (1-n) ~ (1)
    }
    
    device=torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    # hyper-parameters for RL training
    max_episodes  = 3000
    hidden_dim = 128
    max_steps = 100
    batch_size  = 64

    nenvs = 64
    best_score = -np.inf
    frame_idx   = 0
    replay_buffer_size = int(1e6/max_steps)
    explore_steps = int(1e4/max_steps)  # for random action sampling in the beginning of training
    update_itr = 1
    policy_target_update_interval = hparam['update_interval'] # delayed update for the policy network and target networks
    
    eval_freq = 50
    eval_itr = 100
    loss = None

    DETERMINISTIC=True  # DDPG: deterministic policy gradient
    explore_noise_scale = 0.5
    eval_noise_scale = 0.5

    # Define environment
    envs = domainRandeEnv(env_name=env_name, tag=tag, n=nenvs, randomize=True, seed=1000000, dyn_range=dyn_range)
    action_space = envs.action_space
    state_space = envs.observation_space
    if args.rnn=='lstm':
        replay_buffer = ReplayBufferFastAdaptLSTM(replay_buffer_size)
        model = lstm_model(hidden_dim).to(device)
        lstm_optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif args.rnn=='gru':
        replay_buffer = ReplayBufferFastAdaptGRU(replay_buffer_size)
        model = gru_model(hidden_dim).to(device)
        lstm_optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif args.rnn=='rnn':
        replay_buffer = ReplayBufferFastAdaptGRU(replay_buffer_size)
        model = rnn_model(hidden_dim).to(device)
        lstm_optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    

    for i_episode in range(1,max_episodes+1):
        # print(i_episode)
        state, param = envs.reset()
        last_action = np.stack([envs.action_space.sample() for _ in range(nenvs)],axis=0).squeeze()
        if len(last_action.shape)==0:
            last_action = np.array([last_action])
        last_action = np.zeros_like(last_action)
        episode_state = []
        episode_action = []
        episode_last_action = []
        episode_reward = []
        episode_next_state = []
        episode_done = []
        if args.rnn=='lstm':
            hidden_out = (torch.zeros([1, nenvs, hidden_dim], dtype=torch.float).to(device), \
                        torch.zeros([1, nenvs, hidden_dim], dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
        else:
            hidden_out = torch.zeros([1, nenvs, hidden_dim], dtype=torch.float).to(device)
        hidden_in = hidden_out

        for step in range(max_steps):
            hidden_in = hidden_out
            action = np.stack([envs.action_space.sample() for _ in range(nenvs)],axis=0).squeeze()
            if len(action.shape)==0:
                action = np.array([action])
            next_state, reward, done, _ = envs.step(action) 
            if step == 0:
                ini_hidden_in = hidden_in
                ini_hidden_out = hidden_out
            
            episode_state.append(state)
            episode_action.append(action)
            episode_last_action.append(last_action)
            episode_reward.append(reward)
            episode_next_state.append(next_state)
            episode_done.append(done)


            state = next_state
            last_action = action
            frame_idx += 1
            
            if len(replay_buffer) > explore_steps:
                for i in range(update_itr):
                    _hidden_in, _hidden_out, _state, _action, _last_action, _reward, _next_state, _done, _param = replay_buffer.sample(batch_size)

                    _state      = torch.FloatTensor(_state).to(device)
                    _next_state = torch.FloatTensor(_next_state).to(device)
                    _action     = torch.FloatTensor(_action).to(device)
                    _last_action     = torch.FloatTensor(_last_action).to(device)
                    _param = torch.FloatTensor(_param).to(device).view(batch_size,1,3)
                    _param = _param.expand(batch_size, max_steps, 3)

                    if len(_state.shape)==len(_last_action.shape)+1:
                        _last_action = _last_action.unsqueeze(-1)
                    sa = torch.cat([_state,_last_action], dim=-1)
                    param_predict = model(sa)
                    loss = torch.nn.MSELoss()(param_predict, _param)

                    lstm_optim.zero_grad()
                    loss.backward()
                    lstm_optim.step()
        
        replay_buffer.push_batch(ini_hidden_in, 
                            ini_hidden_out, 
                            episode_state, 
                            episode_action, 
                            episode_last_action,
                            episode_reward, 
                            episode_next_state, 
                            episode_done,
                            param)

        writer_iterval = 10
        if writer is not None and i_episode%writer_iterval == 0:
            if len(replay_buffer) > explore_steps:
                writer.add_scalar('loss', loss.item(), i_episode)

            for name, weight in model.named_parameters():
                writer.add_histogram(f'lstm_net/{name}', weight, i_episode)
                if weight.grad is not None:
                    writer.add_histogram(f'lstm_net/{name}.grad', weight.grad, i_episode)

        if i_episode % eval_freq ==1: # evaluation
            episode_state = []
            episode_action = []
            state, param = envs.reset()
            last_action = np.stack([envs.action_space.sample() for _ in range(nenvs)],axis=0).squeeze()
            if len(last_action.shape)==0:
                    last_action = np.array([last_action])
            for t in range(max_steps):
                
                episode_state.append(state)
                episode_action.append(last_action[:,None])
                action = np.stack([envs.action_space.sample() for _ in range(nenvs)],axis=0).squeeze()
                if len(action.shape)==0:
                    action = np.array([action])
                state, reward, done, _ = envs.step(action)
                last_action = action
                
            _state  = torch.FloatTensor(np.stack(episode_state,axis=1)).to(device)
            _action = torch.FloatTensor(np.stack(episode_action,axis=1)).to(device)
            sa = torch.cat([_state,_action], dim=-1)
            predict_param = model(sa).detach().cpu().numpy()
            label_param = np.expand_dims(param,axis=1)
            dparam = np.mean((predict_param-label_param)**2, axis=-1)

            with Display(visible=False, size=(100, 60)) as disp:
                print((i_episode//eval_freq), "trial")
                fig = plt.figure()
                ax = fig.add_subplot(111)
                for i in range(dparam.shape[0]):
                    ax.plot(np.arange(max_steps), dparam[i], label=np.array2string(param[i]))
                if not os.path.isdir("figs/%stest"%args.rnn):
                    os.makedirs("figs/%stest"%args.rnn)
                lgd = ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
                loss_title = 0 if loss is None else loss.item()
                plt.title("Loss: %.4f"%loss_title)
                plt.savefig("figs/%stest/%d.png"%(args.rnn, i_episode//eval_freq),bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.close()

    return mean_rewards, loss_storage

def train_stablebaseline():
    pass

hparam_set = {
    "q_lr": [3e-4, 1e-4, 3e-5],
    "policy_lr": [3e-4, 1e-4, 3e-5],
    "param_lr": [3e-4, 1e-4, 3e-5],
    "t_max": [10000, 30000, 100000],
    "hidden_dim": [64,128,256],
    "update_interval": [2,5,10]
}

if __name__=='__main__':
    # train(1000, 'CartPole-v1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--tb_log', action='store_true', help="Tensorboard logging")
    parser.add_argument('--gpu', default='0', type=int, help="gpu number")
    parser.add_argument('--hparam', action='store_true', help="find hparam set")
    parser.add_argument('--rnn', choices=['rnn', 'lstm', 'gru'], required=True)
    args = parser.parse_args()
    if args.hparam:
        hparam_test = {}
        for i_hparam in range(10):
            hparam = dict([(k,random.choice(v)) for k,v in hparam_set.items()])
            mean_rewards, loss_storage = train(args, hparam)
            hparam_test[i_hparam] = (hparam,mean_rewards)
        with open("hparam_test.pickle",'wb') as f:
            pkl.dump(hparam_test, f, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        hparam = dict([(k,v[0]) for k,v in hparam_set.items()])
        train(args, hparam)