from email import policy
from turtle import write_docstringdict
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F

from td3.td3 import TD3_Trainer, TD3LSTM_Trainer, TD3FastAdaptLSTM_Trainer
from td3.common.buffers import ReplayBuffer, ReplayBufferLSTM2, ReplayBufferPER, ReplayBufferFastAdaptLSTM

from utils import collect_trajectories, random_sample
from envs.customEnv import domainRandeEnv

import argparse
import gym

import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime



def train(args):

    algorithm_name = 'TD3'
    env_name = args.env
    tag = 'single'
    if args.randomize:
        tag = "randomize"
    elif args.multitask:
        tag = "multitask"

    islstm = "FastAdaptLSTM"
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
        print("[Tensorboard log]:", tbpath)
        writer = SummaryWriter(tbpath)
    if args.randomize or args.multitask:
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
    else:
        dyn_range = {
            # cartpole
            'masscart': 1, # 1/n ~ n
            'masspole': 1, # 1/n ~ n
            'length': 1, # 1/n ~ n
            'force_mag': 1, # 1/n ~ n

            # pendulum
            'max_torque': 1, # 1/n ~ n
            'm': 1, # 1/n ~ n
            'l': 1, # 1/n ~ n

            # drones
            'mass_range': 0, # (1-n) ~ (1+n)
            'cm_range': 0, # (1-n) ~ (1+n)
            'kf_range': 0, # (1-n) ~ (1+n)
            'km_range': 0, # (1-n) ~ (1+n)
            'battery_range': 0 # (1-n) ~ (1)
        }
    
    device=torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    # hyper-parameters for RL training

    if 'Pendulum' in env_name:
        max_episodes  = 10000
        hidden_dim = 64
        max_steps = 150
        batch_size  = 8
        param_num = 3
    elif 'aviary' in env_name:
        max_episodes  = 20000
        hidden_dim = 128
        max_steps = 400
        batch_size  = 2
        param_num = 12
    else:
        raise NotImplementedError

    nenvs = 25
    best_score = -np.inf
    frame_idx   = 0
    replay_buffer_size = int(1e6/max_steps)
    explore_steps = int(1e5/max_steps)  # for random action sampling in the beginning of training
    update_itr = 1
    policy_target_update_interval = 2 # delayed update for the policy network and target networks

    DETERMINISTIC=True  # DDPG: deterministic policy gradient
    explore_noise_scale = 1.0
    eval_noise_scale = 0.5

    # Define environment
    envs = domainRandeEnv(env_name=env_name, tag=tag, n=nenvs, randomize=args.randomize, seed=100000, dyn_range=dyn_range)
    action_space = envs.action_space
    state_space = envs.observation_space

    replay_buffer = ReplayBufferFastAdaptLSTM(replay_buffer_size)
    td3_trainer = TD3FastAdaptLSTM_Trainer(replay_buffer,
                state_space, 
                action_space, 
                hidden_dim=hidden_dim,
                param_num=param_num,
                out_actf=F.sigmoid if 'aviary' in env_name else None,
                device=device, 
                policy_target_update_interval=policy_target_update_interval)

    # keep track of progress
    mean_rewards = []
    scores_window = deque(maxlen=100)
    loss_storage = {"policy_loss":[],
                    "q_loss_1":[],
                    "q_loss_2":[],
                    "param_loss":[]}

    for i_episode in range(1,max_episodes+1):
        state, params = envs.reset()
        last_action = np.stack([envs.action_space.sample() for _ in range(nenvs)],axis=0).squeeze()
        episode_state = []
        episode_action = []
        episode_last_action = []
        episode_reward = []
        episode_next_state = []
        episode_done = []
        hidden_out = (torch.zeros([1, nenvs, hidden_dim], dtype=torch.float).to(device), \
                    torch.zeros([1, nenvs, hidden_dim], dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)


        policy_loss = []
        q_loss_1,q_loss_2 = [],[]
        param_loss = []
        for step in range(max_steps):
            hidden_in = hidden_out
            action, hidden_out = \
                td3_trainer.policy_net.get_action(state, 
                                                last_action, 
                                                hidden_in, 
                                                deterministic=DETERMINISTIC, 
                                                explore_noise_scale=explore_noise_scale)
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
                    loss_dict = td3_trainer.update(batch_size, deterministic=DETERMINISTIC, eval_noise_scale=eval_noise_scale)
                    # policy_loss, q_loss_1, q_loss_2
                    policy_loss.append(loss_dict['policy_loss'])
                    q_loss_1.append(loss_dict['q_loss_1'])
                    q_loss_2.append(loss_dict['q_loss_2'])
                    param_loss.append(loss_dict['param_loss'])
        
        replay_buffer.push_batch(ini_hidden_in, 
                            ini_hidden_out, 
                            episode_state, 
                            episode_action, 
                            episode_last_action,
                            episode_reward, 
                            episode_next_state, 
                            episode_done,
                            params)
        loss_storage['policy_loss'].append(np.mean(policy_loss))
        loss_storage['q_loss_1'].append(np.mean(q_loss_1))
        loss_storage['q_loss_2'].append(np.mean(q_loss_2))
        loss_storage['param_loss'].append(np.mean(param_loss))

        rewards = np.sum(episode_reward)/nenvs
        mean_rewards.append(rewards)
        scores_window.append(rewards)

        writer_iterval = 10
        if writer is not None and i_episode%writer_iterval == 0:
            writer.add_scalar('loss/loss_p', np.mean(loss_storage['policy_loss'][-writer_iterval:]), i_episode)
            writer.add_scalar('loss/loss_q_1', np.mean(loss_storage['q_loss_1'][-writer_iterval:]), i_episode)
            writer.add_scalar('loss/loss_q_2', np.mean(loss_storage['q_loss_2'][-writer_iterval:]), i_episode)
            writer.add_scalar('loss/loss_param', np.mean(loss_storage['param_loss'][-writer_iterval:]), i_episode)
            writer.add_scalar('rewards', scores_window[-1], i_episode)
        
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            td3_trainer.save_model(os.path.join(savepath,"iter%05d"%i_episode))

        if np.mean(scores_window)>=best_score: 
            td3_trainer.save_model(os.path.join(savepath,"best"))   
            best_score = np.mean(scores_window)

        
    td3_trainer.save_model(os.path.join(savepath,"final"))

    return mean_rewards, loss_storage

def train_stablebaseline():
    pass

if __name__=='__main__':
    # train(1000, 'CartPole-v1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True, choices=['Pendulum-v0','HalfCheetah-v2','takeoff-aviary-v0'])
    parser.add_argument('--multitask',action='store_true', help="Multitask")
    parser.add_argument('--randomize',action='store_true', help="Domain randomize")
    parser.add_argument('--tb_log', action='store_true', help="Tensorboard logging")
    parser.add_argument('--gpu', default='0', type=int, help="gpu number")
    args = parser.parse_args()
    if args.multitask and args.randomize:
        raise "Only give an option between multitask and randomize"
    train(args)