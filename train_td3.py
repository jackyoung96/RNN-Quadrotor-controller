from email import policy
from turtle import write_docstringdict
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F

from td3.td3 import *
from td3.common.buffers import *
from utils import collect_trajectories, random_sample
from envs.customEnv import domainRandeEnv

import argparse
import gym
import psutil

import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from evaluation import evaluation, generate_result, drone_test
import random
import pickle as pkl
import pandas as pd
import itertools

from time import time

def train(args, hparam):
    
    print("hyperparam set:",hparam)
    algorithm_name = 'TD3'
    env_name = args.env

    dtime = datetime.now()

    tag = "randomize"

    rnn_tag = args.rnn if args.rnn != "None" else 'FF'

    savepath = "save/%s/%s/%s/%s/%s"%(algorithm_name,tag,rnn_tag,env_name, dtime.strftime("%y%b%d%H%M%S"))

    if not os.path.isdir(savepath):
        os.makedirs(savepath) 
    
    writer = None
    if args.tb_log:
        if not os.path.isdir('tb_log'):
            os.mkdir('tb_log')
        tbpath = "%s/%s/%s"%(algorithm_name, tag, rnn_tag)
        tbpath = os.path.join('tb_log', env_name, tbpath, dtime.strftime("%y%b%d%H%M%S"))
        print("[Tensorboard log]:", tbpath)
        writer = SummaryWriter(tbpath)
    # if args.randomize or args.multitask:
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
        'mass_range': 0.0, # (1-n) ~ (1+n)
        'cm_range': 0.0, # (1-n) ~ (1+n)
        'kf_range': 0.0, # (1-n) ~ (1+n)
        'km_range': 0.0, # (1-n) ~ (1+n)
        'i_range': 0.0,
        'battery_range': 0.0 # (1-n) ~ (1)
    }
    # else:
    #     dyn_range = {
    #         # cartpole
    #         'masscart': 1, # 1/n ~ n
    #         'masspole': 1, # 1/n ~ n
    #         'length': 1, # 1/n ~ n
    #         'force_mag': 1, # 1/n ~ n

    #         # pendulum
    #         'max_torque': 1, # 1/n ~ n
    #         'm': 1, # 1/n ~ n
    #         'l': 1, # 1/n ~ n

    #         # drones
    #         'mass_range': 0.3, # (1-n) ~ (1+n)
    #         'cm_range': 0.3, # (1-n) ~ (1+n)
    #         'kf_range': 0.3, # (1-n) ~ (1+n)
    #         'km_range': 0.3, # (1-n) ~ (1+n)
    #         'battery_range': 0.3 # (1-n) ~ (1) 
    #     }
    
    device=torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    # hyper-parameters for RL training
    if 'Pendulum' in env_name:
        max_episodes  = 3000
        hidden_dim = hparam['hidden_dim']
        max_steps = 150
        batch_size  = 64 if args.rnn != "None" else 64 * max_steps
        param_num = 3
        nenvs = 16
        explore_noise_scale = 0.5
        eval_noise_scale = 0.5
        her_pre_steps = 500
        her_sample_length = 50
    elif 'aviary' in env_name:
        max_episodes  = 6000
        hidden_dim = 128
        max_steps = 300
        batch_size  = 64 if args.rnn != "None" else 8 * max_steps
        param_num = 14
        nenvs = 2
        explore_noise_scale = 0.25
        eval_noise_scale = 0.25
        her_pre_steps = 1e3
        her_sample_length = 100
    else:
        raise NotImplementedError

    best_score = -np.inf
    frame_idx   = 0
    replay_buffer_size = int(1e6/max_steps) if args.rnn != "None" else 1e6
    explore_steps = int(1e5/max_steps) if args.rnn != "None" else 1e5  # for random action sampling in the beginning of training
    update_itr = 1
    policy_target_update_interval = hparam['update_interval'] # delayed update for the policy network and target networks
    
    eval_freq = 100
    eval_itr = 50

    DETERMINISTIC=True  # DDPG: deterministic policy gradient
    

    # Define environment
    envs = domainRandeEnv(env_name=env_name, tag="%s%s%s"%(algorithm_name, tag, rnn_tag), n=nenvs, randomize=True, seed=1000000, dyn_range=dyn_range)
    action_space = envs.action_space
    state_space = envs.observation_space
    
    if args.rnn in ["RNN", "LSTM", "GRU"]:
        # RNN w/o param
        if args.rnn=='LSTM':
            replay_buffer = ReplayBufferLSTM(replay_buffer_size)
        else:
            replay_buffer = ReplayBufferGRU(replay_buffer_size)
        td3_trainer = TD3RNN_Trainer(replay_buffer,
                    state_space, 
                    action_space, 
                    rnn_type=args.rnn,
                    out_actf=F.tanh,
                    action_scale=1.0 if 'aviary' in env_name else 10.0,
                    device=device, 
                    policy_target_update_interval=policy_target_update_interval,
                    **hparam)
    elif args.rnn in ["RNN2", "LSTM2", "GRU2"]:
        # Param + FF
        if args.rnn=='LSTM2':
            replay_buffer = ReplayBufferFastAdaptLSTM(replay_buffer_size)
        else:
            replay_buffer = ReplayBufferFastAdaptGRU(replay_buffer_size)
        td3_trainer = TD3RNN_Trainer2(replay_buffer,
                    state_space, 
                    action_space, 
                    param_num=param_num,
                    rnn_type=args.rnn,
                    out_actf=F.tanh,
                    action_scale=1.0 if 'aviary' in env_name else 10.0,
                    device=device, 
                    policy_target_update_interval=policy_target_update_interval,
                    **hparam)
    elif args.rnn in ["RNN3", "LSTM3", "GRU3"]:
        # Param + RNN
        if args.rnn=='LSTM3':
            replay_buffer = ReplayBufferFastAdaptLSTM(replay_buffer_size)
        else:
            replay_buffer = ReplayBufferFastAdaptGRU(replay_buffer_size)
        td3_trainer = TD3RNN_Trainer3(replay_buffer,
                    state_space, 
                    action_space, 
                    param_num=param_num,
                    rnn_type=args.rnn,
                    out_actf=F.tanh,
                    action_scale=1.0 if 'aviary' in env_name else 10.0,
                    device=device, 
                    policy_target_update_interval=policy_target_update_interval,
                    **hparam)
    elif args.rnn in ["RNNHER", "LSTMHER", "GRUHER"]:
        # batch_size = batch_size*int(max_steps//her_sample_length / 2)
        if args.rnn=='LSTMHER':
            replay_buffer = HindsightReplayBufferLSTM(replay_buffer_size, 
                                epsilon_pos=np.sqrt(3*(0.1**2)),
                                epsilon_ang=np.deg2rad(10))
                                # sample_length=her_sample_length)
        else:
            replay_buffer = HindsightReplayBufferGRU(replay_buffer_size, 
                                epsilon_pos=np.sqrt(3*(0.1**2)),
                                epsilon_ang=np.deg2rad(10))
                                # sample_length=her_sample_length)
        # For aviary, it include the last action in the state
        # goal_dim = state_space.shape[0]-4 if 'aviary' in env_name else state_space.shape[0]
        goal_dim = 12 if 'aviary' in env_name else state_space.shape[0]
        td3_trainer = TD3HERRNN_Trainer(replay_buffer,
                    state_space, 
                    action_space, 
                    param_num=param_num,
                    goal_dim=goal_dim,
                    rnn_type=args.rnn,
                    out_actf=F.tanh,
                    action_scale=1.0 if 'aviary' in env_name else 10.0,
                    device=device, 
                    policy_target_update_interval=policy_target_update_interval,
                    **hparam)
    elif args.rnn == "None":
        replay_buffer = ReplayBuffer(replay_buffer_size)
        td3_trainer = TD3_Trainer(replay_buffer,
                    state_space, 
                    action_space, 
                    out_actf=F.tanh,
                    action_scale=1.0 if 'aviary' in env_name else 10.0,
                    device=device, 
                    policy_target_update_interval=policy_target_update_interval,
                    **hparam)
    else:
        raise "Something wrong"

    # keep track of progress
    mean_rewards = []
    scores_window = deque(maxlen=100)
    loss_storage = {"policy_loss":[],
                    "q_loss_1":[],
                    "q_loss_2":[],
                    'param_loss':[]}


    for i_episode in range(1,max_episodes+1):
        # print(i_episode)
        time_test = {"get_action":[0],"step":[0],"sample":[0],"tensor":[0],'q':[0],'p':[0],'q_target':[0],'backward':[0]}
        state, param = envs.reset()
        last_action = np.stack([envs.action_space.sample() for _ in range(nenvs)],axis=0).squeeze()
        last_action = np.zeros_like(last_action)
        episode_state = []
        episode_action = []
        episode_last_action = []
        episode_reward = []
        episode_next_state = []
        episode_done = []
        if "LSTM" in args.rnn:
            hidden_out = (torch.zeros([1, nenvs, hidden_dim], dtype=torch.float).to(device), \
                        torch.zeros([1, nenvs, hidden_dim], dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
        else:
            hidden_out = torch.zeros([1, nenvs, hidden_dim], dtype=torch.float).to(device)

        policy_loss = []
        q_loss_1,q_loss_2,param_loss = [],[],[]

        # Goal for HER
        if 'aviary' in env_name:
            thetas = [np.random.uniform(-np.pi,np.pi) for _ in range(nenvs)]
            goal = np.array([[0,0,0, # position
                            np.cos(theta),np.sin(theta),0, # rotation matrix
                            -np.sin(theta),np.cos(theta),0,
                            0,0,1] for theta in thetas]) # angular velocity
            # goal = np.array([[0,0,0, # position
            #                 np.cos(theta),np.sin(theta),0, # rotation matrix
            #                 -np.sin(theta),np.cos(theta),0,
            #                 0,0,1,
            #                 0,0,0, # velocity
            #                 0,0,0] for theta in thetas]) # angular velocity
            # goal = np.array([[0,0,0] for theta in thetas])
        else:
            goal = np.array([[1,0,0]]*nenvs)

        for step in range(max_steps):
            if args.rnn != "None":
                hidden_in = hidden_out
                t_start = time()
                if not "HER" in args.rnn:
                    if len(replay_buffer) > explore_steps:
                        action, hidden_out = \
                            td3_trainer.policy_net.get_action(state, 
                                                            last_action, 
                                                            hidden_in, 
                                                            deterministic=DETERMINISTIC, 
                                                            explore_noise_scale=explore_noise_scale)
                    else:
                        action = np.stack([envs.action_space.sample() for _ in range(nenvs)],axis=0).squeeze()
                else:
                    if len(replay_buffer) > explore_steps:
                        action, hidden_out = \
                            td3_trainer.policy_net.get_action(state, 
                                                            last_action, 
                                                            hidden_in,
                                                            goal=goal,
                                                            deterministic=DETERMINISTIC, 
                                                            explore_noise_scale=explore_noise_scale)
                    else:
                        action = np.stack([envs.action_space.sample() for _ in range(nenvs)],axis=0).squeeze()
                time_test['get_action'].append(time()-t_start)
                t_start = time()
                next_state, reward, done, _ = envs.step(action) 
                time_test['step'].append(time()-t_start)
                if step == 0:
                    ini_hidden_in = hidden_in
                    ini_hidden_out = hidden_out
                episode_state.append(state)
                episode_action.append(action)
                episode_last_action.append(last_action)
                episode_reward.append(reward)
                episode_next_state.append(next_state)
                episode_done.append(done)
            else:
                if len(replay_buffer) > explore_steps:
                    action = \
                        td3_trainer.policy_net.get_action(state, 
                                                        deterministic=DETERMINISTIC, 
                                                        explore_noise_scale=explore_noise_scale)
                else:
                        action = np.stack([envs.action_space.sample() for _ in range(nenvs)],axis=0).squeeze()
                next_state, reward, done, _ = envs.step(action)
                replay_buffer.push_batch(state, action, reward, next_state, done)
                episode_reward.append(reward)
                episode_state.append(state)

            state = next_state
            last_action = action
            frame_idx += 1
            
            if len(replay_buffer) > explore_steps:
                for i in range(update_itr):
                    # if "HER" in args.rnn:
                    #     if i_episode >= her_pre_steps:
                    #         if not td3_trainer.use_her:
                    #             td3_trainer.use_her = True
                    #             td3_trainer.reset_critic() # reset critic
                    # if "HER" in args.rnn:
                    #     td3_trainer.replay_buffer.gamma = 1-i_episode/max_episodes
                    loss_dict = td3_trainer.update(batch_size, deterministic=DETERMINISTIC, eval_noise_scale=eval_noise_scale)
                    # policy_loss, q_loss_1, q_loss_2
                    policy_loss.append(loss_dict['policy_loss'])
                    q_loss_1.append(loss_dict['q_loss_1'])
                    q_loss_2.append(loss_dict['q_loss_2'])
                    if 'fast' in args.rnn:
                        param_loss.append(loss_dict['param_loss'])
        
        time_test = {k:"%.2f"%np.sum(v) for k,v in time_test.items()}
        if len(replay_buffer) > explore_steps:
            time_test
        
        if args.rnn != "None": 
            if args.rnn in ['RNN', "LSTM", "GRU"]:
                replay_buffer.push_batch(ini_hidden_in, 
                                ini_hidden_out, 
                                episode_state, 
                                episode_action, 
                                episode_last_action,
                                episode_reward, 
                                episode_next_state, 
                                episode_done)
            elif args.rnn in ['RNNHER', "LSTMHER", "GRUHER"]:
                replay_buffer.push_batch(ini_hidden_in, 
                                ini_hidden_out, 
                                episode_state, 
                                episode_action, 
                                episode_last_action,
                                episode_reward, 
                                episode_next_state, 
                                episode_done,
                                param,
                                goal)
            else:           
                replay_buffer.push_batch(ini_hidden_in, 
                                ini_hidden_out, 
                                episode_state, 
                                episode_action, 
                                episode_last_action,
                                episode_reward, 
                                episode_next_state, 
                                episode_done,
                                param)

        loss_storage['policy_loss'].append(np.mean(policy_loss))
        loss_storage['q_loss_1'].append(np.mean(q_loss_1))
        loss_storage['q_loss_2'].append(np.mean(q_loss_2))
        if 'fast' in args.rnn:
            loss_storage['param_loss'].append(np.mean(param_loss))

        rewards = np.sum(episode_reward)/nenvs
        mean_rewards.append(rewards)
        scores_window.append(rewards)

        writer_iterval = 10
        if writer is not None and i_episode%writer_iterval == 0:
            writer.add_scalar('loss/loss_p', np.mean(loss_storage['policy_loss'][-writer_iterval:]), i_episode)
            writer.add_scalar('loss/loss_q_1', np.mean(loss_storage['q_loss_1'][-writer_iterval:]), i_episode)
            writer.add_scalar('loss/loss_q_2', np.mean(loss_storage['q_loss_2'][-writer_iterval:]), i_episode)
            if 'fast' in args.rnn:
                writer.add_scalar('loss/loss_param', np.mean(loss_storage['param_loss'][-writer_iterval:]), i_episode)
            writer.add_scalar('rewards', scores_window[-1], i_episode)
            
            if args.lr_scheduler:
                writer.add_scalar('lr/q_lr',td3_trainer.scheduler_q1.get_last_lr()[0], i_episode)
                writer.add_scalar('lr/policy_lr',td3_trainer.scheduler_policy.get_last_lr()[0], i_episode)

            for name, weight in td3_trainer.policy_net.named_parameters():
                writer.add_histogram(f'policy_net/{name}', weight, i_episode)
                if weight.grad is not None:
                    writer.add_histogram(f'policy_net/{name}.grad', weight.grad, i_episode)

            for name, weight in td3_trainer.q_net1.named_parameters():
                writer.add_histogram(f'q_net/{name}', weight, i_episode)
                if weight.grad is not None:
                    writer.add_histogram(f'q_net/{name}.grad', weight.grad, i_episode)
            
            if 'fast' in args.rnn:
                if hasattr(td3_trainer, 'scheduler_param'):
                    if args.lr_scheduler:
                        writer.add_scalar('lr/param_lr',td3_trainer.scheduler_param.get_last_lr()[0], i_episode)
                for name, weight in td3_trainer.param_net.named_parameters():
                    writer.add_histogram(f'policy_net/{name}', weight, i_episode)
                    if weight.grad is not None:
                        writer.add_histogram(f'policy_net/{name}.grad', weight.grad, i_episode)

            if 'aviary' in env_name:
                writer.add_scalar('loss/position[m]', np.linalg.norm((6*np.stack(episode_state)[:,:,:3]), axis=-1).mean(), i_episode)
                writer.add_scalar('loss/velocity[m/s]', np.linalg.norm((3*np.stack(episode_state)[:,:,12:15]), axis=-1).mean(), i_episode)
                writer.add_scalar('loss/ang_velocity[deg/s]', np.linalg.norm((2*180*np.stack(episode_state)[:,:,15:18]), axis=-1).mean(), i_episode)
                writer.add_scalar('loss/angle[deg]', 180/np.pi*np.arccos(np.stack(episode_state)[:,:,11].flatten()).mean(), i_episode)

            if i_episode % eval_freq == 0 and i_episode != 0:
                eval_rew, eval_success = evaluation(env_name, agent=td3_trainer, dyn_range=dyn_range, eval_itr=eval_itr, seed=i_episode)
                writer.add_scalar('eval/reward', eval_rew, i_episode)
                writer.add_scalar('eval/success_rate', eval_success, i_episode)

        if i_episode % 500 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            td3_trainer.save_model(os.path.join(savepath,"iter%05d"%i_episode))

        if np.mean(scores_window)>=best_score: 
            td3_trainer.save_model(os.path.join(savepath,"best"))
            best_score = np.mean(scores_window)
        
    td3_trainer.save_model(os.path.join(savepath,"final"))
    print('\rFinal\tAverage Score: {:.2f}'.format(np.mean(scores_window)))

    envs.close()
    del envs
    
    return mean_rewards, loss_storage, eval_rew, eval_success, dtime

def test(args):
    env_name = args.env

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
        'mass_range': 0.0, # (1-n) ~ (1+n)
        'cm_range': 0.0, # (1-n) ~ (1+n)
        'kf_range': 0.0, # (1-n) ~ (1+n)
        'km_range': 0.0, # (1-n) ~ (1+n)
        'i_range': 0.0,
        'battery_range': 0.0 # (1-n) ~ (1)
    }
    
    device=torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    # hyper-parameters for RL training
    hparam = {'hidden_dim': 128}

    if 'Pendulum' in env_name:
        param_num = 3
    elif 'aviary' in env_name:
        param_num = 14
    else:
        raise NotImplementedError

    # Define environment
    envs = domainRandeEnv(env_name=env_name, dyn_range=dyn_range)
    action_space = envs.action_space
    state_space = envs.observation_space
    envs.close()

    if args.rnn in ["RNN", "LSTM", "GRU"]:
        if args.rnn=='LSTM':
            replay_buffer = ReplayBufferLSTM(1e6)
        else:
            replay_buffer = ReplayBufferGRU(1e6)
        replay_buffer = ReplayBufferLSTM(1e6)
        td3_trainer = TD3RNN_Trainer(replay_buffer,
                    state_space, 
                    action_space, 
                    rnn_type=args.rnn,
                    out_actf=F.tanh,
                    action_scale=1.0 if 'aviary' in env_name else 10.0,
                    device=device, 
                    **hparam)
    elif args.rnn in ["RNN2", "LSTM2", "GRU2"]:
        if args.rnn=='LSTM2':
            replay_buffer = ReplayBufferFastAdaptLSTM(1e6)
        else:
            replay_buffer = ReplayBufferFastAdaptGRU(1e6)
        td3_trainer = TD3RNN_Trainer2(replay_buffer,
                    state_space, 
                    action_space, 
                    param_num=param_num,
                    rnn_type=args.rnn,
                    out_actf=F.tanh,
                    action_scale=1.0 if 'aviary' in env_name else 10.0,
                    device=device, 
                    **hparam)
    elif args.rnn in ["RNN3", "LSTM3", "GRU3"]:
        if args.rnn=='LSTM3':
            replay_buffer = ReplayBufferFastAdaptLSTM(1e6)
        else:
            replay_buffer = ReplayBufferFastAdaptGRU(1e6)
        td3_trainer = TD3RNN_Trainer3(replay_buffer,
                    state_space, 
                    action_space, 
                    param_num,
                    rnn_type=args.rnn,
                    out_actf=F.tanh,
                    action_scale=1.0 if 'aviary' in env_name else 10.0,
                    device=device, 
                    **hparam)
    elif args.rnn in ["RNNHER", "LSTMHER", "GRUHER"]:
        if args.rnn=='LSTMHER':
            replay_buffer = HindsightReplayBufferLSTM(1e5, 
                                sample_length=100)
        else:
            replay_buffer = HindsightReplayBufferGRU(1e5, 
                                epsilon=np.sqrt(3*(0.1**2)))
                                # sample_length=her_sample_length)
        # For aviary, it include the last action in the state
        # goal_dim = state_space.shape[0]-4 if 'aviary' in env_name else state_space.shape[0]
        goal_dim = 3 if 'aviary' in env_name else state_space.shape[0]
        td3_trainer = TD3HERRNN_Trainer(replay_buffer,
                    state_space, 
                    action_space, 
                    param_num=param_num,
                    goal_dim=goal_dim,
                    rnn_type=args.rnn,
                    out_actf=F.tanh,
                    action_scale=1.0 if 'aviary' in env_name else 10.0,
                    device=device, 
                    policy_target_update_interval=1,
                    **hparam)
    elif args.rnn == "None":
        replay_buffer = ReplayBuffer(1e6)
        td3_trainer = TD3_Trainer(replay_buffer,
                    state_space, 
                    action_space, 
                    out_actf=F.tanh,
                    action_scale=1.0 if 'aviary' in env_name else 10.0,
                    device=device, 
                    **hparam)
    else:
        raise "Something wrong!!"

    td3_trainer.load_model(args.path)
    
    if 'aviary' in env_name:
        eval_rew, eval_success, eval_position, eval_angle = drone_test(env_name, task=args.task, agent=td3_trainer, dyn_range=dyn_range, test_itr=10, seed=0, record=args.record)
        print("EVALUATION REWARD:",eval_rew)
        print("EVALUATION SUCCESS RATE:",eval_success)
        print("EVALUATION POSITION ERROR[m]:",eval_position)
        print("EVALUATION ANGLE ERROR[deg]:",np.rad2deg(eval_angle))
    else:
        eval_rew, eval_success = evaluation(env_name, agent=td3_trainer, dyn_range=dyn_range, eval_itr=5, seed=0)
        print("EVALUATION REWARD:",eval_rew)
        print("EVALUATION SUCCESS RATE:",eval_success)
        generate_result(env_name, td3_trainer, dyn_range, test_itr=5, seed=0, record=args.record)
    # evaluation(env_name, td3_trainer, dyn_range, 1, 0)

hparam_set = {
    # "q_lr": [3e-4, 1e-4, 3e-5],
    # "policy_lr": [3e-4, 1e-4, 3e-5],
    # "param_lr": [3e-4, 1e-4, 3e-5],
    # "t_max": [10000, 30000, 100000],
    # "hidden_dim": [64,128,256],
    # "update_interval": [2,5,10]

    "q_lr": [3e-4],
    "policy_lr": [3e-5],
    "param_lr": [3e-5],
    "t_max": [50000, 30000, 100000],
    "hidden_dim": [128],
    "update_interval": [3,2,4,5]
}

if __name__=='__main__':
    # train(1000, 'CartPole-v1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True, choices=['Pendulum-v0','HalfCheetah-v2','takeoff-aviary-v0'])
    parser.add_argument('--gpu', default='0', type=int, help="gpu number")
    parser.add_argument('--rnn', choices=['None','RNN','GRU','LSTM',
                                            'RNN2','GRU2','LSTM2',
                                            'RNN3','GRU3','LSTM3',
                                            'RNNHER','GRUHER','LSTMHER'], default='None', help='Use memory network (LSTM)')

    parser.add_argument('--tb_log', action='store_true', help="Tensorboard logging")
    parser.add_argument('--hparam', action='store_true', help="find hparam set")
    parser.add_argument('--lr_scheduler', action='store_true', help="Use lr scheduler")

    parser.add_argument('--test', action='store_true')
    parser.add_argument('--path', type=str, default=None, help='required only at test phase')
    parser.add_argument('--record', action='store_true', help='whether record or not')
    parser.add_argument('--task', default='stabilize',choices=['stabilize','takeoff','waypoint'],
                        help='For takeoff-aviary-v0 environment')


    args = parser.parse_args()
    if not args.test:
        if args.hparam:
            if not os.path.isdir('hparamDB'):
                os.makedirs('hparamDB')

            df_hparam = pd.DataFrame()

            # random hparam search
            # hparam_list = [dict([(k,random.choice(v)) for k,v in hparam_set.items()]) for _ in range(10)]
            
            # all hparam permutation search
            keys, values = zip(*hparam_set.items())
            hparam_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
            print("Total hparam test:",len(hparam_list))

            for i_hparam, hparam in enumerate(hparam_list):
                print("%dth hparam test"%i_hparam)
                hparam['lr_scheduler'] = args.lr_scheduler
                mean_rewards, loss_storage, eval_rew, eval_success, dtime = train(args, hparam)
                hparam['mean_reward'] = np.mean(mean_rewards)
                hparam['dtime'] = dtime.strftime("%y%b%d%H%M%S")
                hparam['final_eval_reward'] = eval_rew
                hparam['final_eval_success_rate'] = eval_success
                df_hparam = df_hparam.append([hparam])
                df_hparam.to_csv("hparamDB/hparam_test_%s.csv"%args.rnn)
        else:
            hparam = dict([(k,v[0]) for k,v in hparam_set.items()])
            hparam['lr_scheduler'] = args.lr_scheduler
            train(args, hparam)
    else:
        if args.path is None:
            assert "model path is required"
        test(args)