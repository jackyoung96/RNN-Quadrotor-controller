import torch
import gym
import matplotlib
from envs.customEnv import domainRandomize
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
from envs.customEnvDrone import domainRandomAviary

from td3.td3 import *
from td3.common.buffers import *

import argparse
import os
from utils import save_frames_as_gif
from pyvirtualdisplay import Display
import numpy as np
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def evaluation(env_name, agent, dyn_range, eval_itr, seed):

    # hyper-parameters for RL training
    DETERMINISTIC=True  # DDPG: deterministic policy gradient      
    
    device = agent.device
    eval_success = 0
    eval_reward = 0
    if hasattr(agent.q_net1, '_goal_dim'):
        goal_dim=agent.q_net1._goal_dim
    
    if 'aviary' in env_name:
        max_steps=500
    else:
        max_steps=900
    
    with torch.no_grad():
        for i_eval in range(eval_itr):
            if 'aviary' in env_name:
                eval_env = gym.make(id=env_name, # arbitrary environment that has state normalization and clipping
                    drone_model=DroneModel.CF2X,
                    initial_xyzs=np.array([[0.0,0.0,10000.0]]),
                    initial_rpys=np.array([[0.0,0.0,0.0]]),
                    physics=Physics.PYB_GND_DRAG_DW,
                    freq=200,
                    aggregate_phy_steps=1,
                    gui=False,
                    record=False, 
                    obs=ObservationType.KIN,
                    act=ActionType.RPM)
                eval_env = domainRandomAviary(eval_env, 'test', 0, seed+i_eval,
                    observable=['pos', 'rotation', 'vel', 'angular_vel', 'rpm'],
                    frame_stack=1,
                    task='stabilize2',
                    reward_coeff={'pos':0.2, 'vel':0.0, 'ang_vel':0.02, 'd_action':0.01},
                    episode_len_sec=max_steps/200,
                    max_rpm=66535,
                    initial_xyzs=[[0.0,0.0,10000.0]], # Far from the ground
                    freq=200,
                    rpy_noise=np.pi/4,
                    vel_noise=2.0,
                    angvel_noise=np.pi/2,
                    mass_range=dyn_range.get('mass_range', 0.0),
                    cm_range=dyn_range.get('cm_range', 0.0),
                    kf_range=dyn_range.get('kf_range', 0.0),
                    km_range=dyn_range.get('km_range', 0.0),
                    i_range=dyn_range.get('i_range', 0.0),
                    battery_range=dyn_range.get('battery_range', 0.0))
            else:
                eval_env = gym.make(env_name)
                eval_env.seed(seed+i_eval)
            setattr(eval_env,'env_name', env_name)
            domainRandomize(eval_env, dyn_range=dyn_range, seed=seed+i_eval)
            state = eval_env.reset()[None,:]
            total_rew = 0
            last_action = eval_env.action_space.sample()[None,:]
            last_action = np.zeros_like(last_action)
            if hasattr(agent, 'rnn_type'):
                if 'LSTM' == agent.rnn_type:
                    hidden_out = (torch.zeros([1, 1, agent.hidden_dim], dtype=torch.float).to(device), \
                                torch.zeros([1, 1, agent.hidden_dim], dtype=torch.float).to(device))
                elif 'GRU' == agent.rnn_type or 'RNN' == agent.rnn_type:
                    hidden_out = torch.zeros([1, 1, agent.hidden_dim], dtype=torch.float).to(device)

            with torch.no_grad():
                total_step, step, success = 0,0,0
                if 'aviary' in env_name:
                    theta = np.random.uniform(-np.pi,np.pi)
                    goal = np.array([[0,0,0, # position
                                    np.cos(theta),np.sin(theta),0, # rotation matrix
                                    -np.sin(theta),np.cos(theta),0,
                                    0,0,1,
                                    0,0,0, # velocity
                                    0,0,0]]) # angular velocity
                else:
                    goal = np.array([[1,0,0]])

                for _ in range(max_steps):
                    if hasattr(agent, 'rnn_type'):
                        hidden_in = hidden_out
                        if not hasattr(agent.q_net1, '_goal_dim'):
                            action, hidden_out = \
                                agent.policy_net.get_action(state, 
                                                                last_action, 
                                                                hidden_in, 
                                                                deterministic=DETERMINISTIC, 
                                                                explore_noise_scale=0.)
                        else:
                            action, hidden_out = \
                                agent.policy_net.get_action(state, 
                                                                last_action, 
                                                                hidden_in, 
                                                                goal=goal,
                                                                deterministic=DETERMINISTIC, 
                                                                explore_noise_scale=0.)
                    else:
                        action = agent.policy_net.get_action(state, 
                                                            deterministic=DETERMINISTIC, 
                                                            explore_noise_scale=0.)
                    next_state, reward, done, _ = eval_env.step(action) 
                    # print("DEBUG")
                    # print("POS", state[0,:3])
                    # print("VEL", state[0,12:15])
                    # print("ANGVEL", state[0,15:18])
                    # print("REW", reward)
                    if not isinstance(action, np.ndarray):
                        action = np.array([action])
                    state, last_action = next_state[None,:], action[None,:]
                    total_step += 1

                    if "Pendulum" in env_name:
                        step = step+1 if state[0,0] > np.cos(5 * np.pi/180) else 0
                        if step > 100:
                            success = 1
                            break
                    elif "aviary" in env_name:
                        # 1/6 scaling ->  meter unit 
                        if np.linalg.norm(6*state[0,:3]) < np.linalg.norm([0.1]*3) and\
                            np.arccos(state[0,11]) < 10*np.pi/180:
                            step = step+1 
                        else:
                            step = 0
                        if step > 100:
                            success = 1
                            break

                    total_rew += reward
            
            eval_env.close()
            del eval_env

            eval_success += success
            eval_reward += total_rew
            # print("%d iteration reward %.3f success %d"%(i_eval,total_rew,success))

    # print("total average reward %.3f success rate %d"%(eval_reward / eval_itr,eval_success / eval_itr))
    return eval_reward/eval_itr, eval_success/eval_itr

def generate_result(env_name, agent, dyn_range, test_itr, seed, record=False):
    if record:
        disp = Display(visible=False, size=(100, 60))
        disp.start()
        # hyper-parameters for RL training
    DETERMINISTIC=True  # DDPG: deterministic policy gradient      
    
    if 'Pendulum' in env_name:
        max_steps = 1000
    elif 'aviary' in env_name:
        max_steps = 400
    else:
        raise NotImplementedError

    device = agent.device
    eval_success = 0
    eval_reward = 0
    frames_all = []

    pd_param = pd.DataFrame()

    with torch.no_grad():
        for i_eval in range(test_itr):
            print("---------%02d------------"%i_eval)
            if 'aviary' in env_name:
                eval_env = gym.make(id=env_name, # arbitrary environment that has state normalization and clipping
                    drone_model=DroneModel.CF2X,
                    initial_xyzs=np.array([[0.0,0.0,1.5]]),
                    initial_rpys=np.array([[0.0,0.0,0.0]]),
                    physics=Physics.PYB_GND_DRAG_DW,
                    freq=240,
                    aggregate_phy_steps=1,
                    gui=False,
                    record=record, 
                    obs=ObservationType.KIN,
                    act=ActionType.RPM)
                eval_env = domainRandomAviary(eval_env, 'test', 0, seed+i_eval,
                    observable=['pos', 'rotation', 'vel', 'angular_vel', 'rpm'],
                    frame_stack=1,
                    task='stabilize2',
                    reward_coeff={'pos':0.2, 'vel':0.0, 'ang_vel':0.02, 'd_action':0.01},
                    episode_len_sec=max_steps/200,
                    max_rpm=66535,
                    initial_xyzs=[[0.0,0.0,1.5]], # Far from the ground
                    freq=200,
                    rpy_noise=np.pi/4,
                    vel_noise=2.0,
                    angvel_noise=np.pi/2,
                    mass_range=dyn_range.get('mass_range', 0.0),
                    cm_range=dyn_range.get('cm_range', 0.0),
                    kf_range=dyn_range.get('kf_range', 0.0),
                    km_range=dyn_range.get('km_range', 0.0),
                    i_range=dyn_range.get('i_range', 0.0),
                    battery_range=dyn_range.get('battery_range', 0.0))
            else:
                eval_env = gym.make(env_name)
                eval_env.seed(seed+i_eval)
            setattr(eval_env,'env_name', env_name)
            param = domainRandomize(eval_env, dyn_range=dyn_range, seed=seed+i_eval)
            state = eval_env.reset()[None,:]
            total_rew = 0
            last_action = eval_env.action_space.sample()[None,:]
            last_action = np.zeros_like(last_action)
            if hasattr(agent, 'rnn_type'):
                if 'LSTM' == agent.rnn_type:
                    hidden_out = (torch.zeros([1, 1, agent.hidden_dim], dtype=torch.float).to(device), \
                                torch.zeros([1, 1, agent.hidden_dim], dtype=torch.float).to(device))
                else:
                    hidden_out = torch.zeros([1, 1, agent.hidden_dim], dtype=torch.float).to(device)

            total_step, step, success = 0,0,0
            frames = []
            if 'aviary' in env_name:
                theta = np.random.uniform(-np.pi,np.pi)
                goal = np.array([[0,0,0, # position
                                np.cos(theta),np.sin(theta),0, # rotation matrix
                                -np.sin(theta),np.cos(theta),0,
                                0,0,1,
                                0,0,0, # velocity
                                0,0,0]]) # angular velocity
            else:
                goal = np.array([[1,0,0]])

            for i_step in range(max_steps):
                if getattr(agent, 'rnn_type', 'None') in ['GRU','RNN','LSTM']:
                    hidden_in = hidden_out
                    if not hasattr(agent.q_net1, '_goal_dim'):
                        action, hidden_out = \
                            agent.policy_net.get_action(state, 
                                                            last_action, 
                                                            hidden_in, 
                                                            deterministic=DETERMINISTIC, 
                                                            explore_noise_scale=0.)
                    else:
                        action, hidden_out = \
                            agent.policy_net.get_action(state, 
                                                            last_action, 
                                                            hidden_in, 
                                                            goal=goal,
                                                            deterministic=DETERMINISTIC, 
                                                            explore_noise_scale=0.)
                    if hasattr(agent, 'param_net'):
                        predict_param = agent.predict_param(hidden_in).squeeze()
                        pd_param = pd_param.append([{"episode": i_eval,
                                                    "step": i_step,
                                                    "param_gt": param,
                                                    "param_pred": predict_param,
                                                    "dparam":np.sum((predict_param-param)**2)}], ignore_index=True)
                else:
                    action = agent.policy_net.get_action(state, 
                                                        deterministic=DETERMINISTIC, 
                                                        explore_noise_scale=0.)
                if record:
                    frames.append(eval_env.render(mode="rgb_array"))
                next_state, reward, done, _ = eval_env.step(action) 
                if not isinstance(action, np.ndarray):
                    action = np.array([action])
                state, last_action = next_state[None,:], action[None,:]
                total_step += 1

                if "Pendulum" in env_name:
                    step = step+1 if state[0,0] > np.cos(5 * np.pi/180) else 0
                    if step > 100:
                        success = 1
                        break
                elif "aviary" in env_name:
                    # print(state[0,:3], reward)
                    if np.linalg.norm(6*state[0,:3]) < np.linalg.norm([0.1]*3) and\
                        np.arccos(state[0,11]) < 10*np.pi/180:
                        step = step+1 
                    else:
                        step = 0
                    if step > 100:
                        success = 1
                        break

                total_rew += reward
            
            if record:
                frames.extend([np.zeros_like(frames[-1])] * 20)
            frames_all.extend(frames)
            eval_env.close()

            eval_success += success
            eval_reward += total_rew

            print("%d iteration reward %.3f success %d"%(i_eval,total_rew,success))

    
    if record:
        if not os.path.isdir("gifs"):
            os.mkdir("gifs")
        num = 0
        for file in os.listdir("gifs"):
            if env_name+"_"+type(agent).__name__ in file:
                num = max(num,int(file.strip(".gif").split("_")[-1]) + 1)
        
        save_frames_as_gif(frames_all, path="gifs", filename="%s_%s_%03d.gif"%(env_name, type(agent).__name__, num))
        disp.stop()
    
    # plt.figure()
    if hasattr(agent, 'param_net'):
        pd_param.to_csv(os.path.join('gifs',"%s_%s_indi.csv"%(env_name, type(agent).__name__)))
        sns.lineplot(data=pd_param, x="step", y="dparam", hue="episode", style="episode")
        plt.savefig(os.path.join('gifs',"%s_%s_indi.png"%(env_name, type(agent).__name__)))
        plt.close()
        # plt.figure()
        sns.lineplot(data=pd_param, x="step", y="dparam")
        plt.savefig(os.path.join('gifs',"%s_%s_all.png"%(env_name, type(agent).__name__)))
        plt.close()
    
    print("total average reward %.3f success rate %d"%(eval_reward / test_itr,eval_success / test_itr))
    
    return eval_reward / test_itr, eval_success / test_itr