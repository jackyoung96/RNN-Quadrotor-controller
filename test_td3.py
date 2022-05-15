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

def drone_test(env_name, task, agent, dyn_range, test_itr=10, seed=0, record=False):
    if record:
        disp = Display(visible=False, size=(100, 60))
        disp.start()
        # hyper-parameters for RL training
    DETERMINISTIC=True  # DDPG: deterministic policy gradient      
    
    max_steps = 400

    device = agent.device
    eval_success = 0
    eval_reward = 0
    frames_all = []

    with torch.no_grad():
        for i_eval in range(test_itr):
            eval_env = gym.make(id=env_name, # arbitrary environment that has state normalization and clipping
                drone_model=DroneModel.CF2X,
                initial_xyzs=np.array([[0.0,0.0,1.5]]),
                initial_rpys=np.array([[0.0,0.0,0.0]]),
                physics=Physics.PYB_GND_DRAG_DW,
                freq=240,
                aggregate_phy_steps=1,
                gui=False,
                record=record if i_eval==test_itr-1 else False, 
                obs=ObservationType.KIN,
                act=ActionType.RPM)
            if task == 'stabilize':
                initial_xyzs = [[0.0,0.0,1.5]]
                rpy_noise=np.pi/4
                vel_noise=2.0
                angvel_noise=np.pi/2
                goals = None
                goal = None
            elif task == 'takeoff':
                initial_xyzs = [[0.0,0.0,0.025]]
                rpy_noise=0
                vel_noise=0
                angvel_noise=0
                goals = [[0.0,0.0,1.0]]
                goal = goals[0:1]
            elif task == 'waypoint':
                initial_xyzs = [[0.0,0.0,0.025]]
                rpy_noise=0
                vel_noise=0
                angvel_noise=0
                goals = [[0.0,0.0,1.0],
                        [1.0,0.0,1.0],
                        [1.0,1.0,1.0],
                        [0.0,1.0,1.0],
                        [0.0,0.0,1.0],
                        [0.0,0.0,0.025]]
                goal = goals[0:1]
            eval_env = domainRandomAviary(eval_env, 'test', 0, seed+i_eval,
                observable=['pos', 'rotation', 'vel', 'angular_vel', 'rpm'],
                frame_stack=1,
                task='stabilize2',
                reward_coeff={'pos':0.2, 'vel':0.0, 'ang_vel':0.02, 'd_action':0.01},
                episode_len_sec=max_steps/200,
                max_rpm=66535,
                initial_xyzs=initial_xyzs, # Far from the ground
                freq=200,
                rpy_noise=rpy_noise,
                vel_noise=vel_noise,
                angvel_noise=angvel_noise,
                mass_range=dyn_range.get('mass_range', 0.0),
                cm_range=dyn_range.get('cm_range', 0.0),
                kf_range=dyn_range.get('kf_range', 0.0),
                km_range=dyn_range.get('km_range', 0.0),
                i_range=dyn_range.get('i_range', 0.0),
                battery_range=dyn_range.get('battery_range', 0.0),
                goal = goal)
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

            step, success = 0,0
            goal_idx = 0
            frames = []

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
                                                            goal=np.array([[0,0,0]]),
                                                            deterministic=DETERMINISTIC, 
                                                            explore_noise_scale=0.)
                else:
                    action = agent.policy_net.get_action(state, 
                                                        deterministic=DETERMINISTIC, 
                                                        explore_noise_scale=0.)

                next_state, reward, done, _ = eval_env.step(action) 
                if not isinstance(action, np.ndarray):
                    action = np.array([action])
                state, last_action = next_state[None,:], action[None,:]
                total_rew += reward

                # print(state[0,:3], reward)
                if np.linalg.norm(6*state[0,:3]) < np.linalg.norm([0.1]*3) and\
                    np.arccos(state[0,11]) < 10*np.pi/180:
                    step = step+1 
                else:
                    step = 0
                if step > 20: # 0.1 second
                    goal_idx += 1
                    if goals is None or goal_idx >= len(goals):
                        success = 1
                        break
                    else:
                        eval_env.goal_pos = goals[goal_idx:goal_idx+1]

            eval_env.close()

            eval_success += success
            eval_reward += total_rew

            print("%d iteration reward %.3f success %d"%(i_eval,total_rew,success))

    
    if record:
        disp.stop()
    
    print("total average reward %.3f success rate %d"%(eval_reward / test_itr,eval_success / test_itr))
    
    return eval_reward / test_itr, eval_success / test_itr