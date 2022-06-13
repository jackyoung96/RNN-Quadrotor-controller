from re import I
import numpy as np
import torch
import torch.nn.functional as F

from td3.td3 import *
from td3.common.buffers import *
from td3.agent import td3_agent

import argparse
from pyvirtualdisplay import Display
import numpy as np
import gym
import pandas as pd
import os

from envs.customEnvDrone import domainRandomAviary
from envs.customEnv import VecDynRandEnv
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

def getgoal(waypoints, step):
    idx = len(waypoints)-1
    for i, (waypoint, t) in enumerate(waypoints):
        if t > step and i < idx:
            idx = i

    return waypoints[idx][0]
    if step >= waypoints[-1][1]:
        return waypoints[idx][0]
    else:
        now = waypoints[idx]
        prev = waypoints[idx-1]
        return (now[0]*(now[1]-step) + prev[0]*(step-prev[1]))/(now[1]-prev[1])
        
class dummyEnv:
    def __init__(self, env, env_name, dyn_range):
        self.env = DummyVecEnv([lambda: env])
        self.env_name = env_name

    def reset(self):
        param = self.env.envs[0].random_urdf()
        return self.env.reset(), param[None,:]

    def step(self, action):
        return self.env.step(action)
    
    def close(self):
        self.env.close()

def main(hparam):
    disp = Display(visible=False, size=(100, 60))
    disp.start()

    env_name = "takeoff-aviary-v0"
    max_steps = 100  # 8.5 sec

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    if hparam['task'] == 'random':
        dyn_range = {
            # drones
            'mass_range': 0.3, # (1-n) ~ (1+n)
            'cm_range': 0.3, # (1-n) ~ (1+n)
            'kf_range': 0.3, # (1-n) ~ (1+n)
            'km_range': 0.3, # (1-n) ~ (1+n)
            'i_range': 0.3,
            'battery_range': 0.3 # (1-n) ~ (1)
        }
    elif hparam['task'] == 'normal':
        dyn_range = {}
    else:
        raise NotImplementedError

    # waypoints = [
    #     (np.array([[0,0,1.0]]),0), # (pos, time)
    #     (np.array([[0.5,0,1.0]]),400),
    #     # (np.array([[0.5,0.5,1.0]]),800),
    #     # (np.array([[0,0.5,1.0]]),1200),
    #     # (np.array([[0,0,1.0]]),1600)
    # ]
    waypoints = [
        (np.array([[0,0,0.025]]),0),
        (np.array([[0,0,1.025]]),400), # (pos, time)
        # (np.array([[0.5,0,1.025]]),800),
        # (np.array([[0,0,1.025]]),1200),
        # (np.array([[0,0.5,1.025]]),1600),
        # (np.array([[0,0,1.025]]),2000)
    ]
    waypoints = [
        (np.array([[0,0,2.025]]),0),
        (np.array([[0,0,2.3]]),400), # (pos, time)
        # (np.array([[0.5,0,1.025]]),800),
        # (np.array([[0,0,1.025]]),1200),
        # (np.array([[0,0.5,1.025]]),1600),
        # (np.array([[0,0,1.025]]),2000)
    ]

    # max_steps = waypoints[-1][1]

    # Define environment
    theta = np.random.uniform(0,2*np.pi)
    env = gym.make(id=env_name, # arbitrary environment that has state normalization and clipping
        drone_model=DroneModel.CF2X,
        initial_xyzs=waypoints[0][0],
        initial_rpys=np.array([[0.0,0.0,theta]]),
        physics=Physics.PYB_GND_DRAG_DW,
        freq=200,
        aggregate_phy_steps=1,
        gui=False,
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM)
    env = domainRandomAviary(env, "testCoRL", 0, 9999999,
        observable=['pos', 'rotation', 'vel', 'angular_vel', 'rpm'],
        frame_stack=1,
        task='stabilize2',
        # reward_coeff={'pos':0.2, 'vel':0.0, 'ang_vel':0.02, 'd_action':0.01},
        reward_coeff={'pos':0.2, 'vel':0.016, 'ang_vel':0.005, 'd_action':0.002},
        episode_len_sec=max_steps/200,
        max_rpm=66535,
        initial_xyzs=waypoints[0][0], # Far from the ground
        initial_rpys=np.array([[0.0,0.0,theta]]),
        freq=200,
        rpy_noise=0,
        vel_noise=0,
        angvel_noise=0,
        mass_range=dyn_range.get('mass_range', 0.0),
        cm_range=dyn_range.get('cm_range', 0.0),
        kf_range=dyn_range.get('kf_range', 0.0),
        km_range=dyn_range.get('km_range', 0.0),
        i_range=dyn_range.get('i_range', 0.0),
        battery_range=dyn_range.get('battery_range', 0.0))
    setattr(env, 'env_name', env_name)
    env = dummyEnv(env, env_name, dyn_range)    

    agent = td3_agent(env=env,
                rnn=hparam['rnn'],
                device=device,
                hparam=hparam)
    agent.load_model(hparam['path'])
    agent.policy_net.eval()

    DETERMINISTIC=True  # DDPG: deterministic policy gradient      
    
    eval_success = 0
    eval_reward = 0
    eval_position = 0 
    eval_angle = 0

    theta = np.random.uniform(0,2*np.pi)
    goal = np.array([[0,0,0, # pos
                    np.cos(theta),-np.sin(theta),0,
                    np.sin(theta),np.cos(theta),0,
                    0,0,1, # rotation matrix
                    0,0,0, # vel
                    0,0,0, # ang vel
                    0,0,0,0]]) # dummy action

    with torch.no_grad():
        env.env.envs[0].goal = getgoal(waypoints, 0)
        state, param = env.reset()

        last_action = env.env.action_space.sample()[None,:]
        last_action = -np.ones_like(last_action)
        if 'LSTM' == hparam['rnn']:
            hidden_out_zero = (torch.zeros([1, 1, hparam['hidden_dim']], dtype=torch.float).to(device), \
                        torch.zeros([1, 1, hparam['hidden_dim']], dtype=torch.float).to(device))
        else:
            hidden_out_zero = torch.zeros([1, 1, hparam['hidden_dim']], dtype=torch.float).to(device)
        hidden_out = hidden_out_zero

        step, success = 0,0
        e_ps, e_as = [],[]
        state_buffer, action_buffer, reward_buffer = [],[],[]
        critic_buffer = []
        drone_state_buffer = pd.DataFrame()

        for i_step in range(max_steps):
            goal_pos = getgoal(waypoints, i_step)
            if np.any(env.env.envs[0].goal_pos-goal_pos):
                env.env.envs[0].goal_pos = goal_pos
                hidden_out = hidden_out_zero
                last_action = -np.ones_like(last_action)

            if getattr(agent, 'rnn_type', 'None') in ['GRU','RNN','LSTM']:
                hidden_in = hidden_out
                if not hasattr(agent.q_net1, '_goal_dim'):
                    action, hidden_out = \
                        agent.policy_net.get_action(state, 
                                                        last_action, 
                                                        hidden_in, 
                                                        deterministic=DETERMINISTIC, 
                                                        explore_noise_scale=0.0)
                else:
                    action, hidden_out = \
                        agent.policy_net.get_action(state, 
                                                        last_action, 
                                                        hidden_in, 
                                                        goal=goal,
                                                        deterministic=DETERMINISTIC, 
                                                        explore_noise_scale=0.0)
            else:
                action = agent.policy_net.get_action(state, 
                                                    last_action,
                                                    deterministic=DETERMINISTIC, 
                                                    explore_noise_scale=0.0)
            
            next_state, reward, done, _ = env.step(action) 

            # critic_test = agent.q_net1(torch.Tensor(state[None,:]).to(device), torch.Tensor(action[None,:]).to(device), torch.Tensor(param[None,:]).to(device), torch.Tensor(goal[None,:]).to(device)).detach().cpu().item()
            # critic_buffer.append(critic_test)

            e_p = np.linalg.norm(6*next_state[0,:3]) # position (m)
            e_a = np.rad2deg(np.arccos(np.clip(next_state[0,11], -1.0, 1.0))) # angle (deg)
            e_ps.append(e_p)
            e_as.append(e_a)
            
            
            drone_state_buffer = drone_state_buffer.append(
                {
                    't': i_step/200,
                    'x': env.env.envs[0].drone_state()[0],
                    'y': env.env.envs[0].drone_state()[1],
                    'z': env.env.envs[0].drone_state()[2]
                }, ignore_index=True
            )

            state_buffer.append(state)
            action_buffer.append(action)
            if not isinstance(action, np.ndarray):
                action = np.array([action])

            state, last_action = next_state, action

        eval_position = np.mean(e_ps)
        eval_angle = np.mean(e_as)
        
        if e_p < np.linalg.norm([0.1,0.1,0.1]) and e_a < 10:
            eval_success = 1

    drone_state_buffer.to_csv('paperworks/%s.csv'%(hparam['rnn']+hparam['task']), header=False)
    print("EVALUATION SUCCESS RATE:", eval_success)
    print("EVALUATION POSITION ERROR[m]:", eval_position)
    print("EVALUATION ANGLE ERROR[deg]:", np.rad2deg(eval_angle))

    print("EVALUATION ANGVEL ERROR[deg/s]:", np.linalg.norm((2*180*np.stack(state_buffer)[:,:,15:18]), axis=-1).mean())

    disp.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Common arguments
    parser.add_argument('--rnn', choices=['None','RNN2','GRU2','LSTM2',
                                            'RNNHER','GRUHER','LSTMHER',
                                            'RNNsHER','GRUsHER','LSTMsHER']
                                , default='None', help='Use memory network (LSTM)')
    parser.add_argument('--path', type=str, default=None, help='required only at test phase')
    parser.add_argument('--task', default='normal',choices=['randomize', 'normal'],
                        help='For takeoff-aviary-v0 environment')
    args = parser.parse_args()

    hparam = {
        "goal_dim": 18,
        "param_num": 14,
        "hidden_dim": 40,
        "policy_actf": F.tanh,
    }
    hparam.update(vars(args))
    main(hparam)