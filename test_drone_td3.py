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
        self.last_action = -np.ones((4,))[None,:]

    def reset(self):
        param = self.env.envs[0].random_urdf()
        self.last_action = -np.ones((4,))[None,:]
        return self.env.reset(), param[None,:]

    def step(self, action):
        action = 4*(1/200)/0.15 * (action-self.last_action) + self.last_action
        self.last_action = action
        return self.env.step(action)

    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()

def main(hparam):
    # disp = Display(visible=False, size=(100, 60))
    # disp.start()

    env_name = "takeoff-aviary-v0"
    max_steps = 400  # 8.5 sec

    # device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("Device:",device)

    if 'random' in hparam['task']:
        dyn_range = {
            # drones
            'mass_range': 0.3, # (1-n) ~ (1+n)
            'cm_range': 0.3, # (1-n) ~ (1+n)
            'kf_range': 0.3, # (1-n) ~ (1+n)
            'km_range': 0.3, # (1-n) ~ (1+n)
            'i_range': 0.3,
            'battery_range': 0.3 # (1-n) ~ (1)
        }
    elif 'normal' in hparam['task']:
        dyn_range = {}
    else:
        raise NotImplementedError

    if 'waypoint' in hparam['task']:
        waypoints = [
            (np.array([[0,  0,  0.025]]),0),
            (np.array([[0,  0,  1.025]]),400), # (pos, time)
            (np.array([[0.5,0,  1.025]]),800),
            (np.array([[0.5,0.5,1.025]]),1200),
            (np.array([[0,  0.5,1.025]]),1600),
            (np.array([[0,  0,  1.025]]),2000)
        ]
        theta = np.random.uniform(0,2*np.pi)
        initial_rpys = np.array([[0.0,0.0,theta]])
        rpy_noise = 0
        vel_noise = 0
        angvel_noise = 0
        max_steps = waypoints[-1][1]
    elif 'stabilize' in hparam['task']:
        waypoints = [
            (np.array([[0,  0,  1.0]]),0),
            (None, 300), # (pos, time)
        ]
        initial_rpys = np.random.uniform(-np.pi/4, np.pi/4, size=(1,3))
        rpy_noise = np.pi
        vel_noise = 1.0
        angvel_noise = np.pi/2
        max_steps = waypoints[-1][1]
    else:
        raise NotImplementedError

    

    # Define environment
    theta = np.random.uniform(0,0)
    env = gym.make(id=env_name, # arbitrary environment that has state normalization and clipping
        drone_model=DroneModel.CF2X,
        initial_xyzs=waypoints[0][0],
        initial_rpys=initial_rpys,
        physics=Physics.PYB_GND_DRAG_DW,
        freq=200,
        aggregate_phy_steps=1,
        gui=True,
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM)
    env = domainRandomAviary(env, "testCoRL", 0, 9999999,
        observable=['pos', 'rotation', 'vel', 'angular_vel', 'rpm'],
        frame_stack=1,
        task='stabilize2',
        reward_coeff={'pos':0.0, 'vel':0.0, 'ang_vel':0.0, 'd_action':0.0, 'rotation': 0.0},
        episode_len_sec=max_steps/200,
        max_rpm=66535,
        initial_xyzs=waypoints[0][0], # Far from the ground
        initial_rpys=initial_rpys,
        freq=200,
        rpy_noise=rpy_noise,
        vel_noise=vel_noise,
        angvel_noise=angvel_noise,
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
        if 'LSTM' in hparam['rnn']:
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
            if goal_pos is not None and np.any(env.env.envs[0].goal_pos-goal_pos):
                env.env.envs[0].goal_pos = goal_pos
                # hidden_out = hidden_out_zero
                # last_action = -np.ones_like(last_action)

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
            env.render()
            input()

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

            state = next_state
            last_action = action


        eval_position = np.mean(e_ps)
        eval_angle = np.mean(e_as)
        
        if np.sum(np.where(np.array(e_as) < 10, 1, 0)[-100:]) == 100: 
            eval_success = 1

    drone_state_buffer.to_csv('paperworks/%s.csv'%(hparam['rnn']+hparam['task']), header=False)
    print("EVALUATION SUCCESS RATE:", eval_success)
    print("EVALUATION POSITION ERROR[m]:", eval_position)
    print("EVALUATION ANGLE ERROR[deg]:", np.rad2deg(eval_angle))

    print("EVALUATION ANGVEL ERROR[deg/s]:", np.linalg.norm((2*180*np.stack(state_buffer)[:,:,15:18]), axis=-1).mean())

    # disp.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Common arguments
    parser.add_argument('--rnn', choices=['None','RNN2','GRU2','LSTM2',
                                            'RNNHER','GRUHER','LSTMHER',
                                            'RNNsHER','GRUsHER','LSTMsHER']
                                , default='None', help='Use memory network (LSTM)')
    parser.add_argument('--path', type=str, default=None, help='required only at test phase')
    parser.add_argument('--task', default='normal-waypoint',
                                  choices=['normal-waypoint','random-waypoint',
                                            'normal-stabilize','random-stabilize'],
                                  help='For takeoff-aviary-v0 environment')
    args = parser.parse_args()

    hparam = {
        "goal_dim": 18,
        "param_num": 14,
        "hidden_dim": 48,
        "policy_actf": F.tanh,
    }
    hparam.update(vars(args))
    main(hparam)