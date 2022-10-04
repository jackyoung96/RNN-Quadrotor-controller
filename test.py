import torch
import gym
import matplotlib
from envs.customEnv import domainRandomize
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
from envs.customEnvDrone import domainRandomAviary

from td3.td3 import *
from td3.common.buffers import *
from td3.common.utils import rot_matrix_similarity

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
import time

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper

env_name = "takeoff-aviary-v0"
tag = "justtest"
idx = 0
seed = 0
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
    'kf_range': 0.3, # (1-n) ~ (1+n)
    'km_range': 0.3, # (1-n) ~ (1+n)
    'i_range': 0.3,
    'battery_range': 0.3 # (1-n) ~ (1)
}

class VecDynRandEnv(VecEnvWrapper):
    def __init__(self, venv: VecEnv):
        super().__init__(venv=venv, observation_space=venv.observation_space)

    def reset(self) -> np.ndarray:
        params = []
        for env in self.venv.venv.envs:
            params.append(env.random_urdf())
        params = np.stack(params, axis=0)
        obs = self.venv.reset()
        return obs, params

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        return obs, reward, done, info

env = gym.make(id=env_name, # arbitrary environment that has state normalization and clipping
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
env = domainRandomAviary(env, tag+str(time.time_ns()), idx, seed,
    observable=['pos', 'rotation', 'vel', 'angular_vel', 'rpm'],
    frame_stack=1,
    task='stabilize2',
    reward_coeff={'pos':0.2, 'vel':0.0, 'ang_vel':0.02, 'd_action':0.01},
    episode_len_sec=2,
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
setattr(env, 'env_name', env_name)

env0 = DummyVecEnv([lambda: env]*2)
env0 = VecNormalize(env0, norm_obs=True, norm_reward=False)
env0 = VecDynRandEnv(env0)

state, param = env0.reset()
print(env0.observation_space)

for i in range(10):
    state, param = env0.reset()
    print(param)
