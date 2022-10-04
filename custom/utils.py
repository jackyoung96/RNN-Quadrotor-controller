import os
import shutil

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback

import gym
import numpy as np
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from envs.singleEnv.customEnv import customAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
import torch

def make_env(gui=False,record=False, **kwargs):
    env = gym.make(id="takeoff-aviary-v0", # arbitrary environment that has state normalization and clipping
                    drone_model=DroneModel.CF2X,
                    initial_xyzs=np.array([[0.0,0.0,1.0]]),
                    initial_rpys=np.array([[0.0,0.0,0.0]]),
                    physics=Physics.PYB_GND_DRAG_DW,
                    freq=240,
                    aggregate_phy_steps=1,
                    gui=gui,
                    record=record, 
                    obs=ObservationType.KIN,
                    act=ActionType.RPM)
    env = customAviary(env, **kwargs)

    return env

def angular_velocity(R, dt):
    R0, R1 = R
    A = np.matmul(R1, R0.transpose())
    theta = np.arccos((np.trace(A)-1)/2)
    W = 1/(2*(dt)) * (theta/np.sin(theta)) * (A-A.transpose())
    return np.array([W[2,1], W[0,2], W[1,0]]) 

def motorRun(cf, thrust):
    for i in range(4):
        cf.param.set_value("motorPowerSet.m%d"%(i+1), thrust[i])