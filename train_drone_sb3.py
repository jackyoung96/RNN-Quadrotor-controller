from email import policy
from turtle import write_docstringdict
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F

from td3.td3 import *
from td3.common.buffers import *
from td3.agent import td3_agent
from envs.customEnv import dynRandeEnv
from utils import wandb_artifact

import argparse
from pyvirtualdisplay import Display
import os
from torch.utils.tensorboard import SummaryWriter
import wandb
from datetime import datetime

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.monitor import Monitor

from time import time

# stable baseline 3
from wandb.integration.sb3 import WandbCallback
import stable_baselines3 as SB3
from stable_baselines3 import SAC, TD3, PPO
from sb3_contrib import RecurrentPPO

############################
####### Global Vars ########
############################

dyn_range = {
    # drones
    'mass_range': 0.3, # (1-n) ~ (1+n)
    'cm_range': 0.3, # (1-n) ~ (1+n)
    'kf_range': 0.3, # (1-n) ~ (1+n)
    'km_range': 0.3, # (1-n) ~ (1+n)
    'i_range': 0.3,
    'battery_range': 0.0 # (1-n) ~ (1)
}
hparam_set = {
    "learning_rate": (np.random.uniform,[-4, -3]),
    "learning_starts": (np.random.randint,[80000,80001]),
    "activation": (np.random.choice, [[torch.nn.ReLU]]),

    # PPO
    # "n_steps": (np.random.randint,[4,11]),
    "n_steps": (np.random.randint,[800,801]),

    # SAC, TD3
    "update_itr": (np.random.randint,[1,11]),

    "goal_dim": (np.random.randint,[18,19]),
    "param_num": (np.random.randint,[14,15]),
    "hidden_dim": (np.random.randint,[5,8]),

    "max_steps": (np.random.randint,[800,801]),
    "her_length": (np.random.randint,[800,801]),
    "rnn_dropout": (np.random.uniform,[0, 0])
}

def train(args, hparam):

    #####################################
    # hyper-parameters for RL training ##
    #####################################

    max_episodes  = int(1e4)
    max_steps = hparam['max_steps']

    hparam['learning_rate'] = 10**hparam['learning_rate']
    hparam['hidden_dim'] = int(2**hparam['hidden_dim'])
    hidden_dim = hparam['hidden_dim']
    observable = ['rel_pos', 'rotation', 'rel_vel', 'rel_angular_vel']
    rew_coeff = {'pos':1.0, 'vel':0.0, 'ang_vel':0.1, 'd_action':0.05, 'rotation': 0.0}
    hparam['observable'] = observable
    hparam['rew_coeff'] = rew_coeff
    # hparam['n_steps'] = 2**hparam['n_steps']

    batch_size  = 128
    nenvs = 1
    
    #########################################
    ### Path, Basic variable setting ########
    #########################################

    print("hyperparam set:",hparam)
    algorithm_name = hparam['model']
    env_name = "takeoff-aviary-v0"
    dtime = datetime.now()
    
    # tensorboard
    now = dtime.strftime("%y%b%d%H%M%S")
    if args.tb_log:

        # wandb
        run = wandb.init(project="SB3-drone", config=hparam,
                        sync_tensorboard=True,
                        save_code=True)
        wandb.run.name = "%s_%s"%(algorithm_name, now)
        wandb.run.save()
    
    if args.gpu >= 0:
        device=torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
    else:
        device=torch.device('cpu')
    
    print("Device:",device)

    ####################################
    # Define environment and agent #####
    ####################################

    env = dynRandeEnv(
        initial_xyzs=np.array([[0,0,10000]]),
        initial_rpys=np.array([[0,0,0]]),
        observable=observable,
        dyn_range=dyn_range,
        rpy_noise=np.pi,
        vel_noise=2,
        angvel_noise=2*np.pi,
        reward_coeff=rew_coeff,
        frame_stack=1,
        episode_len_sec=max_steps/200,
        gui=False,
        record=False,
    )
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=hparam['obs_norm'], norm_reward=hparam['rew_norm'])
    
    if hparam['model']=='SAC':
        policy_kwargs = dict(activation_fn=hparam['activation'],
                     net_arch=dict(pi=[hidden_dim]*4, qf=[128]*4))
        trainer = SAC('MlpPolicy', env, verbose=0, device=device,
                batch_size=batch_size,
                learning_rate=hparam['learning_rate'],
                learning_starts=hparam['learning_starts'],
                train_freq=hparam['update_itr'],
                policy_kwargs=policy_kwargs,
                tensorboard_log=f"runs/{run.id}" if hparam['tb_log'] else None
        )
        total_timesteps = max_episodes*max_steps
    elif hparam['model']=='PPO':
        policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                     net_arch=[dict(pi=[hidden_dim]*4, vf=[128]*4)])
        trainer = PPO('MlpPolicy', env, verbose=0, device=device,
                n_steps=hparam['n_steps'],
                batch_size=batch_size,
                learning_rate=hparam['learning_rate'],
                policy_kwargs=policy_kwargs,
                tensorboard_log=f"runs/{run.id}" if hparam['tb_log'] else None
        )
        total_timesteps = max_episodes*max_steps
    elif hparam['model']=='TD3':
        policy_kwargs = dict(activation_fn=hparam['activation'],
                     net_arch=dict(pi=[hidden_dim]*4, qf=[128]*4))
        trainer = TD3('MlpPolicy', env, verbose=0, device=device,
                batch_size=batch_size,
                learning_rate=hparam['learning_rate'],
                learning_starts=hparam['learning_starts'],
                train_freq=(hparam['update_itr'], "episode"),
                policy_kwargs=policy_kwargs,
                tensorboard_log=f"runs/{run.id}" if hparam['tb_log'] else None
        )
        total_timesteps = max_episodes*max_steps
    elif hparam['model']=='RecurrentPPO':
        policy_kwargs = dict(activation_fn=hparam['activation'],
                     net_arch=[dict(pi=[hidden_dim]*4, vf=[128]*4)])
        trainer = RecurrentPPO('MlpPolicy', env, verbose=0, device=device,
                n_steps=hparam['n_steps'],
                batch_size=batch_size,
                learning_rate=hparam['learning_rate'],
                policy_kwargs=policy_kwargs,
                tensorboard_log=f"runs/{run.id}" if hparam['tb_log'] else None
        )
        total_timesteps = max_episodes*max_steps
    else:
        raise "Please use proper model"

    trainer.learn(total_timesteps=total_timesteps,
        callback=WandbCallback(
            model_save_path=f"models/{run.id}",
            model_save_freq=2000*max_steps,
            gradient_save_freq=100*max_steps,
            verbose=0,
        ) if hparam['tb_log'] else None,
    )
\

if __name__=='__main__':
    # train(1000, 'CartPole-v1')
    parser = argparse.ArgumentParser()

    # Common arguments
    parser.add_argument('--gpu', default='0', type=int, help="gpu number")
    parser.add_argument('--model', choices=['TD3','PPO','RecurrentPPO','SAC']
                                , required=True, help='SB3 Models')

    # Arguments for training 
    parser.add_argument('--rew_norm', action='store_true', help="Reward normalization")
    parser.add_argument('--obs_norm', action='store_true', help="Observation normalization")
    parser.add_argument('--tb_log', action='store_true', help="Tensorboard logging")

    # Arguments for test
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--path', type=str, default=None, help='required only at test phase')
    parser.add_argument('--record', action='store_true', help='whether record or not')
    parser.add_argument('--task', default='stabilize',choices=['stabilize', 'stabilize-record', 'takeoff'],
                        help='For takeoff-aviary-v0 environment')


    args = parser.parse_args()

    hparam = dict([(k,v[0](*v[1])) for k,v in hparam_set.items()])
    hparam.update(vars(args))
    train(args, hparam)