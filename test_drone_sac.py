import numpy as np
from collections import deque
import torch

from sac.sac import *
from sac.common.buffers import *
from sac.agent import sac_agent
from envs.customEnv import dynRandeEnv

import argparse
from pyvirtualdisplay import Display
import os
import wandb
from datetime import datetime

from evaluation import drone_test

############################
####### Global Vars ########
############################

import warnings
warnings.simplefilter("ignore", UserWarning)

dyn_range = {
    # drones
    'mass_range': 0.3, # (1-n) ~ (1+n)
    'cm_range': 0.3, # (1-n) ~ (1+n)
    'kf_range': 0.3, # (1-n) ~ (1+n)
    'km_range': 0.3, # (1-n) ~ (1+n)
    'i_range': 0.3,
    't_range': 0.3,
    'norm_range': 0.3, # Just for normalization
}

hparam_set = {
    "learning_rate": (np.random.uniform,[1e-4,1e-4]),
    "learning_starts": (np.random.randint,[80000,80001]),
    "activation": (np.random.choice, [[F.relu]]),

    # SAC, TD3
    "update_itr": (np.random.randint,[10,11]),

    "goal_dim": (np.random.randint,[18,19]),
    "param_dim": (np.random.randint,[15,16]),
    "hidden_dim": (np.random.randint,[6,7]),
    "critic_dim": (np.random.randint,[7,8]),
    "policy_net_layers": (np.random.randint,[3,4]),
    "critic_net_layers": (np.random.randint,[4,5]),

    "max_steps": (np.random.randint,[800,801]),
    "seq_length": (np.random.randint,[100,101]),
    "rnn_dropout": (np.random.uniform,[0,0]),
    "replay_buffer_size": (np.random.randint,[int(1e6), int(1e6+1)]),
    "gradient_steps": (np.random.randint,[1,2]),
}

def main(args, hparam):

    global dyn_range
    
    # Set randomness seed 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #####################################
    # hyper-parameters for RL training ##
    #####################################

    # hparam['learning_rate'] = 10**hparam['learning_rate']
    hparam['hidden_dim'] = int(2**hparam['hidden_dim'])
    hparam['critic_dim'] = int(2**hparam['critic_dim'])
    policy_dim = hparam['hidden_dim']
    replay_buffer_size = hparam['replay_buffer_size']
    observable = ['rel_pos', 'rotation', 'rel_vel', 'rel_angular_vel']
    if hparam['rnn']!='None':
        observable += ['param']
    rew_coeff = {'pos':1.0, 'vel':0.0, 'ang_vel': 0.0, 'ang_vel_xy': 0.0, 'ang_vel_z': 0.0, 'd_action':0.0, 'rotation': 0.0}
    hparam['observable'] = observable
    hparam['rew_coeff'] = rew_coeff
    hparam['dyn_range'] = dyn_range

    #########################################
    ### Path, Basic variable setting ########
    #########################################
    
    if args.gpu >= 0:
        device=torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
    else:
        device=torch.device('cpu')
    
    

    ####################################
    # Define environment and agent #####
    ####################################

    dummy_env = dynRandeEnv(
        initial_xyzs=np.array([[0,0,10000.0]]),
        initial_rpys=np.array([[0,0,0]]),
        observable=observable,
        dyn_range=dyn_range,
        rpy_noise=2*np.pi,
        vel_noise=1,
        angvel_noise=np.pi,
        reward_coeff=rew_coeff,
        frame_stack=1,
        episode_len_sec=800/100,
        gui=False,
        record=False,
        wandb_render=True,
        is_noise=True,
    )
    trainer = sac_agent(env=dummy_env,
                rnn=args.rnn,
                device=device,
                hparam=hparam,
                replay_buffer_size=replay_buffer_size)

    
    if args.task =='stabilize':
        # Test code
        max_steps = 800
        dyn_range = {
            'norm_range': 0.3, # Just for normalization
        }
        set_dyn = {
            'mass': args.mass,
            'cm': np.array(args.cm),
            'I': np.array(args.I),
            'T': args.T,
            "KF": np.array(args.KF),
            "KM": np.array(args.KM)
        }
        eval_env = dynRandeEnv(
            initial_xyzs=np.array([[0,0,10000.0]]),
            initial_rpys=np.array([[np.pi,0,0]]),
            observable=observable,
            dyn_range=dyn_range, # No dynamic randomization for evaluation env
            # rpy_noise=2*np.pi,
            # vel_noise=1,
            # angvel_noise=np.pi,
            rpy_noise=0,
            vel_noise=0,
            angvel_noise=0,
            reward_coeff=rew_coeff,
            frame_stack=1,
            episode_len_sec=max_steps/100,
            gui=False,
            record=False,
            wandb_render=True,
            is_noise=True,
            goal=np.array([[0,0,10001.0]])
        )
        trainer.load_model(args.path)
        total_info = []

        for itr in range(args.test):
            
            obs = eval_env.reset(set_dyn=set_dyn)
            if args.rnn == 'PID':
                trainer.reset(env=eval_env)
            state = eval_env._getDroneStateVector(0).squeeze()
            reward_sum = 0
            state_buffer,obs_buffer, action_buffer = [],[],[]
            goal_state = np.zeros_like(state)
            goal_state[:3] = eval_env.goal_pos[0]
            last_action = -np.ones(eval_env.action_space.shape)[None,:]
            success = 0

            if "LSTM" in args.rnn:
                hidden_out = (torch.zeros([1, 1, policy_dim], dtype=torch.float).to(device), \
                            torch.zeros([1, 1, policy_dim], dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
            else:
                hidden_out = torch.zeros([1, 1, policy_dim], dtype=torch.float).to(device)
            
            for step in range(max_steps):

                # if step % 100 == 0:
                #     if "LSTM" in args.rnn:
                #         hidden_out = (torch.zeros([1, 1, policy_dim], dtype=torch.float).to(device), \
                #                     torch.zeros([1, 1, policy_dim], dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
                #     else:
                #         hidden_out = torch.zeros([1, 1, policy_dim], dtype=torch.float).to(device)

                hidden_in = hidden_out
                # Get action
                if args.rnn == 'PID':
                    action = trainer.get_action(state)
                elif args.rnn in ["None"]:
                    action = \
                        trainer.get_action(obs, 
                                            last_action,
                                            param=False,
                                            deterministic=True)
                    hidden_in = hidden_out = None
                else:
                    action, hidden_out = \
                        trainer.get_action(obs, 
                                    last_action, 
                                    hidden_in,
                                    param=False,
                                    deterministic=True)

                obs_buffer.append(obs)
                state_buffer.append(state)
                action_buffer.append(action)

                obs, reward, done, info = eval_env.step(action)
                state = eval_env._getDroneStateVector(0).squeeze()

                if done:
                    obs = eval_env.reset()
                
                if np.linalg.norm(state[:3]-goal_state[:3]) < 0.5:
                    success = 1
                    break

                reward_sum+=reward
                last_action = action
            
            state_buffer.append(goal_state)
            rnn_name = args.rnn_name
            if rnn_name == "":
                rnn_name = args.rnn
            if not os.path.isdir('paperworks/%s_fixdyn'%(rnn_name)):
                os.mkdir('paperworks/%s_fixdyn'%(rnn_name))
            np.savetxt('paperworks/%s_fixdyn/test_state_%02d.txt'%(rnn_name,itr),np.stack(state_buffer),delimiter=',')
            np.savetxt('paperworks/%s_fixdyn/test_obs_%02d.txt'%(rnn_name,itr),np.stack(obs_buffer),delimiter=',')
            np.savetxt('paperworks/%s_fixdyn/test_action_%02d.txt'%(rnn_name,itr),np.concatenate(action_buffer),delimiter=',')
            # print("iteration : ",itr)
            info.update({'reward': reward_sum,
                        'position_error': np.linalg.norm(np.stack(obs_buffer)[:,:3],axis=1).mean(),
                        'yaw_rate': np.abs(np.stack(obs_buffer)[:,17]).mean(),
                        'success': success,
                        'success_timestep': step*success})

            # print(info)

            total_info.append(info)
        
        final_info = {}
        for info in total_info:
            for key,value in info.items():
                if key != 'episode':
                    if key == 'reward':
                        final_info[key] = final_info.get(key,0) + value
                    else:
                        final_info[key] = final_info.get(key,0) + np.abs(value)
        total_success = final_info['success']
        for key,value in final_info.items():
            if key == ' success_timestep':
                final_info[key] = value / total_success
            else:
                final_info[key] = value / args.test

        print("------%s Average results------"%args.rnn)
        print("pos error",final_info['position_error'])
        print("yaw rate", final_info['yaw_rate'])
        print("success rate", final_info['success'])
        print("average time", final_info['success_timestep'])


    elif args.task=='takeoff':
        # Test code
        goal_poses = [
            # 2 seconds each
            # [0.5,0,10000],
            [1,0,10000],
            # [1,0.5,10000],
            [1,1,10000],
            # [0.5,1,10000],
            [0,1,10000],
            # [0,0.5,10000],
            [0,0,10000]
        ]
        pose_idx=0
        max_steps = 1200 # 20 sec
        dyn_range = {
            # drones
            'norm_range': 0.3, # Just for normalization
        }
        set_dyn = {
            'mass': args.mass,
            'cm': np.array(args.cm),
            'I': np.array(args.I),
            'T': args.T,
            "KF": np.array(args.KF),
            "KM": np.array(args.KM)
        }
        eval_env = dynRandeEnv(
            initial_xyzs=np.array([[0,0,10000]]),
            initial_rpys=np.array([[0,0,0]]),
            observable=observable,
            dyn_range=dyn_range, # No dynamic randomization for evaluation env
            rpy_noise=0,
            vel_noise=0,
            angvel_noise=0,
            reward_coeff=rew_coeff,
            frame_stack=1,
            episode_len_sec=max_steps/100,
            gui=False,
            record=False,
            wandb_render=True,
            is_noise=True,
            goal=np.array(goal_poses[0])
        )
        trainer.load_model(args.path)
        total_info = []

        for itr in range(args.test):
            
            obs = eval_env.reset(set_dyn=set_dyn)
            if args.rnn == 'PID':
                trainer.reset(env=eval_env)
            state = eval_env._getDroneStateVector(0).squeeze()
            reward_sum = 0
            state_buffer,obs_buffer, action_buffer = [],[],[]
            goal_state = np.zeros_like(state)
            goal_state[:3] = eval_env.goal_pos[0]
            last_action = -np.ones(eval_env.action_space.shape)[None,:]
            success = 0

            if "LSTM" in args.rnn:
                hidden_out = (torch.zeros([1, 1, policy_dim], dtype=torch.float).to(device), \
                            torch.zeros([1, 1, policy_dim], dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
            else:
                hidden_out = torch.zeros([1, 1, policy_dim], dtype=torch.float).to(device)
            
            for step in range(max_steps):

                # if step%100==0:
                #     if "LSTM" in args.rnn:
                #         hidden_out = (torch.zeros([1, 1, policy_dim], dtype=torch.float).to(device), \
                #                     torch.zeros([1, 1, policy_dim], dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
                #     else:
                #         hidden_out = torch.zeros([1, 1, policy_dim], dtype=torch.float).to(device)

                hidden_in = hidden_out
                # Get action
                if args.rnn == 'PID':
                    action = trainer.get_action(state)
                elif args.rnn in ["None"]:
                    action = \
                        trainer.get_action(obs, 
                                            last_action,
                                            param=False,
                                            deterministic=True)
                    hidden_in = hidden_out = None
                else:
                    action, hidden_out = \
                        trainer.get_action(obs, 
                                    last_action, 
                                    hidden_in,
                                    param=False,
                                    deterministic=True)

                obs_buffer.append(obs)
                state_buffer.append(state)
                action_buffer.append(action)

                obs, reward, done, info = eval_env.step(action)
                state = eval_env._getDroneStateVector(0).squeeze()
                
                
                if done:
                    obs = eval_env.reset()

                if np.linalg.norm(state[:3]-goal_state[:3]) < 0.3:
                # if step % 100==99:
                    pose_idx += 1
                    if pose_idx >= len(goal_poses):
                        success = 1
                        break
                    eval_env.goal_pos[0] = np.array(goal_poses[pose_idx])
                    goal_state[:3] = eval_env.goal_pos[0]
                    if args.rnn == 'PID':
                        trainer.reset(env=eval_env)
                    elif "LSTM" in args.rnn:
                        hidden_out = (torch.zeros([1, 1, policy_dim], dtype=torch.float).to(device), \
                                    torch.zeros([1, 1, policy_dim], dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
                    else:
                        hidden_out = torch.zeros([1, 1, policy_dim], dtype=torch.float).to(device)

                reward_sum+=reward
                last_action = action
            

            rnn_name = args.rnn_name
            if rnn_name == "":
                rnn_name = args.rnn
            if not os.path.isdir('paperworks/%s_fixdyn'%(rnn_name)):
                os.mkdir('paperworks/%s_fixdyn'%(rnn_name))
            np.savetxt('paperworks/%s_fixdyn/takeoff_state_%02d.txt'%(rnn_name,itr),np.stack(state_buffer),delimiter=',')
            np.savetxt('paperworks/%s_fixdyn/takeoff_obs_%02d.txt'%(rnn_name,itr),np.stack(obs_buffer),delimiter=',')
            np.savetxt('paperworks/%s_fixdyn/takeoff_action_%02d.txt'%(rnn_name,itr),np.concatenate(action_buffer),delimiter=',')
            # print("iteration : ",itr)
            info.update({'reward': reward_sum,
                        'position_error': np.linalg.norm(np.stack(obs_buffer)[:,:3]*6,axis=1).mean(),
                        'yaw_rate': np.abs(np.stack(obs_buffer)[:,17]*np.pi*2).mean(),
                        'success': success,
                        'success_timestep': step*success})

            # print(info)

            total_info.append(info)
        
        final_info = {}
        for info in total_info:
            for key,value in info.items():
                if key != 'episode':
                    if key == 'reward':
                        final_info[key] = final_info.get(key,0) + value
                    else:
                        final_info[key] = final_info.get(key,0) + np.abs(value)
        total_success = final_info['success']
        for key,value in final_info.items():
            if key == ' success_timestep':
                final_info[key] = value / total_success
            else:
                final_info[key] = value / args.test

        print("------%s Average results------"%args.rnn)
        print("pos error",final_info['position_error'])
        print("yaw rate", final_info['yaw_rate'])
        print("success rate", final_info['success'])
        print("average time", final_info['success_timestep'])

if __name__=='__main__':
    # train(1000, 'CartPole-v1')
    parser = argparse.ArgumentParser()

    # Common arguments
    parser.add_argument('--gpu', default='0', type=int, help="gpu number")
    parser.add_argument('--rnn', choices=['None','PID',
                                                'RNNparam','GRUparam','LSTMparam',
                                                'RNN','GRU','LSTM',
                                                'RNNfull','GRUfull','LSTMfull',
                                                'RNNpolicy','GRUpolicy','LSTMpolicy',
                                                'RNNpolicyfull','GRUpolicyfull','LSTMpolicyfull'])

    # parser.add_argument('--dyn', choices=['mass', 'cm', 'kf', 'km', 'i', 't', 'no'], default=None)

    # Arguments for test
    parser.add_argument('--test', type=int, default=0, help='how many times for testing. 0 means training')
    parser.add_argument('--path', type=str, default=None, help='required only at test phase')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', default='stabilize',choices=['stabilize', 'takeoff'],
                        help='For takeoff-aviary-v0 environment')
    parser.add_argument('--rnn_name', type=str, default="")

    parser.add_argument('--mass', type=float, default=0.0)
    parser.add_argument('--cm',nargs=2, type=float, default=[0, 0])
    parser.add_argument('--I',nargs=3, type=float, default=[0, 0, 0])
    parser.add_argument('--T', type=float, default=0.0)
    parser.add_argument('--KF',nargs=4, type=float, default=[0, 0, 0, 0])
    parser.add_argument('--KM',nargs=4, type=float, default=[0, 0, 0, 0])

    args = parser.parse_args()

    hparam = dict([(k,v[0](*v[1])) for k,v in hparam_set.items()])
    hparam.update(vars(args))
    main(args, hparam)