import torch
import gym
import matplotlib
from envs.customEnv import domainRandomize
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
from envs.customEnvDrone import domainRandomAviary

from td3.td3 import TD3_Trainer, TD3LSTM_Trainer
from td3.common.buffers import ReplayBuffer, ReplayBufferLSTM2, ReplayBufferPER

import argparse
import os
from utils import save_frames_as_gif
from pyvirtualdisplay import Display
import numpy as np
from copy import deepcopy
import pandas as pd
from tqdm import tqdm


def evaluation(args):
    np.random.seed(args.seed)

    for env_name in ['Pendulum-v0']:
    
        gym_ratio = args.rand_ratio
        dyn_range = {
            # cartpole
            'masscart': gym_ratio, # 1/n ~ n
            'masspole': gym_ratio, # 1/n ~ n
            'length': gym_ratio, # 1/n ~ n
            'force_mag': gym_ratio, # 1/n ~ n

            # pendulum
            'max_torque': gym_ratio, # 1/n ~ n
            'm': gym_ratio, # 1/n ~ n
            'l': gym_ratio, # 1/n ~ n

            # drones
            'mass_range': 0.3, # (1-n) ~ (1+n)
            'cm_range': 0.3, # (1-n) ~ (1+n)
            'kf_range': 0.1, # (1-n) ~ (1+n)
            'km_range': 0.1, # (1-n) ~ (1+n)
            'battery_range': 0.3 # (1-n) ~ (1)
        }

        # hyper-parameters for RL training
        max_steps   = 150  # Pendulum needs 150 steps per episode to learn well, cannot handle 20
        replay_buffer_size = 1e4
        policy_target_update_interval = 10 # delayed update for the policy network and target networks
        DETERMINISTIC=True  # DDPG: deterministic policy gradient      
        
        if 'Pendulum' in env_name:
            hidden_dim = 128
            max_steps = 1500
            envs = gym.make(env_name)
            envs.seed(args.seed)
            setattr(envs,'env_name', env_name)
            
            domainRandomize(envs, dyn_range=dyn_range, seed=args.seed)
        elif 'aviary' in env_name:
            hidden_dim = 128
            max_steps = 4000
            envs = gym.make(id=env_name, # arbitrary environment that has state normalization and clipping
                drone_model=DroneModel.CF2X,
                initial_xyzs=np.array([[0.0,0.0,2.0]]),
                initial_rpys=np.array([[0.0,0.0,0.0]]),
                physics=Physics.PYB_GND_DRAG_DW,
                freq=240,
                aggregate_phy_steps=1,
                gui=False,
                record=False, 
                obs=ObservationType.KIN,
                act=ActionType.RPM)
            envs = domainRandomAviary(env, 'test', 0, args.seed,
                observable=['pos', 'rotation', 'vel', 'angular_vel', 'rpm'],
                frame_stack=1,
                task='stabilize2',
                reward_coeff={'xyz':0.2, 'vel':0.016, 'ang_vel':0.08, 'd_action':0.002},
                episode_len_sec=2,
                max_rpm=66535,
                initial_xyz=[[0.0,0.0,50.0]], # Far from the ground
                freq=200,
                rpy_noise=1.2,
                vel_noise=1.0,
                angvel_noise=2.4,
                mass_range=dyn_range.get('mass_range', 0.0),
                cm_range=dyn_range.get('cm_range', 0.0),
                kf_range=dyn_range.get('kf_range', 0.0),
                km_range=dyn_range.get('km_range', 0.0),
                battery_range=dyn_range.get('battery_range', 0.0))
            setattr(envs, 'env_name', env_name)
        else:
            raise NotImplementedError

        device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
        action_space = envs.action_space
        state_space = envs.observation_space

        if 'Pendulum' in env_name:
            replay_buffer = ReplayBuffer(replay_buffer_size)
            single_agent,\
            random_agent,\
            multitask_agent = [TD3_Trainer(
                                    replay_buffer,
                                    state_space, 
                                    action_space, 
                                    hidden_dim=hidden_dim,
                                    device=device, 
                                    policy_target_update_interval=policy_target_update_interval) for _ in range(3)]
            replay_buffer = ReplayBufferLSTM2(replay_buffer_size)
            lstm_single_agent,\
            lstm_random_agent,\
            lstm_multitask_agent,\
            lstm_fastadapt_agent = [TD3LSTM_Trainer(
                                    replay_buffer,
                                    state_space, 
                                    action_space, 
                                    hidden_dim=hidden_dim,
                                    device=device, 
                                    policy_target_update_interval=policy_target_update_interval) for _ in range(4)]
        elif 'HalfCheetah' in env_name:
            raise NotImplementedError
        elif 'aviary' in env_name:
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        tag = 'best' if args.best else 'final'
        single_agent.load_model('save/TD3/single/FF/%s/%s'%(env_name,tag))
        random_agent.load_model('save/TD3/randomize/FF/%s/%s'%(env_name,tag))
        multitask_agent.load_model('save/TD3/multitask/FF/%s/%s'%(env_name,tag))
        lstm_single_agent.load_model('save/TD3/single/LSTM/%s/%s'%(env_name,tag))
        lstm_random_agent.load_model('save/TD3/randomize/LSTM/%s/%s'%(env_name,tag))
        lstm_multitask_agent.load_model('save/TD3/multitask/LSTM/%s/%s'%(env_name,tag))
        lstm_fastadapt_agent.load_model('save/TD3/randomize/FastAdaptLSTM/%s/%s'%(env_name,tag))


        agent_FF_list = [single_agent, 
                        random_agent, 
                        multitask_agent]
        agent_LSTM_list = [lstm_single_agent, 
                            lstm_random_agent, 
                            lstm_multitask_agent, 
                            lstm_fastadapt_agent]
        algo_FF_list = ['td3_single',
                        'td3_random', 
                        'td3_multitask']
        algo_LSTM_list = ['td3lstm_single',
                        'td3lstm_random', 
                        'td3lstm_multitask',
                        'td3lstm_fastadapt']
        agent_list = agent_FF_list + agent_LSTM_list
        algo_list = algo_FF_list + algo_LSTM_list


        if "CartPole" in env_name:
            df = pd.DataFrame(columns=['algo','masscart','masspole','length','force_mag','reward_sum'])
        elif "Pendulum" in env_name:
            df = pd.DataFrame(columns=['algo','max_torque','m','l','reward_sum','steps'])
        elif "aviary" in env_name:
            df = pd.DataFrame(columns=['algo','mass','cm','kf', 'km','reward_sum'])
        else:
            raise NotImplementedError

        # TODO 완성하기
        df_h = pd.DataFrame(columns=['algo','iter','step','hidden','d_hidden','params'])
        with torch.no_grad():
            for i_eval in range(args.num_eval):

                base_env = gym.make(env_name)
                base_env.seed(args.seed+i_eval)
                setattr(base_env,'env_name', env_name)
                domainRandomize(base_env, dyn_range=dyn_range, seed=args.seed+i_eval)
                start_state = base_env.reset()

                env_list = [deepcopy(base_env) for _ in range(len(algo_list))]
                for env,agent,algo in zip(env_list, agent_list, algo_list):
                    
                    total_rew = 0
                    state = deepcopy(start_state)[None,:]
                    last_action = envs.action_space.sample()[None,:]
                    hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device), \
                                torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device))

                    with torch.no_grad():
                        total_step, step, success = 0,0,0
                        for step in range(max_steps):
                            if 'lstm' in algo:
                                hidden_in = hidden_out
                                action, hidden_out = \
                                    agent.policy_net.get_action(state, 
                                                                    last_action, 
                                                                    hidden_in, 
                                                                    deterministic=DETERMINISTIC, 
                                                                    explore_noise_scale=0.)
                                param = {}
                                if "CartPole" in env_name:
                                    param = {'masscart': env.env.masscart,
                                            'masspole': env.env.masspole,
                                            'length': env.env.length,
                                            'force_mag': env.env.force_mag}
                                elif "Pendulum" in env_name:
                                    param = {'max_torque':env.max_torque,
                                            'm':env.env.m,
                                            'l':env.env.l}
                                elif "aviary" in env_name:
                                    param = {'mass':env.mass,
                                            'com':env.com,
                                            'kf':env.kf,
                                            'km':env.km,
                                            'battery':env.battery}
                                h_in = hidden_in[0].detach().cpu().numpy()
                                h_out = hidden_out[0].detach().cpu().numpy()
                                df_h = df_h.append({
                                    'algo':algo,
                                    'iter':i_eval,
                                    'step':step,
                                    'hidden':h_out,
                                    'd_hidden': h_out-h_in,
                                    'params':param
                                }, ignore_index=True)
                            else:
                                action = agent.policy_net.get_action(state, 
                                                                    deterministic=DETERMINISTIC, 
                                                                    explore_noise_scale=0.)
                            next_state, reward, done, _ = env.step(action) 
                            if not isinstance(action, np.ndarray):
                                action = np.array([action])
                            state, last_action = next_state[None,:], action[None,:]
                            total_step += 1

                            if "Pendulum" in env_name:
                                step = step+1 if state[0,0] > 0.98 else 0
                                if step > 100:
                                    success = 1
                                    break
                            elif "aviary" in env_name:
                                step = step+1 if np.linalg.norm(state[:3]) < 0.1 else 0
                                if step > 100:
                                    success = 1

                            total_rew += reward
                    
                    if "CartPole" in env_name:
                        df = df.append({'algo': algo,
                                    'masscart': env.env.masscart,
                                    'masspole': env.env.masspole,
                                    'length': env.env.length,
                                    'force_mag': env.env.force_mag,
                                    'reward_sum': total_rew,
                                    'success':success}, ignore_index=True)
                    elif "Pendulum" in env_name:
                        df = df.append({'algo':algo,
                                    'max_torque':env.max_torque,
                                    'm':env.env.m,
                                    'l':env.env.l,
                                    'reward_sum':total_rew,
                                    'steps':total_step,
                                    'success':success}, ignore_index=True)
                    elif "aviary" in env_name:
                        df = df.append({'algo':algo,
                                    'mass':env.mass,
                                    'com':env.com,
                                    'kf':env.kf,
                                    'km':env.km,
                                    'battery':env.battery,
                                    'steps':total_step,
                                    'success':success}, ignore_index=True)
                    else:
                        raise NotImplementedError
                    
                    env.close()
        


        if not os.path.isdir("evaluation"):
            os.mkdir('evaluation')
        
        
        df.to_pickle(os.path.join('evaluation',"%s_S%d_N%d_R%d.pkl"%(env_name, args.seed, args.num_eval, int(args.rand_ratio*100))))
        # df.to_csv(os.path.join('evaluation',"%s_S%d_N%d_R%d.csv"%(env_name, args.seed, args.num_eval, int(args.rand_ratio*100))))

        df_h.to_pickle(os.path.join('evaluation',"%s_S%d_N%d_R%d_hiddens.pkl"%(env_name, args.seed, args.num_eval, int(args.rand_ratio*100))))
        # df_h.to_csv(os.path.join('evaluation',"%s_S%d_N%d_R%d_hiddens.csv"%(env_name, args.seed, args.num_eval, int(args.rand_ratio*100))))

        print("\n%s evaluation result"%(env_name))
        print('------ reward sum --------')
        for algo in algo_list:
            print(algo,":",np.mean(df.loc[df['algo']==algo]['reward_sum']))
        print('------ success rate --------')
        for algo in algo_list:
            print(algo,":",np.mean(df.loc[df['algo']==algo]['success']))
        if "Pendulum" in env_name or "aviary" in env_name:
            print('------ Average steps for success --------')
            for algo in algo_list:
                print(algo,":",np.mean(df.loc[df['algo']==algo].loc[df['success']==1]['steps']))

def use_data():
    algo_FF_list = ['td3_single','td3_random', 'td3_multitask']
    algo_LSTM_list = ['td3lstm_single','td3lstm_random', 'td3lstm_multitask']
    algo_list = algo_FF_list + algo_LSTM_list

    files = [file for file in os.listdir('evaluation') if '.pkl' in file]
    for file in files:
        env_name, seed, num_eval, rand_ratio = file.strip('.pkl').split("_")
        df = pd.read_pickle(os.path.join('evaluation', file))
        print("\n%s evaluation result (seed: %s, num_eval: %s, random ratio: %s)"%(env_name, seed, num_eval, rand_ratio))
        print('------ reward sum --------')
        for algo in algo_list:
            print(algo,":",np.mean(df.loc[df['algo']==algo]['reward_sum']))
        print('------ success rate --------')
        for algo in algo_list:
            print(algo,":",np.mean(df.loc[df['algo']==algo]['success']))
        if "Pendulum" in env_name or "aviary" in env_name:
            print('------ Average steps for success --------')
            for algo in algo_list:
                print(algo,":",np.mean(df.loc[df['algo']==algo].loc[df['success']==1]['steps']))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_eval', default=100, type=int)
    parser.add_argument('--best', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--rand-ratio', type=float, default=3, help="Only for Pendulum")
    parser.add_argument('--use-data', action='store_true')
    args = parser.parse_args()
    with Display(visible=False, size=(100, 60)) as disp:
        if not args.use_data:
            evaluation(args)
        else:
            use_data()