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

from evaluation import evaluation, generate_result, drone_test
import random
import pickle as pkl
import pandas as pd
import itertools

from time import time

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
    'battery_range': 0.3 # (1-n) ~ (1)
}
hparam_set = {
    "goal_dim": [18],
    "param_num": [14],
    "hidden_dim": [48],


    "q_lr": [1e-3],
    "policy_lr": [3e-4],
    "policy_target_update_interval": [2],
    "max_steps": [200],
    "her_length": [200]
}

def train(args, hparam):

    #####################################
    # hyper-parameters for RL training ##
    #####################################

    max_episodes  = int(3e5)
    hidden_dim = hparam['hidden_dim']
    max_steps = hparam['max_steps']
    eval_max_steps = 300
    goal_dim = hparam['goal_dim']
    param_num = hparam['param_num']
    her_history_length = hparam['her_length']
    her_gamma = hparam['her_gamma'] # if 1 -> dense reward , 0 -> sparse reward
    policy_target_update_interval = hparam['policy_target_update_interval'] # delayed update for the policy network and target networks
    epsilon_pos = 0.05/6 # 5 cm error
    epsilon_ang = np.deg2rad(10) # 10 deg error
    hparam.update({"epsilon_pos":epsilon_pos,
                   "epsilon_ang":epsilon_ang})

    batch_size  = 128 if args.rnn != "None" else 128 * her_history_length
    nenvs = 1
    explore_noise_scale_init = 0.25
    eval_noise_scale_init = 0.25
    explore_noise_scale = explore_noise_scale_init
    eval_noise_scale = eval_noise_scale_init
    best_score = -np.inf
    frame_idx   = 0
    replay_buffer_size = 2e5 if args.rnn != "None" else 2e5 * max_steps
    explore_episode = 1000
    update_itr = 2
    writer_interval = 200
    eval_freq = 2000
    eval_itr = 25

    DETERMINISTIC=True  # DDPG: deterministic policy gradient
    
    #########################################
    ### Path, Basic variable setting ########
    #########################################

    print("hyperparam set:",hparam)
    algorithm_name = 'TD3'
    env_name = "takeoff-aviary-v0"
    dtime = datetime.now()
    tag = "randomize"
    rnn_tag = args.rnn if args.rnn != "None" else 'FF'
    savepath = "save/%s/%s/%s/%s/%s"%(algorithm_name,tag,rnn_tag,env_name, dtime.strftime("%y%b%d%H%M%S"))

    if not os.path.isdir(savepath):
        os.makedirs(savepath) 
    
    # tensorboard
    writer = None
    now = dtime.strftime("%y%b%d%H%M%S")
    if args.tb_log:
        if not os.path.isdir('tb_log'):
            os.mkdir('tb_log')
        tbpath = "%s/%s/%s"%(algorithm_name, tag, rnn_tag)
        tbpath = os.path.join('tb_log', env_name, tbpath, now)
        print("[Tensorboard log]:", tbpath)
        writer = SummaryWriter(tbpath)

        # wandb
        wandb.init(project="TD3-drone-final", config=hparam)
        wandb.run.name = "%s_%s"%(rnn_tag, now)
        wandb.run.save()
    
    device=torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print("Device:",device)

    ####################################
    # Define environment and agent #####
    ####################################

    envs = dynRandeEnv(env_name=env_name, obs_norm=False, tag="%s%s%s"%(algorithm_name, tag, rnn_tag), task='stabilize', nenvs=nenvs, dyn_range=dyn_range, episode_len=max_steps/200, seed=args.seed)
    eval_env = dynRandeEnv(env_name=env_name, obs_norm=False, tag="%s%s%seval"%(algorithm_name, tag, rnn_tag), task='stabilize', nenvs=1, dyn_range=dyn_range, episode_len=max_steps/200, seed=args.seed+1234567)
    td3_trainer = td3_agent(env=envs,
                rnn=args.rnn,
                device=device,
                hparam=hparam,
                replay_buffer_size=replay_buffer_size)

    # keep track of progress
    mean_rewards = []
    scores_window = deque(maxlen=100)
    loss_storage = {"policy_loss":deque(maxlen=writer_interval),
                    "q_loss_1":deque(maxlen=writer_interval),
                    "q_loss_2":deque(maxlen=writer_interval),
                    'param_loss':deque(maxlen=writer_interval)}


    for i_episode in range(1,max_episodes+1):
        
        state, param = envs.reset()
        last_action = np.stack([envs.env.action_space.sample() for _ in range(nenvs)],axis=0).squeeze()
        last_action = -np.ones_like(last_action)[None,:]
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
        thetas = [np.random.uniform(-np.pi,np.pi) for _ in range(nenvs)]
        goal = np.array([[0,0,0, # position
                        np.cos(theta),np.sin(theta),0, # rotation matrix
                        -np.sin(theta),np.cos(theta),0,
                        0,0,1,
                        0,0,0, # velocity
                        0,0,0,# angular velocity
                        0,0,0,0] for theta in thetas]) # dummy action goal

        for step in range(max_steps):
            hidden_in = hidden_out
            if args.rnn == "None":
                action = \
                    td3_trainer.get_action(state, 
                                            last_action,
                                            deterministic=DETERMINISTIC, 
                                            explore_noise_scale=explore_noise_scale)
                hidden_in = hidden_out = None
            elif args.rnn in ["RNNHER", "LSTMHER","GRUHER"]:
                action, hidden_out = \
                        td3_trainer.get_action(state, 
                                    last_action, 
                                    hidden_in,
                                    goal=envs.normalize_obs(goal),
                                    deterministic=DETERMINISTIC, 
                                    explore_noise_scale=explore_noise_scale)
            else:
                action, hidden_out = \
                    td3_trainer.get_action(state, 
                                last_action, 
                                hidden_in, 
                                deterministic=DETERMINISTIC, 
                                explore_noise_scale=explore_noise_scale)
            next_state, reward, done, _ = envs.step(action) 

            episode_state.append(envs.unnormalize_obs(state))
            episode_action.append(action)
            episode_last_action.append(last_action)
            episode_reward.append(reward)
            episode_next_state.append(envs.unnormalize_obs(next_state))
            episode_done.append(done)

            state = next_state
            last_action = action
            frame_idx += 1
        
        # Push into Experience replay buffer
        if args.rnn in ["RNNHER","LSTMHER","GRUHER"]:
            td3_trainer.replay_buffer.push_batch(episode_state, 
                            episode_action, 
                            episode_last_action,
                            episode_reward, 
                            episode_next_state, 
                            episode_done,
                            param,
                            goal)
        else:           
            td3_trainer.replay_buffer.push_batch(episode_state, 
                            episode_action, 
                            episode_last_action,
                            episode_reward, 
                            episode_next_state, 
                            episode_done,
                            param)

        # Update TD3 trainer
        if i_episode > explore_episode:
            for i in range(update_itr):
                loss_dict = td3_trainer.update(batch_size, norm_ftn=envs.normalize_obs, deterministic=DETERMINISTIC, eval_noise_scale=eval_noise_scale)
                policy_loss.append(loss_dict['policy_loss'])
                q_loss_1.append(loss_dict['q_loss_1'])
                q_loss_2.append(loss_dict['q_loss_2'])

        # Noise decay
        explore_noise_scale = (0.9 * (1-i_episode/max_episodes) + 0.1) * explore_noise_scale_init
        eval_noise_scale = (0.9 * (1-i_episode/max_episodes) + 0.1) * eval_noise_scale_init

        loss_storage['policy_loss'].append(np.mean(policy_loss))
        loss_storage['q_loss_1'].append(np.mean(q_loss_1))
        loss_storage['q_loss_2'].append(np.mean(q_loss_2))
        rewards = np.sum(episode_reward)/nenvs
        mean_rewards.append(rewards)
        scores_window.append(rewards)

        ##########################################
        ## Tensorboard & WandB writing ###########
        ##########################################

        if writer is not None and i_episode%writer_interval == 0:
            writer.add_scalar('loss/loss_p', np.mean(loss_storage['policy_loss']), i_episode)
            writer.add_scalar('loss/loss_q_1', np.mean(loss_storage['q_loss_1']), i_episode)
            writer.add_scalar('loss/loss_q_2', np.mean(loss_storage['q_loss_2']), i_episode)
            writer.add_scalar('rewards', scores_window[-1], i_episode)
            wandb.log({'loss/loss_p':np.mean(loss_storage['policy_loss']),
                        'loss/loss_q_1': np.mean(loss_storage['q_loss_1']),
                        'loss/loss_q_2': np.mean(loss_storage['q_loss_2']),
                        'rewards': scores_window[-1]}, step=i_episode)
            
            if args.lr_scheduler:
                writer.add_scalar('lr/q_lr',td3_trainer.scheduler_q1.get_last_lr()[0], i_episode)
                writer.add_scalar('lr/policy_lr',td3_trainer.scheduler_policy.get_last_lr()[0], i_episode)
                wandb.log({'lr/q_lr':td3_trainer.scheduler_q1.get_last_lr()[0],
                        'lr/policy_lr':td3_trainer.scheduler_policy.get_last_lr()[0]},
                         step=i_episode)

            # Model parameters 
            for name, weight in td3_trainer.policy_net.named_parameters():
                writer.add_histogram(f'policy_net/{name}', weight, i_episode)
                if weight.grad is not None:
                    writer.add_histogram(f'policy_net/{name}.grad', weight.grad, i_episode)
            for name, weight in td3_trainer.q_net1.named_parameters():
                writer.add_histogram(f'q_net/{name}', weight, i_episode)
                if weight.grad is not None:
                    writer.add_histogram(f'q_net/{name}.grad', weight.grad, i_episode)

            if 'aviary' in env_name:
                unnormed_state = np.stack(episode_state)
                writer.add_scalar('loss/position[m]', np.linalg.norm((6*np.stack(unnormed_state)[:,:,:3]), axis=-1).mean(), i_episode)
                writer.add_scalar('loss/velocity[m_s]', np.linalg.norm((3*np.stack(unnormed_state)[:,:,12:15]), axis=-1).mean(), i_episode)
                writer.add_scalar('loss/ang_velocity[deg_s]', np.linalg.norm((2*180*np.stack(unnormed_state)[:,:,15:18]), axis=-1).mean(), i_episode)
                writer.add_scalar('loss/angle[deg]', 180/np.pi*np.arccos(np.clip(np.stack(unnormed_state)[:,:,11].flatten(),-1.0,1.0)).mean(), i_episode)
                wandb.log({'loss/position[m]': np.linalg.norm((6*np.stack(unnormed_state)[:,:,:3]), axis=-1).mean(),
                        'loss/velocity[m_s]': np.linalg.norm((3*np.stack(unnormed_state)[:,:,12:15]), axis=-1).mean(),
                        'loss/ang_velocity[deg_s]': np.linalg.norm((2*180*np.stack(unnormed_state)[:,:,15:18]), axis=-1).mean(),
                        'loss/angle[deg]': 180/np.pi*np.arccos(np.clip(np.stack(unnormed_state)[:,:,11].flatten(),-1.0,1.0)).mean(),
                        'loss/rpm': (1+np.stack(unnormed_state)[:,:,-4:]).mean()},
                         step=i_episode)
                


        ######################################
        ### Evaluation #######################
        ######################################

        if i_episode % eval_freq == 0 and i_episode != 0:
            eval_env.env.training = False
            td3_trainer.policy_net.eval()
            eval_rew, eval_success, eval_position, eval_angle = drone_test(eval_env, agent=td3_trainer, max_steps=eval_max_steps, test_itr=eval_itr, record=False)
            td3_trainer.policy_net.train()
            if writer is not None:
                writer.add_scalar('eval/reward', eval_rew, i_episode)
                writer.add_scalar('eval/success_rate', eval_success, i_episode)
                writer.add_scalar('eval/position', eval_position, i_episode)
                writer.add_scalar('eval/angle', np.rad2deg(eval_angle), i_episode)
                wandb.log({'eval/reward':eval_rew,
                            'eval/success_rate':eval_success,
                            'eval/position': eval_position,
                            'eval/angle': np.rad2deg(eval_angle)},
                            step=i_episode)

        ########################################
        ### Model save #########################
        ########################################

        if i_episode % 2000 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            td3_trainer.save_model(os.path.join(savepath,"iter%07d"%i_episode))
            envs.save(os.path.join(savepath,"iter%07d"%i_episode))

        if np.mean(scores_window)>=best_score: 
            td3_trainer.save_model(os.path.join(savepath,"best"))
            envs.save(os.path.join(savepath,"best"))
            best_score = np.mean(scores_window)

        if i_episode % 10000 == 0:
            artifact = wandb.Artifact('agent-%s'%now, type='model')
            artifact.add_dir(savepath+"/")
            wandb.log_artifact(artifact)
        
    td3_trainer.save_model(os.path.join(savepath,"final"))
    print('\rFinal\tAverage Score: {:.2f}'.format(np.mean(scores_window)))
    artifact = wandb.Artifact('agent-%s'%now, type='model')
    artifact.add_dir(savepath+"/")
    wandb.log_artifact(artifact)

    envs.close()
    del envs
    
    return mean_rewards, loss_storage, eval_rew, eval_success, dtime

def test(args, hparam):
    if args.record:
        disp = Display(visible=False, size=(100, 60))
        disp.start()
    env_name = "takeoff-aviary-v0"
    max_steps = 300
    
    device=torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    # Define environment
    eval_env = dynRandeEnv(env_name=env_name, 
                    #    load_path=args.path,
                       tag="test", 
                       task=args.task,
                       episode_len=max_steps/200,
                       nenvs=1, 
                    #    dyn_range=dyn_range, 
                       dyn_range=dict(),
                       seed=int(args.seed+123456789),
                       record=args.record)
    eval_env.env.training = False
    td3_trainer = td3_agent(env=eval_env,
                rnn=args.rnn,
                device=device,
                hparam=hparam)
    td3_trainer.load_model(args.path)
    td3_trainer.policy_net.eval()
    
    action, hidden1 = td3_trainer.policy_net.get_action(np.ones((1,22)),np.zeros((1,4)),torch.zeros((1,1,32)),np.array([[0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0]]),True,0.0)
    action2, hidden2 = td3_trainer.policy_net.get_action(np.concatenate([np.ones((1,18)),action[None,:]], axis=-1),action[None,:],hidden1,np.array([[0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0]]),True,0.0)

    eval_rew, eval_success, eval_position, eval_angle = drone_test(eval_env, agent=td3_trainer, max_steps=max_steps, test_itr=10, record=args.record, log=True)
    print("EVALUATION REWARD:",eval_rew)
    print("EVALUATION SUCCESS RATE:",eval_success)
    print("EVALUATION POSITION ERROR[m]:",eval_position)
    print("EVALUATION ANGLE ERROR[deg]:",np.rad2deg(eval_angle))

    if args.record:
        disp.stop()

if __name__=='__main__':
    # train(1000, 'CartPole-v1')
    parser = argparse.ArgumentParser()

    # Common arguments
    parser.add_argument('--gpu', default='0', type=int, help="gpu number")
    parser.add_argument('--rnn', choices=['None','RNN2','GRU2','LSTM2',
                                            'RNNHER','GRUHER','LSTMHER',
                                            'RNNsHER','GRUsHER','LSTMsHER']
                                , default='None', help='Use memory network (LSTM)')
    parser.add_argument('--policy_actf', type=str, default='tanh', help="policy activation function")
    parser.add_argument('--obs_norm', action='store_true', help='use batchnorm for input normalization')

    # Arguments for training 
    parser.add_argument('--tb_log', action='store_true', help="Tensorboard logging")
    parser.add_argument('--hparam', action='store_true', help="find hparam set")
    parser.add_argument('--lr_scheduler', action='store_true', help="Use lr scheduler")
    parser.add_argument('--reward_norm', action='store_true', help="reward normalization")
    parser.add_argument('--her_gamma', default=0.0, type=float, help="if 0 only her, if 1 no her")
    parser.add_argument('--positive_rew', action='store_true', help="use [0,1] reward instead of [-1,0]")
    parser.add_argument('--small_lr', action='store_true', help='use small lr')
    parser.add_argument('--behavior_path', default=None, help='path for behavior networks')
    parser.add_argument('--seed', type=int, default=0, help='seed')

    parser.add_argument('--single_pos', action='store_true', help="Single pose for HER")

    parser.add_argument('--reward_scale', default=1.0, type=float, help="reward scale for sgHER")
    parser.add_argument('--maintain_length', default=1, type=int, help="maintain_length for sgHER")

    # Arguments for test
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--path', type=str, default=None, help='required only at test phase')
    parser.add_argument('--record', action='store_true', help='whether record or not')
    parser.add_argument('--task', default='stabilize',choices=['stabilize', 'stabilize-record', 'takeoff'],
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
            if args.small_lr:
                hparam = dict([(k,v[-1]) for k,v in hparam_set.items()])
            else:
                hparam = dict([(k,v[0]) for k,v in hparam_set.items()])
            hparam.update(vars(args))
            hparam['policy_actf'] = getattr(F,args.policy_actf)
            train(args, hparam)
    else:
        if args.path is None:
            assert "model path is required"
        hparam = dict([(k,v[0]) for k,v in hparam_set.items()])
        hparam.update(vars(args))
        hparam['policy_actf'] = getattr(F,args.policy_actf)
        test(args, hparam)