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
from utils import wandb_artifact, save_frames_as_gif

import argparse
from pyvirtualdisplay import Display
import os
from torch.utils.tensorboard import SummaryWriter
import wandb
from datetime import datetime

from evaluation import evaluation, generate_result, drone_test, pendulum_test
import random
import pickle as pkl
import pandas as pd
import itertools

from time import time

############################
####### Global Vars ########
############################

dyn_range = {
    # pendulum
    'max_torque': 3, # 1/n ~ n
    'm': 3, # 1/n ~ n
    'l': 3, # 1/n ~ n
}
hparam_set = {
    "goal_dim": [2],
    "param_num": [3],
    "hidden_dim": [32],

    "q_lr": [1e-3, 3e-4, 1e-4],
    "policy_lr": [1e-3, 3e-4, 1e-4],
    "policy_target_update_interval": [2,3,4],
}

def train(args, hparam):

    #####################################
    # hyper-parameters for RL training ##
    #####################################

    max_episodes  = int(2e4)
    hidden_dim = hparam['hidden_dim']
    max_steps = 150
    eval_max_steps = 200
    goal_dim = hparam['goal_dim']
    param_num = hparam['param_num']
    her_history_length = args.her_length
    her_gamma = hparam['her_gamma'] # if 1 -> dense reward , 0 -> sparse reward
    policy_target_update_interval = hparam['policy_target_update_interval'] # delayed update for the policy network and target networks
    epsilon_pos = 0.00025 # dummy
    epsilon_ang = np.deg2rad(5)
    hparam.update({"epsilon_pos":epsilon_pos,
                   "epsilon_ang":epsilon_ang})

    batch_size  = 256 if args.rnn != "None" else 256 * max_steps
    nenvs = 16
    explore_noise_scale_init = 0.25
    eval_noise_scale_init = 0.25
    explore_noise_scale = explore_noise_scale_init
    eval_noise_scale = eval_noise_scale_init
    best_score = -np.inf
    frame_idx   = 0
    replay_buffer_size = 1e5
    explore_episode = 500
    update_itr = 1
    writer_interval = int(100/nenvs)*nenvs
    eval_freq = int(500/nenvs)*nenvs
    eval_itr = 50

    DETERMINISTIC=True  # DDPG: deterministic policy gradient
    
    #########################################
    ### Path, Basic variable setting ########
    #########################################

    print("hyperparam set:",hparam)
    algorithm_name = 'TD3'
    env_name = "Pendulum-v0"
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
        wandb.init(project="TD3-pendulum", config=hparam)
        wandb.run.name = "%s_%s"%(rnn_tag, now)
        wandb.run.save()
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%args.gpu
    device=torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print("Device:",device)

    ####################################
    # Define environment and agent #####
    ####################################

    envs = dynRandeEnv(env_name=env_name, obs_norm=args.obs_norm, tag="%s%s%s"%(algorithm_name, tag, rnn_tag), nenvs=nenvs, dyn_range=dyn_range, seed=0)
    eval_env = dynRandeEnv(env_name=env_name, obs_norm=args.obs_norm, tag="%s%s%seval"%(algorithm_name, tag, rnn_tag), nenvs=1, dyn_range=dyn_range, seed=1234567)
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


    for i_episode in range(1,max_episodes+1, nenvs):
        
        state, param = envs.reset()
        last_action = np.stack([envs.env.action_space.sample() for _ in range(nenvs)],axis=0).squeeze()
        last_action = np.zeros_like(last_action)
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
        q_loss_1,q_loss_2 = [],[]

        # Goal for HER
        goal = np.array([[1,0,0]]*nenvs)

        for step in range(max_steps):
            hidden_in = hidden_out
            if args.rnn == "None":
                action = \
                    td3_trainer.get_action(state, 
                                                    deterministic=DETERMINISTIC, 
                                                    explore_noise_scale=explore_noise_scale)
                hidden_in = hidden_out = None
            elif "HER" in args.rnn:
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
        if "HER" in args.rnn:
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
            for i in range(update_itr*nenvs):
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

        if writer is not None and i_episode%writer_interval == 1:
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


        ######################################
        ### Evaluation #######################
        ######################################

        if i_episode % eval_freq == 1 and i_episode != 1:
            envs.save(os.path.join(savepath+"eval"))
            eval_env.load(os.path.join(savepath+"eval"))
            eval_env.env.training = False
            td3_trainer.policy_net.eval()
            eval_rew, eval_success, eval_angle, imgs = pendulum_test(eval_env, agent=td3_trainer, max_steps=eval_max_steps, test_itr=eval_itr, record=False)
            td3_trainer.policy_net.train()
            if writer is not None:
                writer.add_scalar('eval/reward', eval_rew, i_episode)
                writer.add_scalar('eval/success_rate', eval_success, i_episode)
                writer.add_scalar('eval/angle', np.rad2deg(eval_angle), i_episode)
                wandb.log({'eval/reward':eval_rew,
                            'eval/success_rate':eval_success,
                            'eval/angle': np.rad2deg(eval_angle)},
                            step=i_episode)

        ########################################
        ### Model save #########################
        ########################################

        if i_episode % (int(5000/nenvs)*nenvs) == 0 and i_episode != 1:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            td3_trainer.save_model(os.path.join(savepath,"iter%07d"%i_episode))
            envs.save(os.path.join(savepath,"iter%07d"%i_episode))
            artifact = wandb.Artifact('agent-%s'%now, type='model')
            artifact.add_dir(savepath+"/")
            wandb.log_artifact(artifact)

        if np.mean(scores_window)>=best_score: 
            td3_trainer.save_model(os.path.join(savepath,"best"))
            envs.save(os.path.join(savepath,"best"))
            best_score = np.mean(scores_window)
        
    td3_trainer.save_model(os.path.join(savepath,"final"))
    print('\rFinal\tAverage Score: {:.2f}'.format(np.mean(scores_window)))
    artifact = wandb.Artifact('agent-%s'%now, type='model')
    artifact.add_dir(savepath+"/")
    wandb.log_artifact(artifact)

    wandb.finish()

    envs.close()
    del envs
    
    return mean_rewards, loss_storage, eval_rew, eval_success

def test(args, hparam):
    if args.record:
        disp = Display(visible=False, size=(100, 60))
        disp.start()
    env_name = "Pendulum-v0"
    max_steps = 250
    
    device=torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    if not os.path.isdir(os.path.split(args.path)[0]):
        wandb_artifact("TD3-drone", args.path.split('/')[1])

    # Define environment
    eval_env = dynRandeEnv(env_name=env_name, 
                    #    load_path=args.path,
                       tag="test", 
                       nenvs=1, 
                       dyn_range=dyn_range, 
                       seed=int(123456789),
                       record=args.record)
    eval_env.env.training = False
    td3_trainer = td3_agent(env=eval_env,
                rnn=args.rnn,
                device=device,
                hparam=hparam)
    td3_trainer.load_model(args.path)
    td3_trainer.policy_net.eval()
    
    eval_rew, eval_success, eval_angle, imgs = pendulum_test(eval_env, agent=td3_trainer, max_steps=max_steps, test_itr=10, record=args.record, log=True)
    print("EVALUATION REWARD:",eval_rew)
    print("EVALUATION SUCCESS RATE:",eval_success)
    print("EVALUATION ANGLE ERROR[deg]:",np.rad2deg(eval_angle))

    if not os.path.isdir("videos/pendulum"):
        os.makedirs("videos/pendulum")
    dtime = datetime.now()
    save_frames_as_gif(imgs, "videos/pendulum", dtime.strftime("%y%b%d%H%M%S")+".gif")

    if args.record:
        disp.stop()

if __name__=='__main__':
    # train(1000, 'CartPole-v1')
    parser = argparse.ArgumentParser()

    # Common arguments
    parser.add_argument('--gpu', default='0', type=int, help="gpu number")
    parser.add_argument('--rnn', choices=['None','RNN2','GRU2','LSTM2',
                                            'RNN3','GRU3','LSTM3']
                                , default='None', help='Use memory network (LSTM)')
    parser.add_argument('--policy_actf', type=str, default='tanh', help="policy activation function")
    parser.add_argument('--obs_norm', action='store_true', help='use batchnorm for input normalization')

    # Arguments for training 
    parser.add_argument('--tb_log', action='store_true', help="Tensorboard logging")
    parser.add_argument('--hparam', action='store_true', help="find hparam set")
    parser.add_argument('--lr_scheduler', action='store_true', help="Use lr scheduler")
    parser.add_argument('--reward_norm', action='store_true', help="reward normalization")
    parser.add_argument('--her_gamma', default=0.0, type=float)
    parser.add_argument('--positive_rew', action='store_true', help="use [0,1] reward instead of [-1,0]")
    parser.add_argument('--large_eps', action='store_true', help="use large epsilon")
    parser.add_argument('--angvel_goal', action='store_true', help='use angular velocity instead of angle as the goal')
    parser.add_argument('--her_length', type=int, default=50, help='sequence length for her')
    parser.add_argument('--behavior_path', default=None, help='path for behavior networks')

    # Arguments for test
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--path', type=str, default=None, help='required only at test phase')
    parser.add_argument('--record', action='store_true', help='whether record or not')
    parser.add_argument('--task', default='stabilize',choices=['stabilize', 'stabilize-record', 'takeoff'],
                        help='For takeoff-aviary-v0 environment')


    args = parser.parse_args()
    if not args.test:
        if args.hparam:
            keys, values = zip(*hparam_set.items())
            hparam_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

            for i_hparam, hparam in enumerate(hparam_list):
                hparam.update(vars(args))
                hparam['policy_actf'] = getattr(F,args.policy_actf)
                mean_rewards, loss_storage, eval_rew, eval_success = train(args, hparam)
        else:
            hparam = dict([(k,v[0]) for k,v in hparam_set.items()])
            hparam.update(vars(args))
            hparam['policy_actf'] = getattr(F,args.policy_actf)
            train(args, hparam)
    else:
        hparam = dict([(k,v[0]) for k,v in hparam_set.items()])
        hparam.update(vars(args))
        hparam['policy_actf'] = getattr(F,args.policy_actf)
        test(args, hparam)