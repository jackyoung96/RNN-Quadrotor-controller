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

dyn_range = {
    # drones
    'mass_range': 0.3, # (1-n) ~ (1+n)
    'cm_range': 0.3, # (1-n) ~ (1+n)
    'kf_range': 0.3, # (1-n) ~ (1+n)
    'km_range': 0.3, # (1-n) ~ (1+n)
    'i_range': 0.3,
    't_range': 0.3,
    'battery_range': 0.0 # (1-n) ~ (1)
}

hparam_set = {
    "learning_rate": (np.random.uniform,[1e-4,1e-4]),
    "learning_starts": (np.random.randint,[8000,8001]),
    "activation": (np.random.choice, [[F.relu]]),

    # SAC, TD3
    "update_itr": (np.random.randint,[10,11]),

    "goal_dim": (np.random.randint,[18,19]),
    "param_num": (np.random.randint,[15,16]),
    "hidden_dim": (np.random.randint,[6,7]),
    "critic_dim": (np.random.randint,[7,8]),
    "policy_net_layers": (np.random.randint,[3,4]),
    "critic_net_layers": (np.random.randint,[4,5]),

    "max_steps": (np.random.randint,[800,801]),
    "her_length": (np.random.randint,[800,801]),
    "rnn_dropout": (np.random.uniform,[0,0]),
    "replay_buffer_size": (np.random.randint,[int(1e6), int(1e6+1)]),
    "gradient_steps": (np.random.randint,[1,2]),
}

def train(args, hparam):

    global dyn_range

    #####################################
    # hyper-parameters for RL training ##
    #####################################

    max_episodes  = int(25000)
    max_steps = hparam['max_steps']
    writer_interval = 500
    eval_interval = 500
    model_save_interval = 500
    learning_start = hparam['learning_starts'] // max_steps
    if args.rnn == 'None':
        hparam['gradient_steps'] = hparam['gradient_steps'] * max_steps

    # hparam['learning_rate'] = 10**hparam['learning_rate']
    hparam['hidden_dim'] = int(2**hparam['hidden_dim'])
    hparam['critic_dim'] = int(2**hparam['critic_dim'])
    policy_dim = hparam['hidden_dim']
    critic_dim = hparam['critic_dim']
    policy_net_layers = hparam['policy_net_layers']
    critic_net_layers = hparam['critic_net_layers']
    replay_buffer_size = hparam['replay_buffer_size']
    gradient_steps = hparam['gradient_steps']
    observable = ['rel_pos', 'rotation', 'rel_vel', 'rel_angular_vel']
    if hparam['param']:
        observable += ['param']
    rew_coeff = {'pos':1.0, 'vel':0.0, 'ang_vel': args.rew_angvel, 'ang_vel_xy': args.rew_angvel_xy, 'ang_vel_z': args.rew_angvel_z, 'd_action':0.0, 'rotation': 0.0}
    hparam['observable'] = observable
    hparam['rew_coeff'] = rew_coeff

    if args.no_random:
        dyn_range = {}

    hparam['dyn_range'] = dyn_range

    batch_size  = 128

    #########################################
    ### Path, Basic variable setting ########
    #########################################

    print("hyperparam set:",hparam)
    algorithm_name = 'SAC'
    env_name = "takeoff-aviary-v0"
    dtime = datetime.now()
    now = dtime.strftime("%y%b%d%H%M%S")
    if args.tb_log:
        # wandb
        run = wandb.init(project="SAC-drone-final", config=hparam,
                        # sync_tensorboard=True,
                        save_code=True,
                        monitor_gym=True,)
        wandb.run.name = "%s_%s"%(algorithm_name, now)
        wandb.run.save()
        savepath = f"models/{run.name}"
    
    if args.gpu >= 0:
        device=torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
    else:
        device=torch.device('cpu')
    
    print("Device:",device)

    ####################################
    # Define environment and agent #####
    ####################################

    env, eval_env = [dynRandeEnv(
        initial_xyzs=np.array([[0,0,10000.0]]),
        initial_rpys=np.array([[0,0,0]]),
        observable=observable,
        dyn_range=dyn_range,
        rpy_noise=2*np.pi,
        vel_noise=1,
        angvel_noise=np.pi,
        reward_coeff=rew_coeff,
        frame_stack=1,
        episode_len_sec=max_steps/100,
        gui=args.render,
        record=False,
        wandb_render=True,
        is_noise=not args.no_random,
    ) for _ in range(2)]
    trainer = sac_agent(env=env,
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
        
        state = env.reset()
        last_action = -np.ones(env.action_space.shape)[None,:]
        episode_state = []
        episode_action = []
        episode_last_action = []
        episode_reward = []
        episode_next_state = []
        episode_done = []
        if "LSTM" in args.rnn:
            hidden_out = (torch.zeros([1, 1, policy_dim], dtype=torch.float).to(device), \
                        torch.zeros([1, 1, policy_dim], dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
        else:
            hidden_out = torch.zeros([1, 1, policy_dim], dtype=torch.float).to(device)

        policy_loss = []
        q_loss_1,q_loss_2,param_loss = [],[],[]

        for ep_step in range(max_steps):
            hidden_in = hidden_out
            if args.rnn == "None":
                action = \
                    trainer.get_action(state, last_action)
                hidden_in = hidden_out = None
            elif args.rnn in ["RNNHER", "LSTMHER","GRUHER"]:
                raise NotImplementedError
                action, hidden_out = \
                        trainer.get_action(state, 
                                    last_action, 
                                    hidden_in,
                                    goal=goal)
            else:
                raise NotImplementedError
                action, hidden_out = \
                    trainer.get_action(state, 
                                last_action, 
                                hidden_in)
            next_state, reward, done, info = env.step(action) 
            if not isinstance(reward, np.ndarray):
                reward = np.array([reward])
            if not isinstance(done, np.ndarray):
                done = np.array([done])

            episode_state.append(state)
            episode_action.append(action[0])
            episode_last_action.append(last_action[0])
            episode_reward.append(reward)
            episode_next_state.append(next_state)
            episode_done.append(done)

            state = next_state
            last_action = action

        episode_done[-1] = np.ones_like(episode_done[-1])
        
        # Push into Experience replay buffer
        if args.rnn in ["RNNHER","LSTMHER","GRUHER"]:
            raise NotImplementedError
            trainer.replay_buffer.push_batch(episode_state, 
                            episode_action, 
                            episode_last_action,
                            episode_reward, 
                            episode_next_state, 
                            episode_done,
                            param,
                            goal)
        else:           
            trainer.replay_buffer.push_batch(episode_state, 
                            episode_action, 
                            episode_last_action,
                            episode_reward, 
                            episode_next_state, 
                            episode_done)

        # Update TD3 trainer
        if i_episode > learning_start:
            for _ in range(gradient_steps):
                loss_dict = trainer.update(batch_size)
                policy_loss.append(loss_dict['policy_loss'])
                q_loss_1.append(loss_dict['q_loss_1'])
                q_loss_2.append(loss_dict['q_loss_2'])


        loss_storage['policy_loss'].append(np.mean(policy_loss))
        loss_storage['q_loss_1'].append(np.mean(q_loss_1))
        loss_storage['q_loss_2'].append(np.mean(q_loss_2))
        rewards = np.sum(episode_reward)
        mean_rewards.append(rewards)
        scores_window.append(rewards)

        ##########################################
        ## Tensorboard & WandB writing ###########
        ##########################################

        if args.tb_log and i_episode % writer_interval == 0:
            rollout_log = {}
            for key,item in info.items():
                # Last 100 steps average states (pos, attitude, vel, ang vel)
                rollout_log['rollout/%s'%key] = item
            rollout_log.update({'loss/loss_p':np.mean(loss_storage['policy_loss']),
                        'loss/loss_q_1': np.mean(loss_storage['q_loss_1']),
                        'loss/loss_q_2': np.mean(loss_storage['q_loss_2']),
                        'rollout/rewards': np.mean(scores_window)})
            rollout_log['global_step'] = i_episode * max_steps
            wandb.log(rollout_log, step=i_episode)

        ######################################
        ### Evaluation #######################
        ######################################

        if i_episode % eval_interval == 0 and i_episode != 0:
            trainer.policy_net.eval()
            eval_rew, eval_success, infos = drone_test(eval_env, agent=trainer, max_steps=max_steps, test_itr=50)
            trainer.policy_net.train()

            eval_log = {}
            for info in infos:
                for key,item in info.items():
                    eval_log['eval/%s'%key] = eval_log.get('eval/%s'%key, 0) + item
            for key,item in eval_log.items():
                eval_log[key] = item / len(infos)
            eval_log.update({'eval/reward':eval_rew,
                            'eval/success_rate':eval_success})
            eval_log['global_step'] = i_episode * max_steps

            if args.tb_log:
                wandb.log(eval_log, step=i_episode)

        ########################################
        ### Model save #########################
        ########################################

        if i_episode % model_save_interval == 0 and i_episode != 0:
            if not os.path.isdir(savepath):
                os.makedirs(savepath)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            path = os.path.join(savepath,"iter%07d"%i_episode)
            trainer.save_model(path)
            wandb.save(path+"'_q1.pt'", base_path=savepath)
            wandb.save(path+"'_q2.pt'", base_path=savepath)
            wandb.save(path+"'_policy.pt'", base_path=savepath)

    path = os.path.join(savepath,"final")
    trainer.save_model(path)
    wandb.save(path+"'_q1.pt'", base_path=savepath)
    wandb.save(path+"'_q2.pt'", base_path=savepath)
    wandb.save(path+"'_policy.pt'", base_path=savepath)
    print('\rFinal\tAverage Score: {:.2f}'.format(np.mean(scores_window)))

    env.close()
    eval_env.close()
    del env, eval_env
    
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
                                                'RNN3','GRU3','LSTM3'])

    # Arguments for training 
    parser.add_argument('--tb_log', action='store_true', help="Tensorboard logging")
    parser.add_argument('--param', action='store_true', help="Use param observation")
    parser.add_argument('--rew_angvel_xy', type=float, default=0.0, help="Reward for angvel xy")
    parser.add_argument('--rew_angvel_z', type=float, default=0.0, help="Reward for angvel z")
    parser.add_argument('--rew_angvel', type=float, default=0.0, help="Reward for angvel xyz")
    parser.add_argument('--no_random', action='store_true')

    # parser.add_argument('--dyn', choices=['mass', 'cm', 'kf', 'km', 'i', 't', 'no'], default=None)

    # Arguments for test
    parser.add_argument('--test', type=int, default=0, help='how many times for testing. 0 means training')
    parser.add_argument('--path', type=str, default=None, help='required only at test phase')
    parser.add_argument('--render', action='store_true', help='whether record or not')
    parser.add_argument('--record', action='store_true', help='whether record or not')
    parser.add_argument('--task', default='stabilize',choices=['stabilize', 'stabilize-record', 'takeoff'],
                        help='For takeoff-aviary-v0 environment')


    args = parser.parse_args()

    hparam = dict([(k,v[0](*v[1])) for k,v in hparam_set.items()])
    hparam.update(vars(args))
    train(args, hparam)