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

    max_episodes  = int(37500)
    max_steps = hparam['max_steps']
    update_itr = hparam['update_itr']
    writer_interval = 20
    eval_interval = 500
    model_save_interval = 500
    learning_start = hparam['learning_starts']

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
    if hparam['param'] or (hparam['rnn']!='None'):
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

    env = dynRandeEnv(
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
    )
    eval_env = dynRandeEnv(
        initial_xyzs=np.array([[0,0,10000.0]]),
        initial_rpys=np.array([[0,0,0]]),
        observable=observable,
        dyn_range={}, # No dynamic randomization for evaluation env
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
    )
    trainer = sac_agent(env=env,
                rnn=args.rnn,
                device=device,
                hparam=hparam,
                replay_buffer_size=replay_buffer_size)
    if args.test == 0:
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
                if args.rnn in ["None","RNNparam","GRUparam","LSTMparam"]:
                    action = \
                        trainer.get_action(state, last_action)
                    hidden_in = hidden_out = None
                else:
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

                # Update TD3 trainer\
                total_step = (i_episode-1) * max_steps + ep_step
                if total_step > learning_start:
                    if total_step % update_itr == 0:
                        for _ in range(gradient_steps):
                            loss_dict = trainer.update(batch_size)
                            policy_loss.append(loss_dict['policy_loss'])
                            q_loss_1.append(loss_dict['q_loss_1'])
                            q_loss_2.append(loss_dict['q_loss_2'])
                            param_loss.append(loss_dict.get('param_loss',0))

            episode_done[-1] = np.ones_like(episode_done[-1])
            
            # Push into Experience replay buffer     
            trainer.replay_buffer.push_batch(episode_state, 
                            episode_action, 
                            episode_last_action,
                            episode_reward, 
                            episode_next_state, 
                            episode_done)

            loss_storage['policy_loss'].append(np.mean(policy_loss))
            loss_storage['q_loss_1'].append(np.mean(q_loss_1))
            loss_storage['q_loss_2'].append(np.mean(q_loss_2))
            loss_storage['param_loss'].append(np.mean(param_loss))
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
                            'loss/loss_param': np.mean(loss_storage['param_loss']),
                            'rollout/rewards': np.mean(scores_window)})
                rollout_log['global_step'] = i_episode * max_steps
                wandb.log(rollout_log, step=i_episode)

            ######################################
            ### Evaluation #######################
            ######################################

            if i_episode % eval_interval == 0:
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
    
    else:
        # Test code
        max_steps = 1400
        eval_env = dynRandeEnv(
            initial_xyzs=np.array([[0,0,10000.0]]),
            initial_rpys=np.array([[0,0,0]]),
            observable=observable,
            dyn_range={}, # No dynamic randomization for evaluation env
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
        )
        trainer.load_model(args.path)
        total_info = []

        for itr in range(args.test):

            obs = eval_env.reset()
            state = eval_env._getDroneStateVector(0).squeeze()
            reward_sum = 0
            state_buffer,obs_buffer, action_buffer = [],[],[]
            goal_state = np.zeros_like(state)
            goal_state[:3] = eval_env.goal_pos[0]
            last_action = -np.ones(eval_env.action_space.shape)[None,:]
            
            for i in range(max_steps):
                # obs[:,-15:] = 0
                action = trainer.get_action(obs, last_action)
                # action, *_ = ctrl.computeControlFromState(control_timestep=env.venv.envs[0].env.TIMESTEP,
                #                                                            state=state,
                #                                                            target_pos=env.venv.envs[0].env.goal_pos.squeeze(),
                #                                                            target_rpy=np.array([0,0,0])
                #                                                            )
                # action = 2*(action/24000)-1

                obs_buffer.append(obs)
                state_buffer.append(state)
                action_buffer.append(action)

                obs, reward, done, info = eval_env.step(action)
                state = eval_env._getDroneStateVector(0).squeeze()
                
                if args.render:
                    eval_env.render()
                    input()
                if done:
                    obs = eval_env.reset()
                reward_sum+=reward
            
            state_buffer.append(goal_state)
            np.savetxt('paperworks/test_state_%02d.txt'%itr,np.stack(state_buffer),delimiter=',')
            np.savetxt('paperworks/test_obs_%02d.txt'%itr,np.stack(obs_buffer),delimiter=',')
            np.savetxt('paperworks/test_action_%02d.txt'%itr,np.concatenate(action_buffer),delimiter=',')
            print("iteration : ",itr)
            info.update({'reward': reward_sum})
            print(info)

            total_info.append(info)
        
        final_info = {}
        for info in total_info:
            for key,value in info.items():
                if key != 'episode':
                    final_info[key] = final_info.get(key,0) + value
        for key,value in final_info.items():
            final_info[key] = value / args.test

        print("Average results")
        print(final_info)


if __name__=='__main__':
    # train(1000, 'CartPole-v1')
    parser = argparse.ArgumentParser()

    # Common arguments
    parser.add_argument('--gpu', default='0', type=int, help="gpu number")
    parser.add_argument('--rnn', choices=['None',
                                                'RNNparam','GRUparam','LSTMparam',
                                                'RNN','GRU','LSTM',
                                                'RNNfull','GRUfull','LSTMfull',
                                                'RNNpolicy','GRUpolicy','LSTMpolicy',
                                                'RNNpolicyfull','GRUpolicyfull','LSTMpolicyfull'])

    # Arguments for training 
    parser.add_argument('--tb_log', action='store_true', help="Tensorboard logging")
    parser.add_argument('--param', action='store_true', help="Use param observation")
    parser.add_argument('--rew_angvel_xy', type=float, default=0.0, help="Reward for angvel xy")
    parser.add_argument('--rew_angvel_z', type=float, default=0.0, help="Reward for angvel z")
    parser.add_argument('--rew_angvel', type=float, default=0.0, help="Reward for angvel xyz")
    parser.add_argument('--no_random', action='store_true')
    parser.add_argument('--seed', type=int, default=0)

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
    main(args, hparam)