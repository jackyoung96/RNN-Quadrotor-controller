from ppo import PPODiscreteAgent, PPOContinuousAgent
import torch
import gym
import matplotlib
from matplotlib import animation, rc
from envs.customEnv import domainRandomize
import argparse
import os
from utils import save_frames_as_gif
from pyvirtualdisplay import Display

def test(args):
    env_name, randomize = args.env, args.randomize
    
    if randomize:
        dyn_range = {
        # cartpole
        'masscart': [1.3,1.3], # original 1.0
        'masspole': [0.13, 0.13], # original 0.1
        'length': [0.65, 0.65], # original 0.5
        'force_mag': [8.0, 8.0], # original 10.0

        # pendulum
        'max_torque': [1.7, 1.7], # original 2.0
        'm': [1.3, 1.3], # original 1.0
        'l': [1.3, 1.3], # original 1.0
        }
        default_ratio = 0.1
    else:
        dyn_range = {}
        default_ratio = 0.0

    envs= gym.make(env_name)
    domainRandomize(envs.env, dyn_range)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if isinstance(envs.action_space, gym.spaces.Discrete):
        agent = PPODiscreteAgent(state_size=envs.observation_space.shape[0],
                        action_size=envs.action_space.n, 
                        seed=0,
                        hidden_layers=[64,64],
                        lr_policy=1e-4, 
                        use_reset=True,
                        device=device)
    elif isinstance(envs.action_space, gym.spaces.Box):
        agent = PPOContinuousAgent(state_size=envs.observation_space.shape[0],
                        action_size=envs.action_space.shape[0], 
                        seed=0,
                        hidden_layers=[64,256],
                        lr_policy=1e-4, 
                        use_reset=False,
                        use_common=True,
                        device=device)
    else:
        raise NotImplementedError
    
    if os.path.isfile('save/ppo/policy_%s_best.pth'%env_name):
        agent.policy.load_state_dict(torch.load('save/ppo/policy_%s_best.pth'%env_name, map_location=lambda storage, loc: storage))
    elif os.path.isfile('save/ppo/policy_%s.pth'%env_name):
        agent.policy.load_state_dict(torch.load('save/ppo/policy_%s.pth'%env_name, map_location=lambda storage, loc: storage))
    else:
        raise "No pth file exist"

    obs = envs.reset()
    done = False
    total_rew = 0
    frames = []
    agent.policy.eval()

    with torch.no_grad():
        max_episode_len = 200
        for _ in range(max_episode_len):
            if args.render:
                envs.render()
            if args.record:
                frames.append(envs.render(mode="rgb_array"))
            if "CartPole" in env_name: 
                obs_input = torch.from_numpy(obs).unsqueeze(0).float().to(device)
                action = int(agent.policy.act(obs_input)['a'].cpu().numpy())
            elif "Pendulum" in env_name:
                obs_input = torch.from_numpy(obs).float().to(device).view(1,-1)
                action = agent.policy.act(obs_input)['a'].cpu().numpy()
            obs, rew, is_done, _ = envs.step(action)
            total_rew += rew
            if is_done:
                print("reward :", total_rew)
                break

    if args.record:
        txt_random = 'T' if randomize else 'F'
        if not os.path.isdir("gif"):
            os.mkdir("gif")
        num = 0
        for file in os.listdir("gif"):
            if args.env+"_"+txt_random in file:
                num = max(num,int(file.strip(".gif").split("_")[-1]) + 1)
        
        save_frames_as_gif(frames, path="gif", filename="ppo_%s_%s_%03d.gif"%(args.env, txt_random, num))

    envs.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True, choices=['CartPole-v0','CartPole-v1','Pendulum-v0'])
    parser.add_argument('--render',action='store_true', help="Rendering")
    parser.add_argument('--record',action='store_true', help="record as gif")
    parser.add_argument('--randomize',action='store_true', help="Domain randomize")
    args = parser.parse_args()
    with Display(visible=False, size=(100, 60)) as disp:
        test(args)