import numpy as np
import torch
from matplotlib import animation
import matplotlib.pyplot as plt
import os
import gym
from PIL import Image

def collect_trajectories(envs, agent, rollout_length=200):
    """collect trajectories for a parallelized parallelEnv object
    
    Returns : Shape
    ======
    log_probs_old (tensor)   :  (rollout_length*n,)
    states (tensor)          :  (rollout_length*n, envs.observation_space.shape[0])
    actions (tensor)         :  (rollout_length*n, action_dim)
    rewards (list,np.array)  :  (rollout_length, n)  --> for advs
    values (list,np.array)   :  (rollout_length, n)  --> for advs
    dones (list,np.array)    :  (rollout_length, n)  --> for advs
    vals_last (list,np.array):  (n,)                 --> for advs
    """
    n=len(envs.ps)         # number of parallel instances
    device = agent.device

    log_probs_old, states, actions, rewards, values, dones = [],[],[],[],[],[]

    obs = envs.reset()
    
    for t in range(rollout_length):
        batch_input = torch.from_numpy(obs).float().to(device)
        traj_info = agent.act(batch_input)

        log_prob_old = traj_info['log_pi_a'].detach()
        action = traj_info['a'].cpu().numpy()
        value = traj_info['v'].cpu().detach().numpy()
        obs, reward, is_done, _ = envs.step(action)
        
        if is_done.any():
            if t < rollout_length-1:
                idx = np.where(is_done==True)
                reward[idx] = 0

        log_probs_old.append(log_prob_old) # shape (rollout_length, n)
        states.append(batch_input)         # shape (rollout_length, n, envs.observation_space.shape[0])
        actions.append(action)             # shape (rollout_length, n)
        rewards.append(reward)             # shape (rollout_length, n)
        values.append(value)               # shape (rollout_length, n)
        dones.append(is_done)              # shape (rollout_length, n)
    
    if isinstance(envs.action_space, gym.spaces.Box):
        action_dim = envs.action_space.shape[0]
    else:
        action_dim = 1

    if action_dim==1:
        log_probs_old = torch.stack(log_probs_old).view(-1,)  
    else:
        log_probs_old = torch.stack(log_probs_old).view(-1,action_dim)  
    states = torch.stack(states)
    states = states.view(-1,envs.observation_space.shape[0])
    actions_numpy = np.concatenate([a[None,:] for a in actions], axis=0)
    if isinstance(envs.action_space, gym.spaces.Box):
        if action_dim==1:
            actions = torch.tensor(actions_numpy, dtype=torch.float32, device=device).view(-1,)
        else:
            actions = torch.tensor(actions_numpy, dtype=torch.float32, device=device).view(-1,action_dim)
    elif isinstance(envs.action_space, gym.spaces.Discrete):
        actions = torch.tensor(actions_numpy, dtype=torch.long, device=device).view(-1,)

    obs = torch.from_numpy(obs).float().to(device)
    traj_info_last = agent.act(obs)
    vals_last = traj_info_last['v'].cpu().detach().numpy()

    # print(log_probs_old.shape, states.shape, actions.shape, len(rewards), len(values), len(dones), vals_last.shape)

    return log_probs_old, states, actions, rewards, values, dones, vals_last

def collect_trajectories_multiagents(envs, agents, rollout_length=200):
    """collect trajectories for a parallelized parallelEnv object
    
    Returns : Shape
    ======
    log_probs_old (tensor)   :  (rollout_length, n)
    states (tensor)          :  (rollout_length, n, envs.observation_space.shape[0])
    actions (tensor)         :  (rollout_length, n)
    rewards (list,np.array)  :  (rollout_length, n)  --> for advs
    values (list,np.array)   :  (rollout_length, n)  --> for advs
    dones (list,np.array)    :  (rollout_length, n)  --> for advs
    vals_last (list,np.array):  (n,)                 --> for advs
    """
    n=len(envs.ps)         # number of parallel instances
    device = agents.agents[0].device

    log_probs_old, states, actions, rewards, values, dones = [],[],[],[],[],[]

    obs = envs.reset()
    
    for t in range(rollout_length):
        batch_input = torch.from_numpy(obs).float().to(device)
        traj_info = agents.act(batch_input)

        log_prob_old = traj_info['log_pi_a'].detach()
        action = traj_info['a'].cpu().numpy()
        value = traj_info['v'].cpu().detach().numpy()
        obs, reward, is_done, _ = envs.step(action)
        
        if is_done.any():
            if t < rollout_length-1:
                idx = np.where(is_done==True)
                reward[idx] = 0

        log_probs_old.append(log_prob_old) # shape (rollout_length, n)
        states.append(batch_input)         # shape (rollout_length, n, envs.observation_space.shape[0])
        actions.append(action)             # shape (rollout_length, n)
        rewards.append(reward)             # shape (rollout_length, n)
        values.append(value)               # shape (rollout_length, n)
        dones.append(is_done)              # shape (rollout_length, n)
    
    if isinstance(envs.action_space, gym.spaces.Box):
        action_dim = envs.action_space.shape[0]
    else:
        action_dim = 1

    log_probs_old = torch.stack(log_probs_old) 
    states = torch.stack(states)
    actions_numpy = np.concatenate([a[None,:] for a in actions], axis=0)
    if isinstance(envs.action_space, gym.spaces.Box):
        if not action_dim == 1:
            actions = torch.tensor(actions_numpy, dtype=torch.float32, device=device).view(rollout_length,-1,action_dim)
        else:
            actions = torch.tensor(actions_numpy, dtype=torch.float32, device=device).view(rollout_length,-1)
    elif isinstance(envs.action_space, gym.spaces.Discrete):
        actions = torch.tensor(actions_numpy, dtype=torch.long, device=device).view(rollout_length,-1)

    obs = torch.from_numpy(obs).float().to(device)
    traj_info_last = agents.act(obs)
    vals_last = traj_info_last['v'].cpu().detach().numpy()

    # print(log_probs_old.shape, states.shape, actions.shape, len(rewards), len(values), len(dones), vals_last.shape)

    return log_probs_old, states, actions, rewards, values, dones, vals_last

def random_sample(inds, minibatch_size):
    inds = np.random.permutation(inds)
    batches = inds[:len(inds) // minibatch_size * minibatch_size].reshape(-1, minibatch_size)
    for batch in batches:
        yield torch.from_numpy(batch).long()
    r = len(inds) % minibatch_size
    if r:
        yield torch.from_numpy(inds[-r:]).long()

# def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

#     #Mess with this to change frame size
#     plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

#     patch = plt.imshow(frames[0])
#     plt.axis('off')

#     def animate(i):
#         patch.set_data(frames[i])

#     anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
#     anim.save(os.path.join(path, filename), writer='imagemagick', fps=60)
#     print("%s gif saved"%(filename))
#     plt.close()

def save_frames_as_gif(frames, path, filename):
    imgs = [Image.fromarray(img) for img in frames]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    if not os.path.isdir(path):
        os.makedirs(path)
    imgs[0].save(os.path.join(path,filename), save_all=True, append_images=imgs[1::2], duration=20, loop=0)