from email.policy import default
import gym

import numpy as np
import gym
import os
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod

from .customEnvDrone import customAviary, domainRandomAviary
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
import gym_pybullet_drones
import time

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    """

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        #logger.warn('Render not defined for %s' % self)
        pass
        
    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class parallelEnv(VecEnv):
    def __init__(self, env_name='PongDeterministic-v4',
                 n=4, seed=None,
                 spaces=None):

        env_fns = [ gym.make(env_name) for _ in range(n) ]

        if seed is not None:
            for i,e in enumerate(env_fns):
                e.seed(i+seed)
        
        """
        envs: list of gym environments to run in subprocesses
        adopted from openai baseline
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True




# Random physical properties env
def domainRandomize(env, origin_value=dict(), dyn_range=dict(),seed=None):
    def get_rand(name):
        origin = origin_value.get(name, getattr(env.env, name))
        r = dyn_range.get(name, 1)
        ratio = np.random.uniform(1-r, r-1)
        if abs(ratio) < 0.0001:
            ratio = 1
        elif ratio < 0:
            ratio = 1/(1-ratio)
        else:
            ratio = ratio+1
        
        value = origin * ratio
        norm = 2*(ratio-1/(1-r))/(r+1-1/(1-r))-1 if not r==1 else 0
        return value, norm # Normalized value (-1~1)

    if seed is not None:
        np.random.seed(seed)
    env_name = env.env_name
    if 'CartPole' in env_name:
        env.env.masscart, norm_masscart = get_rand('masscart')
        env.env.masspole, norm_masspole = get_rand('masspole')
        env.env.total_mass = env.env.masspole + env.env.masscart
        env.env.length, norm_length = get_rand('length')
        env.env.polemass_length = env.env.masspole * env.env.length
        env.env.force_mag, norm_forcemag = get_rand('force_mag')

        return np.array([norm_masscart, norm_masspole, norm_length, norm_forcemag])
    elif 'Pendulum' in env_name:
        env.env.max_torque, norm_maxtorque = get_rand('max_torque')
        env.env.m, norm_m = get_rand('m')
        env.env.l, norm_l = get_rand('l')
        return np.array([norm_maxtorque, norm_m, norm_l])
    elif 'aviary' in env_name:
        norm_param = env.random_urdf()
        return norm_param
    else:
        raise NotImplementedError

# multithreading 
def workerDomainRand(remote, parent_remote, env_fn_wrapper, randomize, origin_value, dyn_range, idx):
    parent_remote.close()
    env = env_fn_wrapper.x
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_rand':
            param = domainRandomize(env, origin_value=origin_value,
                                    dyn_range=dyn_range,
                                    seed=idx+np.random.randint(2147483647))
            ob = env.reset()
            remote.send((ob,param))
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class domainRandeEnv(parallelEnv):
    '''
    Domain randomize environment
    '''
    def __init__(self, 
                env_name='CartPole-v1',
                tag='simple',
                n=4, seed=0,
                randomize=False, # True: domain randomize(=Randomize every reset)
                dyn_range=dict() # physical properties range
                ):
        
        self.env_name = env_name
        self.original_phisical_params = {}
        if not 'aviary' in env_name:
            env_fns = [ gym.make(env_name) for _ in range(n) ]
            if "CartPole" in env_name:
                self.original_phisical_params.update({'masscart':env_fns[0].masscart,
                                                        'masspole':env_fns[0].masspole,
                                                        'length':env_fns[0].length,
                                                        "force_mag":env_fns[0].force_mag})
            elif "Pendulum" in env_name:
                self.original_phisical_params.update({'max_torque':env_fns[0].max_torque,
                                                        'm':env_fns[0].m,
                                                        'l':env_fns[0].l})

            for i, env_fn in enumerate(env_fns):
                setattr(env_fn, 'env_name', env_name)
                domainRandomize(env=env_fn, dyn_range=dyn_range, seed=i+seed)
                env_fn.seed(i+seed)
        else:
            env_fns = []
            for idx in range(n):
                env = gym.make(id=env_name, # arbitrary environment that has state normalization and clipping
                    drone_model=DroneModel.CF2X,
                    initial_xyzs=np.array([[0.0,0.0,10000.0]]),
                    initial_rpys=np.array([[0.0,0.0,0.0]]),
                    physics=Physics.PYB_GND_DRAG_DW,
                    freq=200,
                    aggregate_phy_steps=1,
                    gui=False,
                    record=False, 
                    obs=ObservationType.KIN,
                    act=ActionType.RPM)
                env = domainRandomAviary(env, tag+str(time.time_ns()), idx, seed,
                    observable=['pos', 'rotation', 'vel', 'angular_vel', 'rpm'],
                    frame_stack=1,
                    task='stabilize2',
                    reward_coeff={'pos':0.2, 'vel':0.0, 'ang_vel':0.02, 'd_action':0.01},
                    episode_len_sec=2,
                    max_rpm=66535,
                    initial_xyzs=[[0.0,0.0,10000.0]], # Far from the ground
                    freq=200,
                    rpy_noise=np.pi/4,
                    vel_noise=2.0,
                    angvel_noise=np.pi/2,
                    mass_range=dyn_range.get('mass_range', 0.0),
                    cm_range=dyn_range.get('cm_range', 0.0),
                    kf_range=dyn_range.get('kf_range', 0.0),
                    km_range=dyn_range.get('km_range', 0.0),
                    i_range=dyn_range.get('i_range', 0.0),
                    battery_range=dyn_range.get('battery_range', 0.0))
                setattr(env, 'env_name', env_name)
                env_fns.append(env)
        
        """
        envs: list of gym environments to run in subprocesses
        adopted from openai baseline
        """
        self.randomize = randomize

        self.waiting = False
        self.closed = False
        self.nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs)])
        self.ps = [Process(target=workerDomainRand, args=(work_remote, remote, CloudpickleWrapper(env_fn),randomize,self.original_phisical_params,dyn_range,idx))
            for (work_remote, remote, env_fn, idx) in zip(self.work_remotes, self.remotes, env_fns, range(len(env_fns)))]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

        self.env_step = 0
        
    def reset(self):
        self.env_step = 0
        for remote in self.remotes:
            if self.randomize:
                remote.send(('reset_rand', None))
            else:
                remote.send(('reset', None))
                
        if self.randomize:
            results = np.stack([remote.recv() for remote in self.remotes])
            obs, param = zip(*results)
            return np.stack(obs), np.stack(param)
        else:
            return np.stack([remote.recv() for remote in self.remotes]), None

    def step(self, action):
        """
        Reward normalization
        """
        self.env_step += 1
        if "CartPole" in self.env_name:
            obs, reward, is_done, info = super().step(action)
            reward = reward - np.clip(np.power(obs[:,0]/2.4,2),0,1)
            return obs, reward, is_done, info
        elif "Pendulum" in self.env_name:
            obs, reward, is_done, info = super().step(action)
            reward = (reward + 8.1) / 8.1
            return obs, reward, is_done, info
        elif 'aviary' in self.env_name:
            action = action.reshape((self.nenvs,-1))
            obs, reward, is_done, info = super().step(action)
            return obs, reward, is_done, info
        else:
            raise NotImplementedError

class VecDynRandEnv(VecEnvWrapper):
    def __init__(self, venv: VecEnv):
        super().__init__(venv=venv, observation_space=venv.observation_space)

    def reset(self) -> np.ndarray:
        params = []
        for env in self.venv.venv.envs:
            params.append(env.random_urdf())
        params = np.stack(params, axis=0)
        obs = self.venv.reset()
        return obs, params

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        return obs, reward, done, info

class dynRandeEnv:
    def __init__(self, 
                env_name='takeoff-aviary-v0',
                tag='randomize',
                task='stabilize',
                obs_norm=False,
                nenvs=4, seed=0,
                load_path=None,
                dyn_range=dict(), # physical properties range
                record=False):
        self.env_name = env_name
        self.tag = tag
        self.task = task
        self.seed = seed
        self.dyn_range = dyn_range
        self.nenvs = nenvs
        self.dyn_range = dyn_range
        self.record = record

        self.obs_norm = obs_norm

        envs = []
        for idx in range(nenvs):
            env = self.drone_env(idx)
            envs.append(env)
        self.env = DummyVecEnv([lambda: env for env in envs])
        if load_path is not None:
            self.env = VecNormalize.load(os.path.join(load_path,'env.pkl'), self.env)
        else:
            self.env = VecNormalize(self.env, norm_obs=self.obs_norm, norm_reward=False)
        self.env = VecDynRandEnv(self.env)

    def drone_env(self, idx):  
        if self.task == 'stabilize':
            initial_xyzs = [[0.0,0.0,10000.0]]
            rpy_noise=np.pi/4
            vel_noise=2.0
            angvel_noise=np.pi/2
            goal = None
        elif self.task == 'takeoff':
            initial_xyzs = [[0.0,0.0,0.025]]
            rpy_noise=0
            vel_noise=0
            angvel_noise=0
            goal = np.array([[0.0,0.0,1.0]])
        else:
            raise NotImplementedError("please choose task among [stabilize, takeoff]")
        env = gym.make(id=self.env_name, # arbitrary environment that has state normalization and clipping
            drone_model=DroneModel.CF2X,
            initial_xyzs=np.array(initial_xyzs),
            initial_rpys=np.array([[0.0,0.0,0.0]]),
            physics=Physics.PYB_GND_DRAG_DW,
            freq=200,
            aggregate_phy_steps=1,
            gui=self.record,
            record=self.record, 
            obs=ObservationType.KIN,
            act=ActionType.RPM)
        env = domainRandomAviary(env, self.tag+str(time.time_ns()), idx, self.seed+idx,
            observable=['pos', 'rotation', 'vel', 'angular_vel', 'rpm'],
            frame_stack=1,
            task='stabilize2',
            # reward_coeff={'pos':0.2, 'vel':0.0, 'ang_vel':0.02, 'd_action':0.01},
            reward_coeff={'pos':0.2, 'vel':0.016, 'ang_vel':0.005, 'd_action':0.002},
            episode_len_sec=2,
            max_rpm=66535,
            initial_xyzs=initial_xyzs, # Far from the ground
            freq=200,
            rpy_noise=rpy_noise,
            vel_noise=vel_noise,
            angvel_noise=angvel_noise,
            mass_range=self.dyn_range.get('mass_range', 0.0),
            cm_range=self.dyn_range.get('cm_range', 0.0),
            kf_range=self.dyn_range.get('kf_range', 0.0),
            km_range=self.dyn_range.get('km_range', 0.0),
            i_range=self.dyn_range.get('i_range', 0.0),
            battery_range=self.dyn_range.get('battery_range', 0.0),
            goal=goal)
        setattr(env, 'env_name', self.env_name)

        return env

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
    
    def normalize_obs(self, obs):
        return self.env.normalize_obs(obs)

    def unnormalize_obs(self, obs):
        return self.env.unnormalize_obs(obs)

    def save(self, path):
        self.env.save(path+'env.pkl')
    
    def load(self,path):
        envs = []
        for idx in range(self.nenvs):
            env = self.drone_env(idx)
            envs.append(env)
        self.env = DummyVecEnv([lambda: env for env in envs])
        self.env = VecNormalize.load(path+'env.pkl', self.env)
        self.env = VecDynRandEnv(self.env)