import gym
import numpy as np

class dynRandGymEnv(gym.Wrapper):
    def __init__(self, 
                env_name,
                max_episode_len=None,
                dyn_range=dict(),
                seed=0
                ):
        self.env_name = env_name
        self.env = gym.make(env_name)
        super().__init__(self.env)
        
        self.seed = seed
        self.dyn_range = dyn_range
        self.max_episode_len = max_episode_len
        self.env.seed(self.seed)
        self.origin_params = {
            k: getattr(self.env, k) for k in self.dyn_range.keys()
        }

        self.steps = 0

    def get_rand(self, param):
        origin = self.origin_params[param]
        r = self.dyn_range.get(param, 1)
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

    def norm_reward(self, reward):
        if "CartPole" in self.env_name:
            return reward
        elif "Pendulum" in self.env_name:
            reward = (reward + 8.1) / 8.1
            return reward
        else:
            raise NotImplementedError

    def reset_dependency(self):
        '''
            Reset parameters which has dependency
        '''
        if "CartPole" in self.env_name:
            self.env.total_mass = self.env.masspole + self.env.masscart
            self.env.polemass_length = self.env.masspole * self.env.length
        

    def reset(self):
        self.steps = 0

        norm_params = []
        for dyn_param in self.dyn_range.keys():
            param, norm_param = self.get_rand(dyn_param)
            setattr(self.env, dyn_param, param)
            norm_params.append(norm_param)
        self.reset_dependency()

        return self.env.reset(), np.stack(norm_params, axis=0)

    def step(self, action):
        self.steps += 1
        next_state, reward, done, info = self.env.step(action)
        reward = self.norm_reward(reward)

        if self.steps == self.max_episode_len:
            done[:] = False

        return next_state, reward, done, info
        