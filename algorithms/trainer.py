from logging import getLogger
import wandb
from omegaconf import OmegaConf

import torch
import numpy as np

from algorithms.sac.agent import sac_agent

class Trainer:
    def __init__(self, cfg, env, eval_env=None):
        self.cfg = cfg
        self.env = env
        self.eval_env = eval_env

        self.device = self.get_device()
        self.trainer = self.get_trainer()
        self.logger = self.get_logger()

        self.cur_step = 0
        self.cur_episode = 0

        self.goal_step = 0

    def get_device(self):
        if self.cfg.gpu >= 0:
            device=torch.device("cuda:%d"%self.cfg.gpu if torch.cuda.is_available() else "cpu")
            torch.cuda.set_device(device)
        else:
            device=torch.device('cpu')
        return device

    def get_trainer(self):
        # TODO:
        if 'SAC' in self.cfg.algorithms.name:
            trainer = \
                sac_agent(env=self.env,
                        device=self.device,
                        cfg=self.cfg
                        )
            return trainer
        elif 'TD3' in self.cfg.algorithms.name:
            return NotImplementedError
        else:
            return NotImplementedError

    def get_logger(self):
        logger = wandb.init(project=self.cfg.wandb.project,
                                config=OmegaConf.to_container(self.cfg),
                                save_code=True,
                                monitor_gym=True)
        wandb.run.name = "%s-%s-%s-%s"%(self.cfg.envs.name,
                                    self.cfg.algorithms.name,
                                    self.cfg.algorithms.rnn,
                                    self.cfg.seed)
        wandb.run.save()
        return logger

    def run(self):
        for i_episode in range(1, self.cfg.max_episodes+1):
            # Rollout + update
            self.cur_episode = i_episode
            self.rollout()

            # Evaluation
            if i_episode % self.cfg.eval_interval == 0:
                self.evaluation()

    def rollout(self):
        state, param = self.env.reset()
        last_action  = -np.ones(self.env.action_space.shape)[None,:]

        hidden_out = None
        # if "LSTM" in self.cfg.algorithms.rnn:
        #     hidden_out = [torch.zeros([1, 1, self.cfg.algorithms.actor_hidden_dim], dtype=torch.float).to(self.device)] * 2
        # else:
        #     hidden_out = torch.zeros([1, 1, self.cfg.algorithms.actor_hidden_dim], dtype=torch.float).to(self.device)

        # Rollout
        rollout_buffer = {k: [] for k in ['state','action','last_action','reward','next_state','done']}
        log_buffer = {}
        for ep_step in range(self.cfg.max_steps):
            hidden_in = hidden_out
            action, hidden_out = \
                self.trainer.get_action(state,
                                        last_action,
                                        hidden_in)
            next_state, reward, done, info = self.env.step(action)

            # TODO: save to replay buffer
            data = {
                'state': state,
                'action': action,
                'last_action': last_action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            }
            map(lambda x,y: rollout_buffer[x].append(y), data.items())

            state = next_state
            last_action = action

            # update
            self.cur_step += 1
            if self.cur_episode > self.cfg.learning_start:
                if self.cur_step % self.cfg.update_itr == 0:
                    for _ in range(self.cfg.gradient_steps):
                        loss_dict = self.trainer.update(self.cfg.batch_size)
                        loss_dict['global_step'] = self.cur_step
                        # log loss values
                        self.logger.log(loss_dict, self.cur_episode)

        # Push data to replay buffer
        self.trainer.replay_buffer.push_batch(rollout_buffer)
        
        # Logging
        rollout_log = {}
        for key,item in info.items():
            # Last 100 steps average states (pos, attitude, vel, ang vel)
            rollout_log['rollout/%s'%key] = item
        rollout_log['train/return'] = sum(rollout_buffer['reward'])
        rollout_log['global_step'] = self.cur_step
        self.logger.log(rollout_log, step=self.cur_episode)

    def evaluation(self):
        eval_buffer = {'reward': [], 'success_rate': [], 'time_step': []}
        img_buffer = []
        for trial in range(self.cfg.eval_itr):
            state, param = self.eval_env.reset()
            if trial==self.cfg.eval_itr-1:
                img_buffer.append(self.eval_env.render('rgb_array'))
            last_action  = -np.ones(self.eval_env.action_space.shape)[None,:]
            
            hidden_out = None
            # if "LSTM" in self.cfg.rnn:
            #     hidden_out = [torch.zeros([1, 1, self.cfg.policy_dim], dtype=torch.float).to(self.device)] * 2
            # else:
            #     hidden_out = torch.zeros([1, 1, self.cfg.policy_dim], dtype=torch.float).to(self.device)

            eval_buffer['reward'].append(0)
            is_success = 0
            for ep_step in range(1, self.cfg.max_steps+1):
                hidden_in = hidden_out
                action, hidden_out = \
                    self.trainer.get_action(state,
                                            last_action,
                                            hidden_in)
                state, reward, done, info = self.eval_env.step(action)

                if trial==self.cfg.eval_itr-1:
                    img_buffer.append(self.eval_env.render('rgb_array'))

                eval_buffer['reward'][-1] += reward
                last_action = action

                if self.end_check(state):
                    is_success = 1
                    break

            eval_buffer['time_step'].append(ep_step)
            eval_buffer['success_rate'].append(is_success)
        
        # Logging
        for k,v in eval_buffer.items():
            eval_buffer['eval/%s'%k] = np.mean(v)
        eval_buffer['global_steps'] = self.cur_step
        eval_buffer['eval/video'] = wandb.Video(np.stack(img_buffer, axis=0)[::2], 
                                                caption=self.param_str(param), 
                                                fps=10)
        self.logger.log(eval_buffer, step=self.cur_episode)

    def end_check(self, state):
        if self.cfg.envs.name == 'pendulum':
            e_a = np.arccos(np.clip(state[0,0], -1.0, 1.0)) # angle (rad)
            e_a.append(e_a)
            goal_achieve = e_a < np.deg2rad(self.cfg.envs.goal_angle)
            if goal_achieve:
                self.goal_step += 1
            else:
                self.goal_step = 0

            if self.goal_step >= self.cfg.envs.goal_steps:
                return True
            else:
                return False

        elif self.cfg.envs.name == 'drone':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def param_str(self, param):
        result = ""
        for i, key in enumerate(self.env.dyn_range.keys()):
            result += "%s: %f, "%(key, param[i])
        return result[:-2]