from email.policy import default
import queue
import gym

import numpy as np
import gym
from gym.spaces import Box
import os
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod

from .customEnvDrone import customAviary, domainRandomAviary
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
import gym_pybullet_drones
import time
from copy import deepcopy

import pybullet as p
import pybullet_data
from datetime import datetime
from .assets.random_urdf import generate_urdf
from scipy.spatial.transform import Rotation as R

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper

from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary


class dynRandeEnv(TakeoffAviary):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human','rgb_array']}

    def __init__(self, 
                initial_xyzs,
                initial_rpys,
                observable,
                dyn_range,
                rpy_noise,
                vel_noise,
                angvel_noise,
                reward_coeff,
                frame_stack=1,
                episode_len_sec=2,
                gui=False,
                record=False,
                goal=None,
                wandb_render=False,
                is_noise=False,
                **kwargs):

        self.observable = observable
        self.rpy_noise = rpy_noise
        self.vel_noise = vel_noise
        self.angvel_noise = angvel_noise
        self.angvel_bias = np.zeros(3)

        self.mass_range = dyn_range.get('mass_range', 0.0)
        self.cm_range = dyn_range.get('cm_range', 0.0)
        self.i_range = dyn_range.get('i_range', 0.0)
        self.kf_range = dyn_range.get('kf_range', 0.0)
        self.km_range = dyn_range.get('km_range', 0.0)
        self.t_range = dyn_range.get('t_range', 0.0)
        self.norm_range = dyn_range.get('norm_range', 0.3)

        self.frame_stack = frame_stack
        self.frame_buffer = []
        self.reward_coeff = reward_coeff
        self.new_URDF = "cf2x.urdf"
        self.maketime = datetime.now().strftime("%m%d%Y_%H%M%S")

        self.goal = goal
        if goal is None:
            self.goal_pos = deepcopy(initial_xyzs)
        else:
            self.goal_pos = goal.reshape(initial_xyzs.shape)

        super(dynRandeEnv, self).__init__(
            drone_model=DroneModel.CF2X,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=Physics.PYB_DRAG,
            freq=200,
            aggregate_phy_steps=2,
            gui=gui,
            record=record, 
            obs=ObservationType.KIN,
            act=ActionType.RPM
        )
        self.orig_params = {"M":self.M,
                            "L":self.L / np.sqrt(2),
                            "KF":self.KF,
                            "KM":self.KM,
                            "T": 0.15,
                            "BATTERY":1.0}

        self.EPISODE_LEN_SEC = episode_len_sec
        self.MAX_RPM = 21700
        # self.MAX_RPM = 24000

        self.action_space = gym.spaces.Box(low=-1*np.ones(4),
                            high=np.ones(4),
                            dtype=np.float32
                            )
        self.observation_space = self.observable_obs_space()

        self.last_action_custom = -np.ones((4,))[None,:]
        self.droneStates = []
        self.param = np.zeros((14,))
        self.wandb_render = wandb_render
        self.is_noise = is_noise

    def observable_obs_space(self):
        rng = np.inf
        low_dict = {
            'pos': [-rng] * 3,
            'rel_pos': [-rng] * 3,
            'quaternion': [-rng] * 4,
            'rotation': [-rng] * 9,
            'vel': [-rng] * 3,
            'rel_vel': [-rng] * 3,
            'angular_vel': [-rng] * 3,
            'rel_angular_vel': [-rng] * 3,
            'rpm': [-rng] * 4,
            'param': [-rng] * 15
        }
        high_dict = {
            'pos': [rng] * 3,
            'rel_pos': [rng] * 3,
            'quaternion': [rng] * 4,
            'rotation': [rng] * 9,
            'vel': [rng] * 3,
            'rel_vel': [rng] * 3,
            'angular_vel': [rng] * 3,
            'rel_angular_vel': [rng] * 3,
            'rpm': [rng] * 4,
            'param': [rng] * 15
        }
        low, high = [],[]
        for obs in self.observable:
            if obs in low_dict:
                low += low_dict[obs]
                high += high_dict[obs]
            else:
                raise "Observable type is wrong. ({})".format(obs)
        
        low = low * self.frame_stack # duplicate 
        high = high * self.frame_stack # duplicate 

        return gym.spaces.Box(low=np.array(low),
                    high=np.array(high),
                    dtype=np.float32
                )

    def _startVideoRecording(self):
        """Starts the recording of a video output.

        The format of the video output is .mp4, if GUI is True, or .png, otherwise.
        The video is saved under folder `files/videos`.

        """
        if self.RECORD and self.GUI:
            self.VIDEO_ID = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4,
                                                fileName=os.path.dirname(os.path.abspath(__file__))+"/../videos/video-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+".mp4",
                                                physicsClientId=self.CLIENT
                                                )
        if self.RECORD and not self.GUI:
            self.FRAME_NUM = 0
            self.IMG_PATH = os.path.dirname(os.path.abspath(__file__))+"/../videos/video-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+"/"
            os.makedirs(os.path.dirname(self.IMG_PATH), exist_ok=True)

    def _housekeeping(self):
        self.last_action_custom = -np.ones((4,))[None,:]

        self.angvel_bias = np.zeros(3)
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.X_AX = -1*np.ones(self.NUM_DRONES)
        self.Y_AX = -1*np.ones(self.NUM_DRONES)
        self.Z_AX = -1*np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1*np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM=False
        self.last_input_switch = 0
        self.last_action = -1*np.ones((self.NUM_DRONES, 4))
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        self.gui_input = np.zeros(4)
        #### Initialize the drones kinemaatic information ##########
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))
        #### Set PyBullet's parameters #############################
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        #### Load ground plane, drone and obstacles models #########
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)

        # Put gaussian noise to initialize RPY
        init_rpys = []
        for i in range(self.NUM_DRONES):
            init_rpy = self.INIT_RPYS[i,:] + self.rpy_noise*np.random.uniform(-1.0,1.0,self.INIT_RPYS[i,:].shape)
            # init_rpy[i,-1] = init_rpy[i,-1] + np.random.uniform(-np.pi, np.pi) # random yaw
            init_rpys.append(init_rpy)
        self.DRONE_IDS = np.array([p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/assets/"+self.new_URDF,
                                              self.INIT_XYZS[i,:],
                                              p.getQuaternionFromEuler(init_rpys[i]),
                                              flags = p.URDF_USE_INERTIA_FROM_FILE,
                                              physicsClientId=self.CLIENT
                                              ) for i in range(self.NUM_DRONES)])

        # random velocity initialize
        for i in range (self.NUM_DRONES):
            vel = self.vel_noise * np.random.uniform(-1.0,1.0,size=3)
            p.resetBaseVelocity(self.DRONE_IDS[i],\
                                linearVelocity = vel.tolist(),\
                                angularVelocity = (self.angvel_noise * np.random.uniform(-1.0,1.0,size=3)).tolist(),\
                                physicsClientId=self.CLIENT)
            self.goal_pos[i,:] = self.INIT_XYZS[i,:] + np.random.uniform(-1.0,1.0,size=3) if self.goal is None \
                                else self.goal


        for i in range(self.NUM_DRONES):
            #### Show the frame of reference of the drone, note that ###
            #### It severly slows down the GUI #########################
            if self.GUI and self.USER_DEBUG:
                self._showDroneLocalAxes(i)
            #### Disable collisions between drones' and the ground plane
            #### E.g., to start a drone at [0,0,0] #####################
            # p.setCollisionFilterPair(bodyUniqueIdA=self.PLANE_ID, bodyUniqueIdB=self.DRONE_IDS[i], linkIndexA=-1, linkIndexB=-1, enableCollision=0, physicsClientId=self.CLIENT)
        if self.OBSTACLES:
            self._addObstacles()

    def set_urdf(self, set_dyn):
        set_mass, set_cm, set_I, set_T, set_KF, set_KM = set_dyn.get('mass',0.0),\
                                                            set_dyn.get('cm',np.zeros(2)),\
                                                            set_dyn.get('I',np.zeros(3)),\
                                                            set_dyn.get('T',0.0),\
                                                            set_dyn.get('KF',np.zeros(4)),\
                                                            set_dyn.get('KM',np.zeros(4))

        self.new_URDF = self.maketime + "/cf2x.urdf"
        norm_mass, norm_xcm, norm_ycm, norm_ixx, norm_iyy, norm_izz, norm_T = 0,0,0,0,0,0,0
        norm_KF, norm_KM = [0,0,0,0], [0,0,0,0]

        mass = (1+set_mass) * self.orig_params['M']
        x_cm, y_cm = (set_cm) * self.orig_params['L']
        i_xx, i_yy = (1+set_I[:2]) * 1.4e-5
        i_zz = min((1+set_I[2]) * 2.17e-5,i_xx+i_yy) # Inertia property
        T = (1 + set_T)
        
        if self.norm_range != 0:
            norm_mass = 2*(mass/self.orig_params['M']-(1-self.norm_range))/(2*self.norm_range)-1
            norm_xcm = 2*(x_cm/self.orig_params['L']+self.norm_range)/(2*self.norm_range)-1
            norm_ycm = 2*(y_cm/self.orig_params['L']+self.norm_range)/(2*self.norm_range)-1
            norm_ixx = 2*((i_xx/1.4e-5)-(1-self.norm_range))/(2*self.norm_range)-1
            norm_iyy = 2*((i_yy/1.4e-5)-(1-self.norm_range))/(2*self.norm_range)-1
            norm_izz = 2*((i_zz/2.17e-5)-(1-self.norm_range))/(2*self.norm_range)-1
            norm_T = 2*(T-(1-self.norm_range))/(2*self.norm_range)-1
        

        generate_urdf(self.new_URDF, mass, x_cm, y_cm, i_xx, i_yy, i_zz)
        self.M, \
        self.L, \
        self.THRUST2WEIGHT_RATIO, \
        self.J, \
        self.J_INV, \
        self.KF, \
        self.KM, \
        self.COLLISION_H,\
        self.COLLISION_R, \
        self.COLLISION_Z_OFFSET, \
        self.MAX_SPEED_KMH, \
        self.GND_EFF_COEFF, \
        self.PROP_RADIUS, \
        self.DRAG_COEFF, \
        self.DW_COEFF_1, \
        self.DW_COEFF_2, \
        self.DW_COEFF_3 = self._parseURDFParameters()

        self.M = mass
        self.T = self.orig_params['T'] * T
        self.KF = (1+set_KF) * self.orig_params['KF']
        self.KM = (1+set_KM) * self.orig_params['KM']
        if self.norm_range != 0:
            norm_KF = 2*(self.KF/self.orig_params['KF']-(1-self.norm_range))/(2*self.norm_range)-1
            norm_KM = 2*(self.KM/self.orig_params['KM']-(1-self.norm_range))/(2*self.norm_range)-1
        #### Compute constants #####################################
        self.GRAVITY = self.G*self.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / np.sum(self.KF))
        self.MAX_THRUST = (np.sum(self.KF)*self.MAX_RPM**2)
        self.MAX_XY_TORQUE = (2*self.L*np.mean(self.KF)*self.MAX_RPM**2)/np.sqrt(2)
        self.MAX_Z_TORQUE = (2*np.mean(self.KM)*self.MAX_RPM**2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * np.mean(self.KF) * self.GND_EFF_COEFF) / self.MAX_THRUST)

        self.mass = mass
        self.com = [x_cm, y_cm]
        self.kf = self.KF
        self.km = self.KM

        # return np.array([mass, x_cm, y_cm, self.battery, *self.env.KF, *self.env.KM])
        # param_num = 15
        return np.array([norm_mass, norm_xcm, norm_ycm, norm_ixx, norm_iyy, norm_izz, norm_T, *norm_KF, *norm_KM])

    def random_urdf(self):
        self.new_URDF = self.maketime + "/cf2x.urdf"
        norm_mass, norm_xcm, norm_ycm, norm_ixx, norm_iyy, norm_izz, norm_T = 0,0,0,0,0,0,0
        norm_KF, norm_KM = [0,0,0,0], [0,0,0,0]

        mass = np.random.uniform(1-self.mass_range, 1+self.mass_range) * self.orig_params['M']
        x_cm, y_cm = np.random.uniform(-self.cm_range, self.cm_range, size=(2,)) * self.orig_params['L']
        i_xx, i_yy, i_zz = np.random.uniform(1-self.i_range, 1+self.i_range, size=(3,))
        i_xx, i_yy = 1.4e-5 * i_xx, 1.4e-5 * i_yy
        i_zz = min(2.17e-5 * i_zz,i_xx+i_yy) # Inertia property
        T = np.random.uniform(1-self.t_range, 1+self.t_range)
        
        if self.mass_range != 0:
            norm_mass = 2*(mass/self.orig_params['M']-(1-self.mass_range))/(2*self.norm_range)-1
        if self.cm_range != 0:
            norm_xcm = 2*(x_cm/self.orig_params['L']+self.cm_range)/(2*self.norm_range)-1
            norm_ycm = 2*(y_cm/self.orig_params['L']+self.cm_range)/(2*self.norm_range)-1
        if self.i_range != 0:
            norm_ixx = 2*((i_xx/1.4e-5)-(1-self.i_range))/(2*self.norm_range)-1
            norm_iyy = 2*((i_yy/1.4e-5)-(1-self.i_range))/(2*self.norm_range)-1
            norm_izz = 2*((i_zz/2.17e-5)-(1-self.i_range))/(2*self.norm_range)-1
        if self.t_range != 0:
            norm_T = 2*(T-(1-self.t_range))/(2*self.norm_range)-1
        

        generate_urdf(self.new_URDF, mass, x_cm, y_cm, i_xx, i_yy, i_zz)
        self.M, \
        self.L, \
        self.THRUST2WEIGHT_RATIO, \
        self.J, \
        self.J_INV, \
        self.KF, \
        self.KM, \
        self.COLLISION_H,\
        self.COLLISION_R, \
        self.COLLISION_Z_OFFSET, \
        self.MAX_SPEED_KMH, \
        self.GND_EFF_COEFF, \
        self.PROP_RADIUS, \
        self.DRAG_COEFF, \
        self.DW_COEFF_1, \
        self.DW_COEFF_2, \
        self.DW_COEFF_3 = self._parseURDFParameters()

        self.M = mass
        self.T = self.orig_params['T'] * T
        self.KF = self.orig_params['KF'] * np.random.uniform(1.0-self.kf_range, 1.0+self.kf_range, size=(4,))
        self.KM = self.orig_params['KM'] * np.random.uniform(1.0-self.km_range, 1.0+self.km_range, size=(4,))
        if self.kf_range != 0:
            norm_KF = 2*(self.KF/self.orig_params['KF']-(1-self.kf_range))/(2*self.norm_range)-1
        if self.km_range != 0:
            norm_KM = 2*(self.KM/self.orig_params['KM']-(1-self.km_range))/(2*self.norm_range)-1
        #### Compute constants #####################################
        self.GRAVITY = self.G*self.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / np.sum(self.KF))
        self.MAX_THRUST = (np.sum(self.KF)*self.MAX_RPM**2)
        self.MAX_XY_TORQUE = (2*self.L*np.mean(self.KF)*self.MAX_RPM**2)/np.sqrt(2)
        self.MAX_Z_TORQUE = (2*np.mean(self.KM)*self.MAX_RPM**2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * np.mean(self.KF) * self.GND_EFF_COEFF) / self.MAX_THRUST)

        self.mass = mass
        self.com = [x_cm, y_cm]
        self.kf = self.KF
        self.km = self.KM

        # return np.array([mass, x_cm, y_cm, self.battery, *self.env.KF, *self.env.KM])
        # param_num = 15
        return np.array([norm_mass, norm_xcm, norm_ycm, norm_ixx, norm_iyy, norm_izz, norm_T, *norm_KF, *norm_KM])

    def _computeObs(self):
        return self._help_computeObs(self._getDroneStateVector(0))

    def _help_computeObs(self, obs_all):
        obs_idx_dict = {
            'pos': range(0,3),
            'rel_pos': range(0,7),
            'quaternion': range(3,7),
            'rotation': range(3,7),
            'vel': range(10,13),
            'rel_vel': [10,11,12,3,4,5,6],
            'angular_vel': range(13,16),
            'rel_angular_vel': [13,14,15,3,4,5,6],
            'rpm': range(16,20),
        }
        obs = []
        for otype in self.observable:
            if otype == 'param':
                o = deepcopy(self.param)
            else:
                o = obs_all[obs_idx_dict[otype]]
            obs.append(self._normalizeState(o, otype))

        obs = np.hstack(obs).flatten()
        obs_len = obs.shape[0]

        if len(self.frame_buffer) == 0:
            self.frame_buffer = [obs for _ in range(self.frame_stack)]
        else:
            self.frame_buffer.pop(0)
            self.frame_buffer.append(obs)

        return np.hstack(self.frame_buffer).reshape((obs_len * self.frame_stack,))

    def _normalizeState(self,
                                state,
                                type
                               ):
        MAX_LIN_VEL = 3 
        MAX_XYZ = MAX_LIN_VEL * 2# * self.env.EPISODE_LEN_SEC
        MAX_RPY_RATE = 2 * np.pi # temporary

        # Noise
        POS_NOISE = 0.005
        VEL_NOISE = 0.01
        ROT_NOISE = 0.0
        ANGVEL_NOISE = [0.000175, 0.0105, 1000.] # MPU-9250 gyroscope spec / ref https://github.com/amolchanov86/quad_sim2multireal/blob/master/quad_sim/sensor_noise.py#L58
        # noise density, random walk, bias correlation_time

        norm_state = deepcopy(state)

        if type=='pos':
            norm_state = (norm_state - self.goal_pos[0,:3]) 
            if self.is_noise:
                norm_state += np.random.normal(0, 0.005, size=norm_state.shape) # add noise
            norm_state = norm_state / MAX_XYZ

        elif type=='rel_pos':
            r = R.from_quat(norm_state[-4:])
            rot = r.as_matrix()
            pos = (norm_state[:3] - self.goal_pos[0,:3]).reshape((3,1)) 
            norm_state = np.matmul(rot.transpose(),pos).reshape((3,))
            if self.is_noise:
                norm_state += np.random.normal(0, POS_NOISE, size=norm_state.shape) # add noise
            norm_state = norm_state / MAX_XYZ

        elif type=='rotation':
            # don't need normalization
            r = R.from_quat(norm_state)
            norm_state = r.as_matrix().reshape((9,))
            
        elif type=='vel':
            if self.is_noise:
                norm_state += np.random.normal(0, VEL_NOISE, size=norm_state.shape) # add noise
            norm_state = norm_state / MAX_LIN_VEL

        elif type=='rel_vel':
            r = R.from_quat(norm_state[-4:])
            rot = r.as_matrix()
            norm_state = np.matmul(rot.transpose(),norm_state[:3, None]).reshape((3,)) 
            if self.is_noise:
                norm_state += np.random.normal(0, VEL_NOISE, size=norm_state.shape) # add noise
            norm_state = norm_state / MAX_LIN_VEL

        elif type=='angular_vel':
            if self.is_noise:
                norm_state = self._angvel_noise(norm_state, ANGVEL_NOISE)
            norm_state = norm_state / MAX_RPY_RATE
            
        elif type=='rel_angular_vel':
            r = R.from_quat(norm_state[-4:])
            rot = r.as_matrix()
            norm_state = np.matmul(rot.transpose(),norm_state[:3, None]).reshape((3,))
            if self.is_noise:
                norm_state = self._angvel_noise(norm_state, ANGVEL_NOISE)
            norm_state = norm_state / MAX_RPY_RATE
            # norm_state = np.deg2rad(np.matmul(rot.transpose(),norm_state[:3, None]).reshape((3,))) / MAX_RPY_RATE
        
        elif type=='rel_angular_vel_nonoise':
            r = R.from_quat(norm_state[-4:])
            rot = r.as_matrix()
            norm_state = np.matmul(rot.transpose(),norm_state[:3, None]).reshape((3,))
            # norm_state = np.deg2rad(np.matmul(rot.transpose(),norm_state[:3, None]).reshape((3,))) / MAX_RPY_RATE

        elif type=='rpm':
            # range : -1 ~ 1
            norm_state = state * 2 / self.MAX_RPM - 1
        
        elif type=='action':
            # range : 0 ~ 1
            norm_state = state / self.MAX_RPM

        elif type=='param':
            if self.is_noise:
                norm_state = norm_state + np.random.normal(0,0.005,norm_state.shape)
            pass

        return norm_state

    def _angvel_noise(self, angvel, ANGVEL_NOISE, update=True):
        dt = 1 / (self.SIM_FREQ/self.AGGR_PHY_STEPS)
        sigma_g_d = ANGVEL_NOISE[0] / (dt**0.5)
        sigma_b_g_d = (-(sigma_g_d**2) * (ANGVEL_NOISE[2] / 2) * (np.exp(-2*dt/ANGVEL_NOISE[2]) - 1))**0.5
        pi_g_d = np.exp(-dt / ANGVEL_NOISE[2])

        if update:
            self.angvel_bias = pi_g_d * self.angvel_bias + sigma_b_g_d * np.random.normal(0, 1, 3)
        
        return angvel + self.angvel_bias + ANGVEL_NOISE[1] * np.random.normal(0, 1, 3) # + self.gyro_turn_on_bias_sigma * normal(0, 1, 3)

    def _preprocessAction(self,
                          action
                          ):
        return np.array(self.MAX_RPM * (1+action) / 2)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        coeff = {
            'pos': self.reward_coeff['pos'], # 0~3
            'vel': self.reward_coeff['vel'], # 10~13
            'ang_vel': self.reward_coeff['ang_vel'], # 13~16
            'ang_vel_xy': self.reward_coeff['ang_vel_xy'],
            'ang_vel_z': self.reward_coeff['ang_vel_z'],
            'd_action': self.reward_coeff['d_action'], # 16~20
            'rotation': self.reward_coeff['rotation']
        }
        xyz = coeff['pos'] * np.linalg.norm(state[:3]-self.goal_pos[0,:3], ord=2) # for single agent temporarily
        vel = coeff['vel'] * np.linalg.norm(state[10:13],ord=2)
        ang_vel = coeff['ang_vel'] * np.linalg.norm(state[13:16],ord=2)
        rel_angvel = self._normalizeState(state[[13,14,15,3,4,5,6]],'rel_angular_vel_nonoise')
        ang_vel_xy = coeff['ang_vel_xy'] * np.linalg.norm(rel_angvel[:2])
        ang_vel_z = coeff['ang_vel_z'] * np.linalg.norm(rel_angvel[-1])
        
        rot = coeff['rotation'] * self._normalizeState(state[3:7],'rotation')[-1]
        f_s = xyz + vel + ang_vel + ang_vel_xy + ang_vel_z - rot

        d_action = coeff['d_action'] * np.linalg.norm(self._normalizeState(state[16:],'action'),ord=2)
        f_a = d_action
            
        return -(f_s+f_a) * (1/(self.SIM_FREQ/self.AGGR_PHY_STEPS)) # + done_reward
    
    def _computeDone(self):
        if (self.step_counter+self.AGGR_PHY_STEPS)/(self.SIM_FREQ) >= self.EPISODE_LEN_SEC:
            return True
        # elif np.linalg.norm(self._getDroneStateVector(0)[:3]-self.goal_pos[0,:3], ord=2) > 6:
        #     return True
        else:
            return False

    def _computeInfo(self):
        return {}

    def reset(self, set_dyn=None):
        if set_dyn is None:
            self.param = self.random_urdf()
        else:
            assert isinstance(set_dyn, dict), "set_dyn should be dictionary type"
            self.param = self.set_urdf(set_dyn)
        return super().reset()

    def step(self, action):
        if self.is_noise:
            # motor lag
            action = 4*(self.AGGR_PHY_STEPS/self.SIM_FREQ)/self.T * (action-self.last_action_custom) + self.last_action_custom
        self.last_action_custom = action
        if self.is_noise:
            action += np.random.normal(0, 0.005, action.shape)
        state, reward, done, _ = super().step(action)
        
        self.droneStates.append(self._getDroneStateVector(0))
        if len(self.droneStates) > 100:
            self.droneStates.pop(0)
        droneState = np.stack(self.droneStates).mean(axis=0)
        info = {'x': droneState[0],
                'y': droneState[1],
                'z': droneState[2]-self.goal_pos[0,2],
                'roll':droneState[7],
                'pitch':droneState[8],
                'yaw':droneState[9],
                'vx':droneState[10],
                'vy':droneState[11],
                'vz':droneState[12],
                'wx':droneState[13],
                'wy':droneState[14],
                'wz':droneState[15]}
        droneState = np.stack(self.droneStates).var(axis=0)
        info.update({'x_var': droneState[0],
                'y_var': droneState[1],
                'z_var': droneState[2],
                'roll_var':droneState[7],
                'pitch_var':droneState[8],
                'yaw_var':droneState[9],
                'vx_var':droneState[10],
                'vy_var':droneState[11],
                'vz_var':droneState[12],
                'wx_var':droneState[13],
                'wy_var':droneState[14],
                'wz_var':droneState[15],
                'param': self.param})
        return state, reward, done, info

    def render(self,
               mode='rgb_array',
               close=False
               ):
        if not self.wandb_render:
            return super().render('human', close)
        else:
            # rgb array return 
            # X,Y,Z,Roll,Pitch,Yaw,Vx,Vy,Vz,Wx,Wy,Wz,a1,a2,a3,a4
            state = self._getDroneStateVector(0)
            xyz = 100-np.clip(((state[:3] - self.goal_pos[0,:3]) / 10 + 1) * 50, 0,100)  # -10~10 m
            rpy = 100-np.clip((state[7:10] / np.pi + 1) * 50, 0, 100)   # -pi~pi rad
            vs = 100-np.clip((state[10:13] / 3 + 1) * 50, 0, 100) # -3~3 m/s
            ws = 100-np.clip((state[13:16] / 3 * np.pi + 1) * 50, 0, 100) # -3pi~3pi rad/s
            aes = 100-np.clip((state[16:20] / self.MAX_RPM) * 100, 0, 100) # -1~1

            img = np.zeros((101,60,3), dtype=np.uint8)
            img[49:52,:,:] = 255

            img[int(max(xyz[0]-3,0)):int(min(xyz[0]+4,100)),0:3,0] = 255
            img[int(max(xyz[1]-3,0)):int(min(xyz[1]+4,100)),3:6,1] = 255
            img[int(max(xyz[2]-3,0)):int(min(xyz[2]+4,100)),6:9,2] = 255

            img[:,10,:] = 255

            img[int(max(rpy[0]-3,0)):int(min(rpy[0]+4,100)),12:15,0] = 255
            img[int(max(rpy[1]-3,0)):int(min(rpy[1]+4,100)),15:18,1] = 255
            img[int(max(rpy[2]-3,0)):int(min(rpy[2]+4,100)),18:21,2] = 255

            img[:,22,:] = 255

            img[int(max(vs[0]-3,0)):int(min(vs[0]+4,100)),24:27,0] = 255
            img[int(max(vs[1]-3,0)):int(min(vs[1]+4,100)),27:30,1] = 255
            img[int(max(vs[2]-3,0)):int(min(vs[2]+4,100)),30:33,2] = 255

            img[:,34,:] = 255

            img[int(max(ws[0]-3,0)):int(min(ws[0]+4,100)),36:39,0] = 255
            img[int(max(ws[1]-3,0)):int(min(ws[1]+4,100)),39:42,1] = 255
            img[int(max(ws[2]-3,0)):int(min(ws[2]+4,100)),42:45,2] = 255

            img[:,46,:] = 255

            img[int(max(aes[0]-3,0)):int(min(aes[0]+4,100)),48:51,0] = 255
            img[int(max(aes[1]-3,0)):int(min(aes[1]+4,100)),51:54,1] = 255
            img[int(max(aes[2]-3,0)):int(min(aes[2]+4,100)),54:57,2] = 255
            img[int(max(aes[3]-3,0)):int(min(aes[3]+4,100)),57:60,:2] = 255

            return img