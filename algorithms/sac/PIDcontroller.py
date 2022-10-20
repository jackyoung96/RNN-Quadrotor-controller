import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
import math

from gym_pybullet_drones.envs.BaseAviary import DroneModel
from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl

class PIDcontroller:
    def __init__(self, env):
        self.drone_model=DroneModel.CF2X
        self.ctrl = None
        self.goal_pos = env.goal_pos.copy()
        self.timestep = env.TIMESTEP

    def reset(self, env):
        self.ctrl.reset()
        self.goal_pos = env.goal_pos.copy()

    def get_action(self, state):
        action, *_ = self.ctrl.computeControlFromState(control_timestep=self.timestep,
                                                    state=state,
                                                    target_pos=self.goal_pos.reshape(3,),
                                                    )
        action = 2*(action/24000)-1
        
        return action
    
    def load_model(self, *args):
        pass

class SimplePIDcontroller(PIDcontroller):
    def __init__(self, env):
        super(SimplePIDcontroller, self).__init__(env)
        self.ctrl = SimplePIDControl(drone_model=self.drone_model)


class DSLPIDcontroller(PIDcontroller):
    def __init__(self, env):
        super(DSLPIDcontroller, self).__init__(env)
        self.ctrl = DSLPIDControl(drone_model=self.drone_model)
        self.target_rpy = None

    def reset(self, env):
        self.ctrl.reset()
        self.goal_pos = env.goal_pos.copy()
        self.target_rpy = None

    def get_action(self, state):
        if self.target_rpy is None:
            self.target_rpy = np.array([0,0,state[5]+state[15]]) # yaw + yaw_rate * 1 second
        action, *_ = self.ctrl.computeControlFromState(control_timestep=self.timestep,
                                                    state=state,
                                                    target_pos=self.goal_pos.reshape(3,),
                                                    target_rpy=self.target_rpy
                                                    )
        action = 2*(action/24000)-1
        
        return action