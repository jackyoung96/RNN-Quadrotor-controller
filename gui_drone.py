from email.policy import Policy
import gym
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
from envs.customEnvDrone import customAviary
import numpy as np
import torch.nn.functional as F
import torch

from td3.common.policy_networks import PolicyNetworkGoalRNN

def main():
    env = gym.make('takeoff-aviary-v0', # arbitrary environment that has state normalization and clipping
        drone_model=DroneModel.CF2X,
        initial_xyzs=np.array([[0.0,0.0,0.0]]),
        # initial_xyzs=np.array([[0.5,0.3,0.425]]),
        initial_rpys=np.array([[0,0,np.pi/4]]),
        # initial_rpys=np.array([[np.pi/6,np.pi/6,np.pi/4]]),
        physics=Physics.PYB_GND_DRAG_DW,
        freq=200,
        aggregate_phy_steps=1,
        gui=True,
        record=False, 
        obs=ObservationType.KIN,
        act=ActionType.RPM)
    env = customAviary(env,
        initial_xyzs=np.array([[0.0,0.0,0.0]]),
        # initial_xyzs=np.array([[0.5,0.3,0.425]]),
        initial_rpys=np.array([[0,0,np.pi/4]]),
        # initial_rpys=np.array([[np.pi/6,np.pi/6,np.pi/4]]),
        observable=['pos','rotation','vel','angular_vel','rpm'],
        task='takeoff',
        rpy_noise=0,
        vel_noise=0,
        angvel_noise=0
    )
    
    state = env.reset()
    state = state[None,:]
    state_space = gym.spaces.Box(low=-np.inf,high=np.inf,shape=(22,))
    action_space = gym.spaces.Box(low=-1,high=1,shape=(4,))
    hidden_dim=32
    goal_dim=18
    device='cpu'
    policy = PolicyNetworkGoalRNN(state_space, action_space, hidden_dim, goal_dim, device, actf=F.tanh ,out_actf=F.tanh, action_scale=1.0)
    policy.load_state_dict(torch.load('artifacts/agent-22Jun05162636:v19/iter0100000_policy.pt', map_location='cpu'))
    # policy.load_state_dict(torch.load('artifacts/agent-22Jun10032015:v19/iter0100000_policy.pt', map_location='cpu'))
    env.goal_pos = np.array([[0.0,0.0,0.5]])

    hidden = torch.zeros((1,1,32))
    action = - np.ones((1,4))
    theta = np.pi/4
    goal = np.array([[0,0,0,
                        np.cos(theta),-np.sin(theta),0,
                        np.sin(theta),np.cos(theta),0,
                        0,0,1,
                        0,0,0,0,0,0]])

    # state = np.array([
    #     0.000817157735582441, 0.004964136518537998, 0.003833387978374958,
    #     0.997377157211303710, 0.072014190256595611, 0, 
    #     -0.07195847481489181, 0.997378170490264892, 0,
    #     0, 0, 0.999944269657135009,
    #     -0.00832621473819017, -0.02109748870134353, 0.000107035855762660,
    #     -0.00005220051752985, -0.00050417712191119, 0.000345550128258764,
    #     -1,-1,-1,-1
    # ])[None,:]

    count = 0

    while True:
        input()
        count += 1
        # if count % 100 == 0:
        #     hidden = torch.zeros((1,1,32))
        action,hidden = policy.get_action(state,action,hidden,goal,True,0.0)
        state,*_ = env.step(action)
        
        print("pos", state[:3])
        print("rotation")
        print(state[3:6])
        print(state[6:9])
        print(state[9:12])
        print("vel", state[12:15])
        print("angvel", state[15:18])
        print("action", action)
        

        # print("rotation", state[3:12])
        # print('vel',state[12:15])
        # print('ang vel', state[15:])

        # state[:,-4:] = action
        action = action[None,:]
        state = state[None,:]
        # state[:,-4:] = 0
    

if __name__ == '__main__':
    main()