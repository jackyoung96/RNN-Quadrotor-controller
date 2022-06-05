import numpy as np
import torch
import torch.nn.functional as F

from td3.td3 import *
from td3.common.buffers import *
from td3.agent import td3_agent
from envs.customEnv import dynRandeEnv, dummyEnv

import argparse
import numpy as np
import time
import threading

import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper


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
    'battery_range': 0.3 # (1-n) ~ (1)
}
hparam_set = {
    "goal_dim": [18],
    "param_num": [14],
    "hidden_dim": [128],
    "policy_actf": [F.tanh]
}

parser = argparse.ArgumentParser()
parser.add_argument('--rnn', choices=['None','RNN2','GRU2','LSTM2',
                                        'RNNHER','GRUHER','LSTMHER',
                                        'RNNbhvHER','GRUbhvHER','LSTMbhvHER']
                            , default='None', help='Use memory network (LSTM)')

# Arguments for test
parser.add_argument('--path', type=str, default=None, help='required only at test phase')


args = parser.parse_args()
if args.path is None:
    assert "model path is required"
hparam = dict([(k,v[0]) for k,v in hparam_set.items()])
hparam.update(vars(args))

env_name = "takeoff-aviary-v0"
device=torch.device("cpu")


#########################################
### Crazyflie Functions #################
#########################################

def cflib_init(address):
    rospy.loginfo("cflib initialize")
    cflib.crtp.init_drivers()
    print('Scanning interfaces for Crazyflies...')
    available = cflib.crtp.scan_interfaces(address)
    print('Crazyflies found:')
    for i in available:
        print(i[0])

    cf = Crazyflie(rw_cache='./cache')
    cf.open_link(available[0][0])

    cf.param.set_value("motorPowerSet.enable", 1)

    return cf

def cflib_close(cf):
    time.sleep(1)
    cf.close_link()

def motorRun(cf, thrust):
    # maxrpm = 2**16-1
    maxrpm = 40000
    thrust = (maxrpm * (thrust + 1)/2).astype(int)
    print(thrust)
    for i in range(4):
        cf.param.set_value("motorPowerSet.m%d"%(i+1), int(thrust[i]))


#######################################
#### Callback Functions ###############
#######################################

# Load Agent
eval_env = dummyEnv(22, 4)
eval_env.env.training = False

# Global 
LAST_ACTION = np.zeros(4)[None,:]
ACTION = np.zeros(4)
AGENT = td3_agent(env=eval_env,
            rnn=args.rnn,
            device=device,
            hparam=hparam)
AGENT.load_model(args.path)
AGENT.policy_net.eval()

if hasattr(AGENT, 'rnn_type'):
    if 'LSTM' == AGENT.rnn_type:
        HIDDEN_IN = (torch.zeros([1, 1, hparam['hidden_dim']], dtype=torch.float).to(device), \
                    torch.zeros([1, 1, hparam['hidden_dim']], dtype=torch.float).to(device))
    else:
        HIDDEN_IN = torch.zeros([1, 1, hparam['hidden_dim']], dtype=torch.float).to(device)


address = uri_helper.address_from_env(default=0xE7E7E7E701)
CF = cflib_init(address)
print("READY to FLY!!!")

def actionCallback(msg):
    global LAST_ACTION, HIDDEN_IN, AGENT
    state = np.array(msg.data)[None,:-1]
    run = msg.data[-1]
    state = np.concatenate([state, LAST_ACTION], axis=-1)

    if run < 0.5: # == 0
        # action = np.array([-1,-1,-1,-1])
        action = np.array([-0.6]*4)
        LAST_ACTION = np.zeros(4)[None,:]
    else:
        with torch.no_grad():
            if getattr(AGENT, 'rnn_type', 'None') in ['GRU','RNN','LSTM']:
                if not hasattr(AGENT.q_net1, '_goal_dim'):
                    action, hidden_out = \
                        AGENT.policy_net.get_action(state, 
                                                        LAST_ACTION, 
                                                        HIDDEN_IN, 
                                                        deterministic=True, 
                                                        explore_noise_scale=0.0)
                else:
                    action, hidden_out = \
                        AGENT.policy_net.get_action(state, 
                                                        LAST_ACTION, 
                                                        HIDDEN_IN, 
                                                        goal=np.array([[0,0,0, # pos
                                                                        1,0,0,
                                                                        0,1,0,
                                                                        0,0,1, # rotation matrix
                                                                        0,0,0, # vel
                                                                        0,0,0, # ang vel
                                                                        0,0,0,0]]), # dummy action
                                                        deterministic=True, 
                                                        explore_noise_scale=0.0)
                HIDDEN_IN = hidden_out
            else:
                action = AGENT.policy_net.get_action(state, 
                                                    deterministic=True, 
                                                    explore_noise_scale=0.0)
                                            
        LAST_ACTION = action[None,:]
    motorRun(CF, action)
    # ACTION = action

#########################################
##### Main ##############################
#########################################


def main():
    global CF
    try:
        rospy.init_node('sim2real_agent', anonymous=True)
        rospy.Subscriber("crazyflie_state", Float32MultiArray, actionCallback)
        rospy.spin()

    except KeyboardInterrupt:
        motorRun(CF, np.array([-1,-1,-1,-1]))
        time.sleep(0.1)
        cflib_close(CF)

if __name__ == "__main__":
    main()