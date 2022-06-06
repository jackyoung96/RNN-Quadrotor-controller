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


#########################################
### Crazyflie Functions #################
#########################################

CF = None

def cf_init(address):
    address = uri_helper.address_from_env(default=0xE7E7E7E701)
    cflib.crtp.init_drivers()
    print('Scanning interfaces for Crazyflies...')
    available = cflib.crtp.scan_interfaces(address)
    print('Crazyflies found:')
    for i in available:
        print(i[0])

    cf = Crazyflie(rw_cache='./cache')
    cf.open_link(available[0][0])

    return cf

def commandCallback(msg):
    global CF
    
    if msg.data == 'any':
        CF.param.set_value('stabilizer.controller', 0)
    elif msg.data == 'pid':
        CF.param.set_value('stabilizer.controller', 1)
    elif msg.data == 'mellinger':
        CF.param.set_value('stabilizer.controller', 2)
    elif msg.data == 'indi':
        CF.param.set_value('stabilizer.controller', 3)
    elif msg.data == 'nn':
        CF.param.set_value('stabilizer.controller', 4)
    time.sleep(1)


def main():
    rospy.init_node('sim2real_agent', anonymous=True)
    uri = rospy.get_param("uri", 'radio://0/100/2M/E7E7E7E701')
    cf = cf_init(uri)
    rospy.Subscriber("/command", String, commandCallback)
    rospy.spin()
        

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        time.sleep(1)
        cf.close_link()