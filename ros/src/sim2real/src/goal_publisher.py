#!/home/jack/anaconda3/envs/crazyflie/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import TransformStamped
from collections import deque
from crazyflie_driver.msg import Position, GenericLogData

import numpy as np
from scipy.spatial.transform import Rotation as R

###### GLOBAL VARIABLES #########################
maxlen = 100
POS_BUFFER = deque(maxlen=maxlen)
ROT_BUFFER = deque(maxlen=maxlen)
QUAT_BUFFER = deque(maxlen=maxlen)
VEL_BUFFER = deque(maxlen=maxlen)
ANGVEL_BUFFER = deque(maxlen=maxlen)
TIME_BUFFER = deque(maxlen=maxlen)
ACTION_BUFFER = deque(maxlen=maxlen)

pub_pos = rospy.Publisher('cmd_position', Position, queue_size=100)

GOAL = None
STATE = None
##################################################

def stateCallback(data):
    global STATE

    STATE = np.array(data.values)

def commandCallback(msg):
    global GOAL, STATE

    if msg.data in ['takeoff', 'left', 'right', 'forward', 'backward', 'up', 'down', 'landing', 'turnoff']:
        if msg.data == 'takeoff':
            GOAL = STATE[:3] + np.array([0,0,0.2])
        elif msg.data == 'left':
            GOAL = GOAL + np.array([0,0.1,0])
        elif msg.data == 'right':
            GOAL = GOAL + np.array([0,-0.1,0])
        elif msg.data == 'forward':
            GOAL = GOAL + np.array([0.1,0,0])
        elif msg.data == 'backward':
            GOAL = GOAL + np.array([-0.1,0,0])
        elif msg.data == 'up':
            GOAL = GOAL + np.array([0,0,0.1])
        elif msg.data == 'down':
            GOAL = GOAL + np.array([0,0,-0.1])
        elif msg.data == 'landing':
            GOAL[2] = 0.1
        elif msg.data == 'turnoff':
            GOAL = None
        
        goal = Position()
        goal.x = GOAL[0]
        goal.y = GOAL[1]
        goal.z = GOAL[2]
        goal.yaw = STATE[3]
        pub_pos.publish(goal)

def main():
    rospy.init_node('state_publisher', anonymous=True)
    rospy.Subscriber("log1", GenericLogData, stateCallback)
    rospy.Subscriber("command", String, commandCallback)
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

    

    
