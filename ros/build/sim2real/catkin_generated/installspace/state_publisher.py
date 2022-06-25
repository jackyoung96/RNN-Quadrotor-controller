#!/home/jack/anaconda3/envs/crazyflie/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import TransformStamped
from collections import deque

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

pub = rospy.Publisher('crazyflie_state', Float32MultiArray, queue_size=100)
GOAL = None
STATE = None
##################################################

def angular_velocity(R, dt):
    R0, R1 = R
    A = np.matmul(R1, R0.transpose())
    theta = np.arccos((np.trace(A)-1)/2)
    W = 1/(2*(dt)) * (theta/np.sin(theta)) * (A-A.transpose())
    return np.array([W[2,1], W[0,2], W[1,0]]) 

def stateCallback(data):
    global POS_BUFFER, ROT_BUFFER, QUAT_BUFFER, VEL_BUFFER, ANGVEL_BUFFER, TIME_BUFFER, ACTION_BUFFER, GOAL, STATE

    t = data.header.stamp
    t = t.secs+t.nsecs*1e-9
    pos = data.transform.translation
    # position (x,y,z)
    pos = np.array([pos.x,pos.y,pos.z])

    # velocity (x,y,z)
    vel = np.zeros((3,)) if len(POS_BUFFER)==0\
            else (pos-POS_BUFFER[-1])/(t-TIME_BUFFER[-1])

    rot = data.transform.rotation
    # quaternion 
    quat = np.array([rot.x,rot.y,rot.z,rot.w])
    r = R.from_quat(quat)

    # rotation matrix
    rot_matrix = r.as_dcm()

    # angular velocity
    ang_vel = np.zeros((3,))
    if len(ROT_BUFFER) != 0:
        dt = t - TIME_BUFFER[-1]
        dquat = [ROT_BUFFER[-1],rot_matrix]
        ang_vel = angular_velocity(dquat,dt)

    STATE = pos

    state = Float32MultiArray()

    if not GOAL is None:
        # Normalize
        state.data = np.concatenate([(pos-GOAL)/6,
                                    rot_matrix.flatten(),
                                    vel/3,
                                    ang_vel/(2*np.pi),
                                    np.array([1])])
        pub.publish(state)
    else:
        # Normalize
        state.data = np.zeros(19)
        pub.publish(state)

def commandCallback(msg):
    global GOAL, STATE
    if msg.data == 'takeoff':
        GOAL = STATE[:3] + np.array([0,0,0.8])
    elif msg.data == 'left':
        GOAL = GOAL + np.array([0,0.3,0])
    elif msg.data == 'right':
        GOAL = GOAL + np.array([0,-0.3,0])
    elif msg.data == 'forward':
        GOAL = GOAL + np.array([0.3,0,0])
    elif msg.data == 'backward':
        GOAL = GOAL + np.array([-0.3,0,0])
    elif msg.data == 'up':
        GOAL = GOAL + np.array([0,0,0.3])
    elif msg.data == 'down':
        GOAL = GOAL + np.array([0,0,-0.3])
    elif msg.data == 'landing':
        GOAL[2] = 0.1
    elif msg.data == 'turnoff':
        GOAL = None

def main():
    rospy.init_node('state_publisher', anonymous=True)
    rospy.Subscriber("/vicon/CF_JACK/CF_JACK", TransformStamped, stateCallback)
    rospy.Subscriber("/command", String, commandCallback)
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

    

    
