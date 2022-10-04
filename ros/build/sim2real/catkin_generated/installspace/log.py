#!/home/jack/anaconda3/envs/crazyflie/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import TransformStamped
from collections import deque
from crazyflie_driver.msg import Position, GenericLogData

import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

###### GLOBAL VARIABLES #########################

log = pd.DataFrame(columns=['type','timestep','x','y','z','w'])
start = None
##################################################

def EKFCallback(data):
    global log, start
    if start is None:
        start = data.header.stamp.secs + data.header.stamp.nsecs * 1e-9
    t = data.header.stamp.secs + data.header.stamp.nsecs * 1e-9

    log = log.append({'type':"EKF", 
                        "timestep": t - start,
                        "x":data.values[0],
                        "y":data.values[1],
                        "z":data.values[2],
                        "w":data.values[3]}, ignore_index=True)

def viconCallback(data):
    global log, start
    if start is None:
        start = data.header.stamp.secs + data.header.stamp.nsecs * 1e-9
    t = data.header.stamp.secs + data.header.stamp.nsecs * 1e-9

    pos = data.transform.translation
    rot = data.transform.rotation
    r = R.from_quat(np.array([rot.x, rot.y, rot.z, rot.w]))
    yaw = r.as_euler('zyx', degrees=True)[0]
    
    log = log.append({'type':"EKF", 
                        "timestep": t - start,
                        "x":pos.x,
                        "y":pos.y,
                        "z":pos.z,
                        "w":yaw},ignore_index=True)

def main():
    rospy.init_node('logging', anonymous=True)
    rospy.Subscriber("log1", GenericLogData, EKFCallback)
    rospy.Subscriber("/vicon/CF_JACK/CF_JACK", TransformStamped, viconCallback)
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
        log.to_csv('~/Downloads/log.csv', index=False)
    except rospy.ROSInterruptException:
        log.to_csv('~/Downloads/log.csv', index=False)
        pass

    

    
