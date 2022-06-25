#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

def main():
    rospy.init_node('input_test')
    pub = rospy.Publisher('command', String, queue_size=100)

    while not rospy.is_shutdown():
        command = input()
        pub.publish(command)
        rospy.sleep(0.1)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    
