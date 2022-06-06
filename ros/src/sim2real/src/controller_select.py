#!/home/jack/anaconda3/envs/crazyflie python

import time

import rospy
from std_msgs.msg import String

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper


#########################################
### Crazyflie Functions #################
#########################################

CF = None
pub = rospy.Publisher('controller', String, queue_size=100)

def cf_init(address):
    uri = uri_helper.uri_from_env(default=address)
    cflib.crtp.init_drivers()
    cf = Crazyflie(rw_cache='./cache')
    cf.open_link(uri)

    return cf

def pub_controller(name, value):
    result = name + ": " + value
    pub.publish(result)

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
    CF.param.add_update_callback('stabilizer', 'controller', cb = pub_controller)
    time.sleep(1)


def main():
    global CF
    rospy.init_node('sim2real_agent', anonymous=True)
    uri = rospy.get_param("uri", 'radio://0/100/2M/E7E7E7E701')
    CF = cf_init(uri)
    rospy.Subscriber("command", String, commandCallback)
    rospy.spin()
        

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        time.sleep(1)
        CF.close_link()