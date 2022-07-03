import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper
import time
import numpy as np

from cflib.utils.callbacks import Caller
from cflib.crazyflie.log import Log, LogConfig, LogTocElement
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

def cflib_init(address):
    cflib.crtp.init_drivers()
    print('Scanning interfaces for Crazyflies...')
    available = cflib.crtp.scan_interfaces(address)
    print('Crazyflies found:')
    for i in available:
        print(i[0])

    cf = Crazyflie(rw_cache='./cache')
    cf.open_link(available[0][0])

    # cf.param.set_value("motorPowerSet.enable", 1)

    return cf, available[0][0]

def cflib_close(cf):
    time.sleep(1)
    cf.close_link()

def motorRun(cf, thrust):
    maxrpm = 2**16-1
    thrust = maxrpm * (thrust + 1)/2
    print(thrust)
    for i in range(4):
        cf.param.set_value("motorPowerSet.m%d"%(i+1), int(thrust[i]))

#######################################
#### Callback Functions ###############
#######################################




#########################################
##### Main ##############################
#########################################

def main():
    address = uri_helper.address_from_env(default=0xE7E7E7E701)
    # uri = uri_helper.uri_from_env(default='radio://0/100/2M/E7E7E7E7E7')
    # CF = Crazyflie(rw_cache='./cache')
    cf, uri = cflib_init(address)
    # try:
    cf.param.set_value("motorPowerSet.enable", 1)
    start = time.time()
    for i in range(1000):
        rpm = 0.8*i/500-1
        motorRun(cf, np.array([rpm]*4))
        time.sleep(0.005)
    # for i in range(1000):
    #     rpm = 0.8*(1000-i)/500-1
    #     motorRun(scf.cf, np.array([rpm]*4))
    #     time.sleep(0.005)
    time.sleep(1)
    motorRun(cf, np.array([-1]*4))
    time.sleep(1)
    cflib_close(cf)
    # except KeyboardInterrupt:
    #     motorRun(CF, np.array([-1,-1,-1,-1]))
    #     time.sleep(1)
    #     cflib_close(CF)

if __name__ == "__main__":
    main()