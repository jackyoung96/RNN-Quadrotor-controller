#include "ros/ros.h"
#include "std_msgs/Float32MultiArray.h"
#include "geometry_msgs/TransformStamped.h"

ros::Publisher state_pub = n.advertise<std_msgs::Float32MultiArray>("crazyflie_state", 1000);

void stateCallback(const geometry_msgs::TransformStamped::ConstPtr& msg){
    
}

int main(int argc, char **argv)
{

  ros::init(argc, argv, "state_publisher");
  ros::NodeHandle nh;
  std::string sub_topic;
  nh.getParam("sub_topic", sub_topic)
  ros::Subscriber sub = nh.subscribe(sub_topic, 1000, stateCallback);

  ros::spin();
}