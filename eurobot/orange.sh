#!/bin/bash

source /opt/ros/kinetic/setup.bash
source ~/catkin_ws/devel/setup.bash

roslaunch eurobot start.launch color:="orange" > ~/some_log.log&