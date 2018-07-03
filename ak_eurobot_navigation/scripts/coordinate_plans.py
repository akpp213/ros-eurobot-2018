#!/usr/bin/env python
import rospy
import numpy as np
# import math
# import tf2_ros
# from tf.transformations import euler_from_quaternion
# from geometry_msgs.msg import Twist, Point, Polygon
# from std_msgs.msg import String
# from threading import Lock
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan


class CoordinatePlans:
    def __init__(self):
        rospy.init_node("coordinate_plans", anonymous=True)
        # self.NUM_RANGEFINDERS = rospy.get_param("motion_planner/NUM_RANGEFINDERS")
        # self.COLLISION_STOP_DISTANCE = rospy.get_param("motion_planner/COLLISION_STOP_DISTANCE")
        # self.COLLISION_STOP_NEIGHBOUR_DISTANCE = rospy.get_param("motion_planner/COLLISION_STOP_NEIGHBOUR_DISTANCE")
        # self.COLLISION_GAMMA = rospy.get_param("motion_planner/COLLISION_GAMMA")
        # self.LIDAR_C_A = rospy.get_param("motion_planner/LIDAR_COLLISION_STOP_DISTANCE")

        self.planning_main = False
        self.main_plan = None
        self.planning_secondary = False
        self.secondary_plan = None

        rospy.Subscriber("/main_robot/move_command", String, self.main_callback, queue_size=1)
        rospy.Subscriber("/secondary_robot/move_command", String, self.secondary_callback, queue_size=1)
        # self.collision_pub = rospy.Publisher("collision_avoider_activated", Int8, queue_size=1)

    def main_callback(self, cmd):
        self.planning_main = True
        if self.planning_secondary:
            plan_coordinated = True

    def secondary_callback(self, cmd):
        self.planning_secondary = True
        if self.planning_main:
            plan_coordinated = True


if __name__ == "__main__":
    plan_coordinator = CoordinatePlans()
rospy.spin()
