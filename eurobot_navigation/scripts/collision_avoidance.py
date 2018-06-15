#!/usr/bin/env python
import rospy
import numpy as np
# import math
# import tf2_ros
# from tf.transformations import euler_from_quaternion
# from geometry_msgs.msg import Twist, Point, Polygon
# from std_msgs.msg import String
# from threading import Lock
from std_msgs.msg import Int8
from sensor_msgs.msg import LaserScan


class CollisionAvoidance:
    def __init__(self):
        rospy.init_node("collision_avoidance", anonymous=True)
        self.NUM_RANGEFINDERS = rospy.get_param("motion_planner/NUM_RANGEFINDERS")
        self.COLLISION_STOP_DISTANCE = rospy.get_param("motion_planner/COLLISION_STOP_DISTANCE")
        self.COLLISION_STOP_NEIGHBOUR_DISTANCE = rospy.get_param("motion_planner/COLLISION_STOP_NEIGHBOUR_DISTANCE")
        self.COLLISION_GAMMA = rospy.get_param("motion_planner/COLLISION_GAMMA")
        self.LIDAR_C_A = rospy.get_param("motion_planner/LIDAR_COLLISION_STOP_DISTANCE")

        self.scan = None
        self.ranges = None
        self.angles = None
        self.indexes = None
        self.scan_mid_ind = None
        rospy.Subscriber("scan", LaserScan, self.scan_callback, queue_size=1)
        self.collision_pub = rospy.Publisher("collision_avoider_activated", Int8, queue_size=1)

    def scan_callback(self, scan):
        self.scan = np.array([np.array(scan.ranges) * 1000, scan.intensities]).T
        # if not self.ranges:
        self.ranges = np.array(scan.ranges) * 1000
        # self.angles = np.linspace(scan.angle_min, scan.angle_max, self.ranges.shape[0])
        self.angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
        self.indexes = np.arange(self.ranges.shape[0])
        self.scan_mid_ind = self.indexes.shape[0] // 2
        # self.ranges = scan.ranges
        self.ranges[self.ranges < scan.range_min * 1000] = 0
        self.ranges[self.ranges > scan.range_max * 1000] = 0
        collision_pnts = self.ranges[0 < self.ranges][self.ranges[0 < self.ranges] < self.LIDAR_C_A]
        if collision_pnts.shape[0] > 0:
            collision_angs = self.angles[0 < self.ranges][self.ranges[0 < self.ranges] < self.LIDAR_C_A]
            collision_inds = self.indexes[0 < self.ranges][self.ranges[0 < self.ranges] < self.LIDAR_C_A]
            side = collision_inds >= self.scan_mid_ind
            avoid_type = 0
            if side.all():
                self.move_diag_left()
                avoid_type = 1
            elif not side.any():
                self.move_diag_right()
                avoid_type = 2

            else:
                self.three_pnt()
            self.collision_pub.publish(avoid_type)
        else:
            self.collision_pub.publish(-1)

    def move_diag_left(self):
        pass

    def move_diag_right(self):
        pass

    def move_left(self):
        pass

    def move_right(self):
        pass

    def three_pnt(self):
        pass


if __name__ == "__main__":
    planner = CollisionAvoidance()
rospy.spin()
