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
import tf2_ros
import tf_conversions


def cvt_local2global(local_point, src_point):
    x, y, a = local_point.T
    X, Y, A = src_point.T
    x1 = x * np.cos(A) - y * np.sin(A) + X
    y1 = x * np.sin(A) + y * np.cos(A) + Y
    a1 = (a + A) % (2 * np.pi)
    return np.array([x1, y1, a1]).T


def cvt_global2local(global_point, src_point):
    x1, y1, a1 = global_point.T
    X, Y, A = src_point.T
    x = x1 * np.cos(A) + y1 * np.sin(A) - X * np.cos(A) - Y * np.sin(A)
    y = -x1 * np.sin(A) + y1 * np.cos(A) + X * np.sin(A) - Y * np.cos(A)
    a = (a1 - A) % (2 * np.pi)
    return np.array([x, y, a]).T


class CollisionAvoidance:
    def __init__(self):
        rospy.init_node("collision_avoidance", anonymous=True)
        self.NUM_RANGEFINDERS = rospy.get_param("motion_planner/NUM_RANGEFINDERS")
        self.COLLISION_STOP_DISTANCE = rospy.get_param("motion_planner/COLLISION_STOP_DISTANCE")
        self.COLLISION_STOP_NEIGHBOUR_DISTANCE = rospy.get_param("motion_planner/COLLISION_STOP_NEIGHBOUR_DISTANCE")
        self.COLLISION_GAMMA = rospy.get_param("motion_planner/COLLISION_GAMMA")
        self.LIDAR_C_A = rospy.get_param("motion_planner/LIDAR_COLLISION_STOP_DISTANCE")

        self.color = rospy.get_param("/field/color")
        self.robot_name = rospy.get_param("robot_name")
        self.lidar_point = np.array([rospy.get_param("lidar_x"), rospy.get_param("lidar_y"), rospy.get_param("lidar_a")])

        self.scan = None
        self.ranges = None
        self.angles = None
        self.indexes = None
        self.scan_mid_ind = None

        # self.buffer = tf2_ros.Buffer()
        # self.listener = tf2_ros.TransformListener(self.buffer)
        # self.br = tf2_ros.TransformBroadcaster()
        #
        # rospy.sleep(1)
        #
        # t, robot_odom_point = self.get_odom()
        # while not t and not rospy.is_shutdown():
        #     t, robot_odom_point = self.get_odom()
        #     rospy.sleep(0.2)
        # lidar_odom_point = cvt_local2global(self.lidar_point, robot_odom_point)
        # self.prev_lidar_odom_point = lidar_odom_point
        # x, y, a = lidar_odom_point
        # print x,y,a

        # rospy.Subscriber("scan", LaserScan, self.scan_callback, queue_size=1)
        self.collision_pub = rospy.Publisher("collision_avoider_activated", String, queue_size=1)
        rospy.loginfo("Collision Avoidance Node Started")

    def scan_callback(self, scan):
        # print "scan callback received"
        # if not self.ranges:
        self.ranges = np.array(scan.ranges) * 1000
        # self.angles = np.linspace(scan.angle_min, scan.angle_max, self.ranges.shape[0])
        self.angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
        self.indexes = np.arange(self.ranges.shape[0])
        self.scan_mid_ind = self.indexes.shape[0] // 2
        # self.ranges = scan.ranges
        #print scan.range_min, scan.range_max
        #print self.ranges
        self.ranges[self.ranges < scan.range_min*1000] = 0
        self.ranges[self.ranges > scan.range_max*1000] = 0
        # print self.ranges
        collision_pnts = self.ranges[0 < self.ranges][self.ranges[0 < self.ranges] < self.LIDAR_C_A]
        if collision_pnts.shape[0] > 0:
            print "something in range"
            collision_angs = self.angles[0 < self.ranges][self.ranges[0 < self.ranges] < self.LIDAR_C_A]
            collision_inds = self.indexes[0 < self.ranges][self.ranges[0 < self.ranges] < self.LIDAR_C_A]
            side = collision_inds >= self.scan_mid_ind
            avoid_type = "back"
            if side.all():
                print "need to move right"
                self.move_diag_right()
                avoid_type = "right"
            elif not side.any():
                print "need to move left"
                self.move_diag_left()
                avoid_type = "left"

            else:
                print "need to back up"
                self.three_pnt()
            self.collision_pub.publish(avoid_type)
        else:
            self.collision_pub.publish(None)

    def get_odom(self):
        try:
            t = self.buffer.lookup_transform('%s_odom' % self.robot_name, self.robot_name, rospy.Time(0))
            q = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
            yaw = tf_conversions.transformations.euler_from_quaternion(q)[2]
            return True, np.array([t.transform.translation.x * 1000, t.transform.translation.y * 1000, yaw])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Transform for PF with error")
            return False, np.array([0, 0, 0])

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
