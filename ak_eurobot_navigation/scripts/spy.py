#!/usr/bin/env python
import rospy
from sensor_msgs.msg import LaserScan, PointCloud
from geometry_msgs.msg import Point
import numpy as np
import tf
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN


MIN_INTENSITY = 3500
LIDAR_DELTA_ANGLE = (np.pi / 180) / 4
LIDAR_X = -0.094
LIDAR_Y = 0.050
LIDAR_START_ANGLE = 1.63
FIELD_X = 3
FIELD_Y = 2
BEYOND_FIELD = 0.1
R = 0.08
visualization = True


class Spy:
    def __init__(self):
        rospy.init_node("spy", anonymous=True)

        self.R = rospy.get_param("R")
        self.eps = rospy.get_param("clustering/eps")
        self.min_samples = rospy.get_param("clustering/min_samples")
        self.listener = tf.TransformListener()
        self.last_main_landmarks = np.array([[], []]).T
        self.last_secondary_landmarks = np.array([[], []]).T
        # name = rospy.get_param("/robot_name")
        # laser_min_angle = rospy.get_param("/secondary_robot/simulate/laser_min_angle")
        # lidar_start_angle = rospy.get_param("/secondary_robot/lidar_a")
        rospy.Subscriber("/secondary_robot/scan", LaserScan, self.scan_callback_secondary, queue_size=1)
        rospy.Subscriber("/main_robot/scan", LaserScan, self.scan_callback_main, queue_size=1)
        self.pub_center = rospy.Publisher("detected_robots", PointCloud, queue_size=1)
        if visualization:
            self.pub_landmarks = rospy.Publisher("landmarks", PointCloud, queue_size=1)

    def scan_callback_main(self, scan):
        self.scan_callback(scan, "main_robot")

    def scan_callback_secondary(self, scan):
        self.scan_callback(scan, "secondary_robot")

    def scan_callback(self, scan, name):
        lidar_data = np.array([np.array(scan.ranges), scan.intensities]).T
        landmarks = self.filter_scan(lidar_data, name)
        if name == "main_robot":
            self.last_main_landmarks = landmarks.copy()
            landmarks = np.concatenate((landmarks, self.last_secondary_landmarks))
        else:
            self.last_secondary_landmarks = landmarks.copy()
            landmarks = np.concatenate((landmarks, self.last_main_landmarks))

        if visualization:
            # create and pub PointArray of detected beacon points
            points = [Point(x=landmarks[i, 0], y=landmarks[i, 1], z=0) for i in range(len(landmarks))]
            array = PointCloud(points=points)
            array.header.frame_id = "map"
            self.pub_landmarks.publish(array)

        centers = []
        if landmarks.shape[0] > 0:
            # clustering
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(landmarks)
            labels = db.labels_
            unique_labels = set(labels)

            for l in unique_labels:
                if l == -1:
                    # noise
                    continue

                class_member_mask = (labels == l)

                group = landmarks[class_member_mask]
                center = self.get_center(group)
                # size = np.linalg.norm(group[-1] - group[0])
                size = max(np.linalg.norm(group - group[::-1], axis=1))
                centers.append(Point(x=center.x[0], y=center.x[1], z=size))

        # create and pub PointArray of detected centers
        array = PointCloud(points=centers)
        array.header.frame_id = "map"
        array.header.stamp = rospy.Time.now()
        self.pub_center.publish(array)

    def filter_scan(self, scan, name):
        """Filters scan to get only landmarks (bright) located on field."""
        loc = self.get_robot_loc(name)
        if loc is None:
            return np.array([np.array([]), np.array([])]).T
        laser_min_angle = rospy.get_param("/"+name+"/simulate/laser_min_angle")
        lidar_start_angle = rospy.get_param("/"+name+"/lidar_a")
        ind = np.where(scan[:, 1] > MIN_INTENSITY)[0]
        a = LIDAR_DELTA_ANGLE * ind + laser_min_angle
        d = scan[ind, 0]

        # x = d * np.cos(a + LIDAR_START_ANGLE) + LIDAR_X
        # y = d * np.sin(a + LIDAR_START_ANGLE) + LIDAR_Y

        x = d * np.cos(a + loc[2] + lidar_start_angle+.0) + loc[0]
        y = d * np.sin(a + loc[2] + lidar_start_angle+.0) + loc[1]

        # inside field only
        ind = np.where(np.logical_and(np.logical_and(x < FIELD_X + BEYOND_FIELD, x > -BEYOND_FIELD), np.logical_and(y < FIELD_Y + BEYOND_FIELD, y > - BEYOND_FIELD)))

        x = x[ind]
        y = y[ind]
        return np.array([x, y]).T

    def get_robot_loc(self, name):
        try:
            (trans, rot) = self.listener.lookupTransform('/map', '/' + name, rospy.Time(0))
            yaw = tf.transformations.euler_from_quaternion(rot)[2]
            state = np.array([trans[0], trans[1], yaw])
            return state
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo("Simulator failed to lookup tf for " + name)
            return None

    def get_center(self, landmarks):
        #bounds = (point - np.array([0.1, 0.1]), point + np.array([0.15, 0.15])
        med = np.median(landmarks, axis=0)
        dist = np.sum(med ** 2) ** .5
        center_by_med = med + R * np.array([med[0] / dist, med[1] / dist])
        center = least_squares(self.fun, center_by_med, args=[landmarks])#, loss="linear", bounds = bounds, args = [landmarks], ftol = 1e-3)
        return center

    def fun(self, point, landmarks):
        return np.sum((landmarks - point) ** 2, axis=1) ** .5 - R


if __name__ == "__main__":
    spy = Spy()
rospy.spin()
