#!/usr/bin/env python
import rospy
from sensor_msgs.msg import LaserScan, PointCloud
from geometry_msgs.msg import Point
import numpy as np
import tf
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt

from threading import Lock

MIN_INTENSITY = 3500
LIDAR_DELTA_ANGLE = (np.pi / 180) / 4
LIDAR_X = -0.094
LIDAR_Y = 0.050
LIDAR_START_ANGLE = 1.63
FIELD_X = 3
FIELD_Y = 2
BEYOND_FIELD = 0.1
# R = 0.15
visualization = True


class Spy:
    def __init__(self):
        rospy.init_node("spy", anonymous=True)

        self.R = rospy.get_param("R")
        self.eps = rospy.get_param("clustering/eps")
        self.min_samples = rospy.get_param("clustering/min_samples")
        self.listener = tf.TransformListener()
        # self.last_main_landmarks = np.array([[], []]).T
        # self.last_secondary_landmarks = np.array([[], []]).T
        self.last_centers = [[], []]

        self.mutex = Lock()
        # self.last_secondary_centers = np.array([])
        # name = rospy.get_param("/robot_name")
        # laser_min_angle = rospy.get_param("/secondary_robot/simulate/laser_min_angle")
        # lidar_start_angle = rospy.get_param("/secondary_robot/lidar_a")
        rospy.Subscriber("/secondary_robot/scan", LaserScan, self.scan_callback_secondary, queue_size=1)
        rospy.Subscriber("/main_robot/scan", LaserScan, self.scan_callback_main, queue_size=1)
        self.pub_center = rospy.Publisher("detected_robots", PointCloud, queue_size=1)
        if visualization:
            self.pub_landmarks = rospy.Publisher("landmarks", PointCloud, queue_size=1)

    def scan_callback_main(self, scan):
        self.scan_callback(scan, "main_robot", "secondary_robot")

    def scan_callback_secondary(self, scan):
        self.scan_callback(scan, "secondary_robot", "main_robot")

    def scan_callback(self, scan, name, teammate):
        lidar_data = np.array([np.array(scan.ranges), scan.intensities]).T
        landmarks = self.filter_scan(lidar_data, name, teammate)
        # if name == "main_robot":
        #     # self.last_main_landmarks = self.change_coords(landmarks, name, "secondary_robot")
        #     self.last_main_landmarks = landmarks.copy()
        #     landmarks = np.concatenate((landmarks, self.last_secondary_landmarks))
        # else:
        #     # self.last_secondary_landmarks = self.change_coords(landmarks, name, "main_robot")
        #     self.last_secondary_landmarks = landmarks.copy()
        #     landmarks = np.concatenate((landmarks, self.last_main_landmarks))

        if visualization:
            # create and pub PointArray of detected beacon points
            points = [Point(x=landmarks[i, 0], y=landmarks[i, 1], z=0) for i in range(len(landmarks))]
            array = PointCloud(points=points)
            array.header.frame_id = "map"
            array.header.stamp = rospy.Time.now()
            self.pub_landmarks.publish(array)

        centers = []
        self.last_centers[self.conv_to_int(name)] = []
        if landmarks.shape[0] > 0:
            # clustering
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(landmarks)
            # print db.labels_
            labels = db.labels_
            unique_labels = set(labels)

            for l in unique_labels:
                if l == -1:
                    # noise
                    continue

                class_member_mask = (labels == l)

                group = landmarks[class_member_mask]
                size = max(np.linalg.norm(group - group[::-1], axis=1))
                R = size/2
                center = self.get_center(group, R, name)
                # print center
                self.last_centers[self.conv_to_int(name)].append(np.append(center, size))
                other_centers = np.array(self.last_centers[self.conv_to_int(name, 1)])
                if other_centers.shape[0] > 0:
                    # print other_centers
                    closest_other = np.argmin(np.linalg.norm(other_centers[:, :2] - center, axis=1))
                    # print closest_other, "closest_other"
                    other_center = other_centers[closest_other]
                    max_size = max(size, other_center[2])
                    if np.linalg.norm(other_center[:2] - center) < max_size:
                        center = (center + other_center[:2]) / 2.0
                        size = max_size
                centers.append(Point(x=center[0], y=center[1], z=size))

        # create and pub PointArray of detected centers
        array = PointCloud(points=centers)
        array.header.frame_id = "map"
        array.header.stamp = rospy.Time.now()
        self.pub_center.publish(array)

    def change_coords(self, landmarks, name, other_name):
        loc = self.get_robot_loc(name)
        other = self.get_robot_loc(other_name)
        if loc is None or other is None:
            return np.array([[], []]).T

        converted_landmarks = 0
        return converted_landmarks

    def filter_scan(self, scan, name, teammate):
        """Filters scan to get only landmarks (bright) located on field."""
        self.mutex.acquire()
        loc = self.get_robot_loc(name)
        if loc is None:
            self.mutex.release()
            return np.array([np.array([]), np.array([])]).T
        laser_min_angle = rospy.get_param("/"+name+"/simulate/laser_min_angle")
        lidar_start_angle = rospy.get_param("/"+name+"/lidar_a")
        ind = np.where(scan[:, 1] > MIN_INTENSITY)[0]
        a = LIDAR_DELTA_ANGLE * ind + laser_min_angle
        d = scan[ind, 0]

        # x = d * np.cos(a + LIDAR_START_ANGLE) + LIDAR_X
        # y = d * np.sin(a + LIDAR_START_ANGLE) + LIDAR_Y

        x = d * np.cos(a + loc[2] + lidar_start_angle+.0) + loc[0]# + .02
        y = d * np.sin(a + loc[2] + lidar_start_angle+.0) + loc[1]# + .02

        # inside field only and not teammate
        team_conds = True
        teammate_loc = self.get_robot_loc(teammate)
        if teammate_loc is not None:
            # teammate_loc[:2] = .02
            teammate_shape = np.array([rospy.get_param("/" + teammate + "/dim_x"),
                                       rospy.get_param("/" + teammate + "/dim_y")]) / 2000.0
            transformation = np.array([[np.cos(teammate_loc[2]), np.sin(teammate_loc[2])],
                                       [-np.sin(teammate_loc[2]), np.cos(teammate_loc[2])]])
            # print teammate_shape
            # print teammate_loc[2]
            tl = teammate_loc[:2] + np.matmul(teammate_shape * [-1, 1], transformation)
            tr = teammate_loc[:2] + np.matmul(teammate_shape * [1, 1], transformation)
            bl = teammate_loc[:2] + np.matmul(teammate_shape * [-1, -1], transformation)
            br = teammate_loc[:2] + np.matmul(teammate_shape * [1, -1], transformation)
            if teammate == "secondary_robot":
                team_conds = ~(self.generate_conds(bl, br, x, y, teammate_loc[2]) |
                               self.generate_conds(tr, tl, x, y, teammate_loc[2]) |
                               self.generate_conds(br, tr, x, y, teammate_loc[2]) |
                               self.generate_conds(tl, bl, x, y, teammate_loc[2]))
            else:
                team_conds = ~(self.generate_conds(br, bl, x, y, teammate_loc[2]) |
                               self.generate_conds(tl, tr, x, y, teammate_loc[2]) |
                               self.generate_conds(tr, br, x, y, teammate_loc[2]) |
                               self.generate_conds(bl, tl, x, y, teammate_loc[2]))
            # plt.show()
            # i=0
            # for a,b in tuple(np.array([x,y]).T):
            #     color = 'og'
            #     if not team_conds[i]:
            #         color = 'or'
            #     plt.plot(a,b,color)
            #     plt.pause(.001)
            #     i+=1
            # plt.show()
        ind = np.where(np.logical_and(np.logical_and(x < FIELD_X + BEYOND_FIELD, x > -BEYOND_FIELD),
                                      np.logical_and(y < FIELD_Y + BEYOND_FIELD, y > - BEYOND_FIELD)) & team_conds)

        x = x[ind]
        y = y[ind]

        self.mutex.release()
        return np.array([x, y]).T

    @staticmethod
    def generate_conds(pt1, pt2, x, y, angle, epsilon=.03):
        if angle % (np.pi/2) == 0:
            return (x >= (min(pt1[0], pt2[0]) - epsilon)) & (x <= (max(pt1[0], pt2[0]) + epsilon)) & \
                   (y >= (min(pt1[1], pt2[1]) - epsilon)) & (y <= (max(pt1[1], pt2[1]) + epsilon))
        else:
            slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
            a = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
            # print slope, angle
            # plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'c')
            for x_val, y_val in tuple(np.array([x, y]).T):
                color1 = 'og'
                if (x_val >= (min(pt1[0], pt2[0]) - epsilon)) & (x_val <= (max(pt1[0], pt2[0]) + epsilon)) & \
                   (y_val >= (x_val - pt1[0] - epsilon * np.sin(a) * np.sign(slope)) * slope + pt1[1] - epsilon * abs(np.cos(a))) & \
                   (y_val <= (x_val - pt1[0] + epsilon * np.sin(a) * np.sign(slope)) * slope + pt1[1] + epsilon * abs(np.cos(a))):
                    color1 = 'or'
                # plt.plot(x_val - epsilon*np.sin(angle), y_val - epsilon*np.cos(angle), color1)
                # plt.plot(x_val + epsilon*np.sin(angle), y_val + epsilon*np.cos(angle), color1)
                # plt.plot(x_val, y_val, color1)
                # plt.plot(x_val, (x_val - pt1[0] - epsilon * np.sin(angle) * np.sign(slope)) * slope + pt1[1]
                #          - epsilon * abs(np.cos(angle)), 'ob')
                # plt.plot(x_val, (x_val - pt1[0] + epsilon * np.sin(angle) * np.sign(slope)) * slope + pt1[1]
                #          + epsilon * abs(np.cos(angle)), 'oy')
                # plt.pause(0.001)
            # plt.show()
            return (x >= (min(pt1[0], pt2[0]) - epsilon)) & (x <= (max(pt1[0], pt2[0]) + epsilon)) & \
                   (y >= (x - pt1[0] - epsilon * np.sin(a) * np.sign(slope)) * slope + pt1[1] - epsilon * abs(np.cos(a))) & \
                   (y <= (x - pt1[0] + epsilon * np.sin(a) * np.sign(slope)) * slope + pt1[1] + epsilon * abs(np.cos(a)))

    def get_robot_loc(self, name):
        try:
            (trans, rot) = self.listener.lookupTransform('/map', '/' + name, rospy.Time(0))
            yaw = tf.transformations.euler_from_quaternion(rot)[2]
            state = np.array([trans[0], trans[1], yaw])
            return state
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo("Simulator failed to lookup tf for " + name)
            return None

    def get_center(self, landmarks, R, name):
        #bounds = (point - np.array([0.1, 0.1]), point + np.array([0.15, 0.15])
        loc = self.get_robot_loc(name)
        if loc is not None:
            landmarks[:, :2] -= loc[:2]
        med = np.median(landmarks, axis=0)
        dist = np.sum(med ** 2) ** .5
        center_by_med = med + R * np.array([med[0] / dist, med[1] / dist])
        # return center_by_med
        center = least_squares(self.fun, center_by_med, args=(landmarks, R))#, loss="linear", bounds = bounds, args = [landmarks], ftol = 1e-3)
        if loc is not None:
            center.x += loc[:2]
        return center.x

    def fun(self, point, landmarks, R):
        return np.sum((landmarks - point) ** 2, axis=1) ** .5 - R

    @staticmethod
    def conv_to_int(name, invert=0):
        val = 0 if name == "main_robot" else 1
        if invert:
            val = abs(val - 1)
        return val


if __name__ == "__main__":
    spy = Spy()
rospy.spin()
