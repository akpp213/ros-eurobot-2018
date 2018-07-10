#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Polygon
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud
import numpy as np
from threading import Lock
import matplotlib.pyplot as plt


class PredictOpponentMotion:
    def __init__(self):
        rospy.init_node("predict_opponent_motion", anonymous=True)

        self.TIMES = rospy.get_param("/predict_opponent_motion/TIMES", 10)
        self.MULT = rospy.get_param("/predict_opponent_motion/MULT")
        self.SENSITIVITY = rospy.get_param("/predict_opponent_motion/SENSITIVITY", 0.9)
        self.SPECIFICITY = rospy.get_param("/predict_opponent_motion/SPECIFICITY", 0.9)

        self.x0 = None
        self.y0 = None
        self.map_width = None
        self.map_height = None
        self.map_depth = None
        self.resolution = None
        self.shape_map = None
        self.open_spots = None
        self.permissible_region = None
        self.temporal_occupancy_grid = None
        self.map_updated = False

        rospy.Publisher("")
        rospy.Subscriber("map", OccupancyGrid, self.update_map, queue_size=3)
        rospy.Subscriber("detected_robots", PointCloud, self.detected_robot, queue_size=1)

    def update_map(self, msg):
        # print "Updating Map"
        if not self.map_updated:
            self.x0 = msg.info.origin.position.x  # -.2
            self.y0 = msg.info.origin.position.y  # -.2
            self.map_width = msg.info.width * msg.info.resolution
            self.map_height = msg.info.height * msg.info.resolution
            self.resolution = msg.info.resolution
            self.shape_map = (msg.info.width, msg.info.height)
            array255 = np.array(msg.data).reshape((msg.info.height, msg.info.width))
            self.open_spots = msg.info.width * msg.info.height - np.count_nonzero(array255)
            uniform_prior = np.ones_like(array255) * 100.0 / self.open_spots
            uniform_prior[array255 == 100] = 100.0
            self.permissible_region = np.ones_like(array255)
            self.permissible_region[array255 == 100] = 0  # set occupied regions (100) to 0 and unoccupied regions to 1
            self.temporal_occupancy_grid = np.repeat(uniform_prior[np.newaxis, :, :], self.TIMES, axis=0)
            self.map_updated = True
            print "Map Updated"

    def detected_robot(self, msg):
        robot_centers = self.map_to_occupancy_grid(msg.points)
        mask = np.zeros(self.temporal_occupancy_grid.shape[1:], dtype=bool)
        mask[robot_centers[:, 0], robot_centers[:, 1]] = True
        # P(state is occupied | observations) = P(state is occupied)*P(observations | state is occupied)/P(observations)
        # P(obs) = P(obs | state occupied) * P(state occupuied) + P(obs | state not occupied) * P(state not occupied)
        self.temporal_occupancy_grid[0, mask] *= self.SENSITIVITY * 100 / \
                                            (self.SENSITIVITY * self.temporal_occupancy_grid[0, mask] +
                                            (1 - self.SENSITIVITY) * (100 - self.temporal_occupancy_grid[0, mask]))
        self.temporal_occupancy_grid[0, ~mask] *= (1 - self.SPECIFICITY) * 100 / \
                                            ((1 - self.SPECIFICITY) * self.temporal_occupancy_grid[0, ~mask] +
                                             self.SPECIFICITY * (100 - self.temporal_occupancy_grid[0, ~mask]))

    def map_to_occupancy_grid(self, pnts):
        return np.array([([pnt.x, pnt.y] - np.array([self.x0, self.y0])) / self.resolution for pnt in pnts])


if __name__ == "main":
    predictor = PredictOpponentMotion()
rospy.spin()
