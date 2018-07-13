#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Polygon
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud
import numpy as np
from threading import Lock
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class PredictOpponentMotion:
    def __init__(self):
        rospy.init_node("predict_opponent_motion", anonymous=True)

        rospy.on_shutdown(self.show_animation)

        self.TIMES = rospy.get_param("/predict_opponent_motion/TIMES", 10)
        # self.MULT = rospy.get_param("/predict_opponent_motion/MULT")
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

        self.fig = plt.figure()
        self.img = []

        # rospy.Publisher("")
        rospy.Subscriber("/secondary_robot/map", OccupancyGrid, self.update_map, queue_size=3)
        rospy.Subscriber("/spy/detected_robots", PointCloud, self.detected_robot, queue_size=1)

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
            uniform_prior = np.ones_like(array255, dtype=np.float64)# * 100.0 / self.open_spots
            print 100.0/self.open_spots
            uniform_prior[array255 == 100] = 100.0
            self.permissible_region = np.ones_like(array255)
            self.permissible_region[array255 == 100] = 0  # set occupied regions (100) to 0 and unoccupied regions to 1
            self.temporal_occupancy_grid = np.repeat(uniform_prior[np.newaxis, :, :], self.TIMES, axis=0)
            # plt.imshow(self.temporal_occupancy_grid[0])
            # plt.show()
            self.map_updated = True
            print "Map Updated"

    def detected_robot(self, msg):
        if self.map_updated:
            # robot_centers = self.map_to_occupancy_grid(msg.points)
            # centers = robot_centers
            # for center in robot_centers:
            #     centers = np.append(centers, [center + 1], axis=0)
            #     centers = np.append(centers, [center - 1], axis=0)
            #     centers = np.append(centers, [center + np.array([0, 1])], axis=0)
            #     centers = np.append(centers, [center + np.array([0, -1])], axis=0)
            #     centers = np.append(centers, [center + np.array([1, 0])], axis=0)
            #     centers = np.append(centers, [center + np.array([-1, 0])], axis=0)
            # mask = np.zeros(self.temporal_occupancy_grid.shape[1:], dtype=bool)
            # print centers
            # mask[centers[:, 0], centers[:, 1]] = True
            # print msg.points
            mask = self.our_robot_circle(msg.points[-1].z/2.0, msg.points[-1])
            # print mask
            # P(state is occupied | observations) = P(state is occupied)*P(observations | state is occupied)/P(observations)
            # P(obs) = P(obs | state occupied) * P(state occupuied) + P(obs | state not occupied) * P(state not occupied)
            self.temporal_occupancy_grid[0, mask] = \
                np.minimum(95, self.temporal_occupancy_grid[0, mask] * self.SENSITIVITY * 100 /
                    (self.SENSITIVITY * self.temporal_occupancy_grid[0, mask] +
                     (1 - self.SENSITIVITY) * (100 - self.temporal_occupancy_grid[0, mask])))
            # print self.temporal_occupancy_grid[0, mask]
            self.temporal_occupancy_grid[0, ~mask] = np.ceil(self.temporal_occupancy_grid[0, ~mask] * (1 - self.SPECIFICITY) * 100 / \
                                                ((1 - self.SPECIFICITY) * self.temporal_occupancy_grid[0, ~mask] +
                                                 self.SPECIFICITY * (100 - self.temporal_occupancy_grid[0, ~mask])))
            # plt.imshow(self.temporal_occupancy_grid[0])
            # plt.pause(0.0001)
            self.img.append([plt.imshow(self.temporal_occupancy_grid[0])])

    def map_to_occupancy_grid(self, pnts):
        return np.array([(([pnt.x, pnt.y, pnt.z] - np.array([self.x0, self.y0, 0])) / self.resolution) for pnt in pnts]).astype(int)

    def our_robot_circle(self, rad, coords):
        # print rad
        # print coords.x+self.x0
        # print coords.y+self.y0
        # 'occupy' all cells
        robot = np.full(self.permissible_region.shape, True, dtype='bool')
        # print robot.all()
        # plt.imshow(robot)
        # plt.pause(1)

        x, y = np.meshgrid(np.arange(0, self.permissible_region.shape[1]), np.arange(0, self.permissible_region.shape[0]))

        coords_prog = np.linspace(np.pi/2+.1, np.pi+.1, 8)

        # upper point
        x1 = (coords.x + self.x0 - rad * np.sin(coords_prog)) / self.resolution
        y1 = (coords.y + self.y0 + rad * np.cos(coords_prog)) / self.resolution

        # lower point
        x2 = (coords.x + self.x0 + rad * np.sin(coords_prog)) / self.resolution
        y2 = (coords.y + self.y0 - rad * np.cos(coords_prog)) / self.resolution

        # left point
        x3 = (coords.x + self.x0 - rad * np.cos(coords_prog)) / self.resolution
        y3 = (coords.y + self.y0 - rad * np.sin(coords_prog)) / self.resolution

        # right point
        x4 = (coords.x + self.x0 + rad * np.cos(coords_prog)) / self.resolution
        y4 = (coords.y + self.y0 + rad * np.sin(coords_prog)) / self.resolution

        # print x1,x2,x3,x4
        # print y1,y2,y3,y4

        # 'free' cells outside of each side of the robot
        for i, coord in enumerate(coords_prog):
            if i < len(coords_prog):
                # plt.imshow(robot)
                # print robot.any()
                # print i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i
                # plt.pause(1)
                a = coord % (2 * np.pi)
                if a < np.pi / 2 or a > 3 * np.pi / 2:
                    robot[y - y1[i] > (x - x1[i]) * np.tan(coord)] = False
                    robot[y - y2[i] < (x - x2[i]) * np.tan(coord)] = False
                else:
                    robot[y - y1[i] < (x - x1[i]) * np.tan(coord)] = False
                    robot[y - y2[i] > (x - x2[i]) * np.tan(coord)] = False
                if a < np.pi:
                    robot[y - y3[i] < (x - x3[i]) * np.tan(np.pi / 2 + coord)] = False
                    robot[y - y4[i] > (x - x4[i]) * np.tan(np.pi / 2 + coord)] = False
                else:
                    robot[y - y3[i] > (x - x3[i]) * np.tan(np.pi / 2 + coord)] = False
                    robot[y - y4[i] < (x - x4[i]) * np.tan(np.pi / 2 + coord)] = False
                # print robot.any()
        # print robot.any()
        # plt.imshow(robot)
        # plt.pause(1)

        return robot

    def show_animation(self):
    #     print self.i
    #     self.fig = plt.figure()
    #     self.im = plt.imshow(self.temporal_occupancy_grid[0], animated=True)
    #     print "HI"
    #     self.ani = FuncAnimation(self.fig, self.next_frame, frames=np.arange(self.i), blit=True)
    #     plt.show()
    #
    # def next_frame(self, num):
    #     self.im.append([self.temporal_occupancy_grid[num]])
    #     print "Hello"
    #     return self.im

        ani = animation.ArtistAnimation(self.fig, self.img, interval=20, blit=True, repeat_delay=0)
        ani.save("updatingMap.mp4")
        # plt.show()


if __name__ == "__main__":
    predictor = PredictOpponentMotion()
rospy.spin()
