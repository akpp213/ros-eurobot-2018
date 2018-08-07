#!/usr/bin/env python
import rospy
from rospy.numpy_msg import numpy_msg
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Polygon
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud

import numpy as np
# np.set_printoptions(threshold=np.nan)

from ak_eurobot_navigation.msg import OccupancyGrid3D

from threading import Lock
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d

import tf


class PredictOpponentMotion:
    def __init__(self):
        rospy.init_node("predict_opponent_motion", anonymous=True)

        self.TIMES = rospy.get_param("/predict_opponent_motion/TIMES", 14)
        # self.MULT = rospy.get_param("/predict_opponent_motion/MULT")
        self.SENSITIVITY = rospy.get_param("/predict_opponent_motion/SENSITIVITY", 0.95)
        self.SPECIFICITY = rospy.get_param("/predict_opponent_motion/SPECIFICITY", 0.7)
        self.UPDATE_RATE = rospy.get_param("~RATE", 40)
        self.TOWERS = np.array(rospy.get_param("/field/towers")) / 1000.0
        self.CUBES = np.array(rospy.get_param("/field/cubes")) / 1000.0
        self.SWITCHES = np.array([rospy.get_param("/field/switches_x"), [150.0, 150.0]]).T / 1000.0
        self.POI = np.concatenate((self.TOWERS[:, :2], self.CUBES, self.SWITCHES), axis=0)
        self.NUM_PLACES = max(2, rospy.get_param("~NUM_PLACES", 3))

        self.x0 = None
        self.y0 = None
        self.map_width = None
        self.map_height = None
        self.map_depth = None
        self.resolution = None
        self.shape_map = None
        self.open_spots = None
        self.permissible_region = None
        self.direction_mask = None
        self.temporal_occupancy_grid = None
        self.probability_grid_3d = None
        self.map_updated = False

        self.last_centers = None
        self.i = 0
        self.j = 0
        self.start_time = None
        self.time_diff = None
        self.stop = False
        self.sigma_x = 40
        self.sigma_y = 40

        self.x, self.y = None, None

        self.loop_time = 0

        self.animate = False
        self.animate_predicted_motion = False
        if self.animate:
            self.fig = plt.figure()
            self.img = []
            rospy.on_shutdown(self.show_animation)

        if self.animate_predicted_motion:
            self.fig = plt.figure()
            # self.ax = plt.axes(projection='3d')
            # self.ax.set_zlim(bottom=0, top=100)
            # self.ax.view_init(5, -135)
            self.img = []
            rospy.on_shutdown(self.show_animation)

        self.listener = tf.TransformListener()

        # rospy.Publisher("")
        rospy.Subscriber("/main_robot/map", OccupancyGrid, self.update_map, queue_size=3)
        rospy.Subscriber("/spy/detected_robots", PointCloud, self.detected_robot, queue_size=1)
        self.movement_pub = rospy.Publisher("movement_marker", Marker, queue_size=2)
        self.poi_pub = rospy.Publisher("poi_marker", Marker, queue_size=2)
        self.pub_opponent_map = rospy.Publisher("map_w_opponent", OccupancyGrid, queue_size=1)
        self.pub_motion_prediction = rospy.Publisher("motion_prediction", numpy_msg(OccupancyGrid3D), queue_size=3)

    def update_map(self, msg):
        # print "Updating Map"
        if not self.map_updated:
            self.x0 = msg.info.origin.position.x  # -.2
            self.y0 = msg.info.origin.position.y  # -.2
            self.map_width = msg.info.width * msg.info.resolution
            self.map_height = msg.info.height * msg.info.resolution
            self.resolution = msg.info.resolution
            self.shape_map = (msg.info.width, msg.info.height)
            self.x, self.y = np.meshgrid(np.arange(self.shape_map[0]), np.arange(self.shape_map[1]))
            # self.ax.set_ylim(bottom=min(0, self.shape_map[1] - self.shape_map[0]), top=self.shape_map[1])
            # self.ax.set_xlim(left=min(0, self.shape_map[0] - self.shape_map[1]), right=self.shape_map[0])
            array255 = np.array(msg.data).reshape((msg.info.height, msg.info.width))
            self.open_spots = msg.info.width * msg.info.height - np.count_nonzero(array255)
            self.uniform_prior = np.ones_like(array255, dtype=np.float64)# * 100.0 / self.open_spots
            self.uniform_prior[array255 == 100] = 100.0
            self.permissible_region = np.ones_like(array255)
            self.permissible_region[array255 == 100] = 0  # set occupied regions (100) to 0 and unoccupied regions to 1
            self.temporal_occupancy_grid = np.repeat(self.uniform_prior[np.newaxis, :, :], self.TIMES, axis=0)
            self.probability_grid_3d = np.zeros_like(self.temporal_occupancy_grid)
            self.direction_mask = np.full(self.permissible_region.shape, 0)
            # plt.imshow(self.temporal_occupancy_grid[0])
            # plt.imshow(self.permissible_region)
            # plt.pause(.1)
            self.map_updated = True
            print "Map Updated"

    def detected_robot(self, msg):
        if self.map_updated:
            robot_centers = self.map_to_occupancy_grid(msg.points)
            main, secondary = self.our_poses()
            if main is None or secondary is None or not len(robot_centers) or self.stop:
                return
            centers = robot_centers[~((np.linalg.norm(robot_centers[:, :2]-main[:2], axis=1) <= 0.3) |
                                    (np.linalg.norm(robot_centers[:, :2]-secondary[:2], axis=1) <= 0.3))]

            # mask = np.full(self.permissible_region.shape, 0)
            # z = np.full(self.permissible_region.shape, 0)
            # print len(centers), "NUM CENTERS"
            # prev = 0
            if self.i == 0:
                now = time.time()
                if self.start_time:
                    self.time_diff = now - self.start_time
                self.start_time = now

                velocities = []
                self.probability_grid_3d = np.zeros((self.TIMES, self.permissible_region.shape[0],
                                                     self.permissible_region.shape[1]))

                # prev = 1
                # self.direction_mask = np.full(self.permissible_region.shape, 0)
                start = time.clock()
                for j, center in enumerate(centers):
                # if self.i == 0:
                    if self.last_centers is None:
                        self.last_centers = np.array([center])
                    else:
                        min_loc = np.argmin(np.linalg.norm(self.last_centers[:, :2] - center[:2], axis=1))
                        displacement = self.last_centers[min_loc, :2] - center[:2]
                        if np.linalg.norm(displacement) <= 0.3:
                            vel = None
                            if self.time_diff:
                                vel = displacement/self.time_diff#-self.loop_time/2)
                                velocities.append(np.linalg.norm(vel))
                                # print vel, displacement, self.time_diff
                            movement = self.direction_of_motion(center, self.last_centers[min_loc])
                            # Point of interest in direction of movement
                            diffs = np.absolute((self.direction_of_motion(self.POI, center) - movement + np.pi)
                                                % (2*np.pi) - np.pi)
                            # poi_in_dom = self.POI[np.argmin(diffs)]
                            # self.vis_poi(poi_in_dom)

                            # Cool Visualization
                            # for k in range(self.NUM_PLACES):
                            #     if k > 0:
                            #         poi_in_dom = self.POI[self.nth_smallest(diffs, k+1)]
                            #     # self.get_prob(k)
                            #     size = np.array([np.linalg.norm(poi_in_dom - center[:2]), center[2]])/self.resolution
                            #     ang = np.arctan2(poi_in_dom[1] - center[1], poi_in_dom[0] - center[0])
                            #     mid_point = (poi_in_dom - center[:2])/2.0 + center[:2]
                            #     self.direction_mask = np.minimum(self.NUM_PLACES, self.direction_mask +
                            #                          self.paths(size, np.append(mid_point, ang), self.get_prob(k)))

                            if vel is not None:
                                for t in range(self.TIMES):
                                    # z2 = np.full(self.permissible_region.shape, 0)
                                    for k in range(self.NUM_PLACES):
                                        poi_in_dom = self.POI[self.nth_smallest(diffs, k + 1)]
                                        if k == 0 and t == 0:
                                            self.vis_movement(self.last_centers[min_loc], center, j)
                                            self.vis_poi(poi_in_dom)
                                        # z2 += self.motion_model(center, vel, t, poi_in_dom, self.get_prob(k)/self.NUM_PLACES)
                                        self.probability_grid_3d[t] += self.motion_model(center, vel, t, poi_in_dom, self.get_prob(k)/self.NUM_PLACES)
                                    if self.animate_predicted_motion:
                                        # self.img.append([self.ax.plot_wireframe(self.x, self.y, self.probability_grid_3d[t])])
                                        # z2[-1, -1] = 100
                                        self.img.append([plt.imshow(self.probability_grid_3d[t])])
                            # print self.last_centers[min_loc] - center
                            self.last_centers[min_loc] = center
                        else:
                            self.last_centers = np.append(self.last_centers, [center], axis=0)
                    self.j = (self.j + 1) % self.TIMES
                    # self.temporal_occupancy_grid[self.j] = self.uniform_prior
                # mask = np.maximum(mask, self.our_robot_circle(center[2]/2.0, center[:2]))

                # z = np.maximum(z, self.gaussian(center))
                self.loop_time = time.clock() - start



            # if self.i == 0:
                # ax = plt.axes(projection='3d')
                # ax.plot_wireframe(self.x, self.y, z2)
                # ax.view_init(75, -90)
                # plt.pause(0.001)
            # ax = plt.axes(projection='3d')
            # ax.plot_wireframe(self.x, self.y, z)
            # ax.view_init(5, -90)
            # # plt.imshow(z)
            # plt.pause(0.001)

            # mask = np.maximum(mask, self.direction_mask)

            # BAYESIAN UPDATE
            # cond = (mask > 0) & (self.temporal_occupancy_grid[0] < 100)
            # cond2 = (mask == 0) & (self.temporal_occupancy_grid[0] < 100)
            # self.temporal_occupancy_grid[0, cond] = \
            #     np.minimum(99, self.temporal_occupancy_grid[0, cond] * self.SENSITIVITY /
            #                (self.SENSITIVITY * self.temporal_occupancy_grid[0, cond] + (1 - self.SENSITIVITY) *
            #                 (100 - self.temporal_occupancy_grid[0, cond]))) * mask[cond] / float(self.NUM_PLACES)
            # # if len(centers) == 2 and not self.animate:
            # #     print self.temporal_occupancy_grid[0, mask]
            # self.temporal_occupancy_grid[0, cond2] = \
            #     np.maximum(1, self.temporal_occupancy_grid[0, cond2] * (1 - self.SPECIFICITY) /
            #                ((1 - self.SPECIFICITY) * self.temporal_occupancy_grid[0, cond2] +
            #                 self.SPECIFICITY * (100 - self.temporal_occupancy_grid[0, cond2])))

            # UPDATE USING GAUSSIAN NOISE
            # cond = (mask > 0) & (self.temporal_occupancy_grid[0] < 100)
            # cond2 = (mask == 0) & (self.temporal_occupancy_grid[0] < 100)
            # self.temporal_occupancy_grid[0, cond] = \
            #     np.minimum(99, self.temporal_occupancy_grid[0, cond] * self.SENSITIVITY * z[cond] /
            #                (self.SENSITIVITY * self.temporal_occupancy_grid[0, cond] + (1 - self.SENSITIVITY) *
            #                 (100 - self.temporal_occupancy_grid[0, cond]))) * mask[cond] / float(self.NUM_PLACES)
            #
            # self.temporal_occupancy_grid[0, cond2] = \
            #     np.maximum(1, self.temporal_occupancy_grid[0, cond2] * (1 - self.SPECIFICITY) * z[cond2] /
            #                ((1 - self.SPECIFICITY) * self.temporal_occupancy_grid[0, cond2] +
            #                 self.SPECIFICITY * (100 - self.temporal_occupancy_grid[0, cond2])))

            # cond = self.temporal_occupancy_grid[0] < 100
            # self.temporal_occupancy_grid[0, cond] = \
            #     np.clip(self.temporal_occupancy_grid[0, cond] * self.SENSITIVITY * z[cond] /
            #             (self.SENSITIVITY * self.temporal_occupancy_grid[0, cond] + (1 - self.SENSITIVITY) *
            #              (100 - self.temporal_occupancy_grid[0, cond])), 1, 99)

            # plt.imshow(self.temporal_occupancy_grid[0])
            # plt.pause(0.001)
            # ax = plt.axes(projection='3d')
            # ax.plot_wireframe(self.x, self.y, self.temporal_occupancy_grid[0])
            # ax.view_init(75, -90)
            # plt.pause(0.001)
            if self.i == 0 and self.time_diff and velocities:
                # permissible = self.temporal_occupancy_grid[0] <= 50
                # self.permissible_region[~permissible] = 0
                # self.permissible_region[permissible] = 1
                # plt.imshow(self.permissible_region)
                # plt.pause(1)
                # self.pub_occupancy_grid()

                self.pub_3d_grid(self.time_diff, velocities)

            self.i = (self.i + 1) % self.UPDATE_RATE

            if self.animate:
                # self.img.append([plt.imshow(self.permissible_region)])
                self.img.append([plt.imshow(self.temporal_occupancy_grid[0])])

    @staticmethod
    def direction_of_motion(pos, last_pos):
        if len(pos.shape) > 1:
            return np.arctan2(pos[:, 1] - last_pos[1], pos[:, 0] - last_pos[0])
        return np.arctan2(pos[1] - last_pos[1], pos[0] - last_pos[0])

    @staticmethod
    def nth_smallest(arr, n):
        if n >= len(arr):
            return np.argmax(arr)
        return np.argpartition(arr, n)[n-1]

    def get_prob(self, n):
        # if n <= 1 - np.ceil(self.NUM_PLACES/2.0) + self.NUM_PLACES/2:
        # if self.NUM_PLACES % 2 == 1:
        return 1.0 + (self.NUM_PLACES - n - 1)/2.0
        # else:
            # return 1.0/min(2, self.NUM_PLACES-1)*(50 - np.clip(self.NUM_PLACES-3, 0, 1)*50/3)
            # return 1 + (self.NUM_PLACES - n - 1)/2

    def our_poses(self):
        try:
            (trans, rot) = self.listener.lookupTransform('/map', '/main_robot', rospy.Time(0))
            yaw = tf.transformations.euler_from_quaternion(rot)[2]
            state_main = np.array([trans[0], trans[1], yaw])
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo("Simulator failed to lookup tf for main robot")
            return None, None

        try:
            (trans, rot) = self.listener.lookupTransform('/map', '/secondary_robot', rospy.Time(0))
            yaw = tf.transformations.euler_from_quaternion(rot)[2]
            state_secondary = np.array([trans[0], trans[1], yaw])
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo("Simulator failed to lookup tf for secondary robot")
            return None, None

        return state_main, state_secondary

    @staticmethod
    def map_to_occupancy_grid(pnts):
        return np.array([[pnt.x, pnt.y, pnt.z] for pnt in pnts])

    def paths(self, size, coords, prob):
        # 'occupy' all cells
        robot = np.full(self.permissible_region.shape, prob)

        # x, y = np.meshgrid(np.arange(self.permissible_region.shape[1]), np.arange(self.permissible_region.shape[0]))
        # upper point
        x1 = (coords[0]-self.x0) / self.resolution - size[1] / 2 * np.sin(coords[2])
        y1 = (coords[1]-self.y0) / self.resolution + size[1] / 2 * np.cos(coords[2])

        # lower point
        x2 = (coords[0]-self.x0) / self.resolution + size[1] / 2 * np.sin(coords[2])
        y2 = (coords[1]-self.y0) / self.resolution - size[1] / 2 * np.cos(coords[2])

        # left point
        x3 = (coords[0]-self.x0) / self.resolution - size[0] / 2 * np.cos(coords[2])
        y3 = (coords[1]-self.y0) / self.resolution - size[0] / 2 * np.sin(coords[2])

        # right point
        x4 = (coords[0]-self.x0) / self.resolution + size[0] / 2 * np.cos(coords[2])
        y4 = (coords[1]-self.y0) / self.resolution + size[0] / 2 * np.sin(coords[2])

        # 'free' cells outside of each side of the robot
        a = coords[2] % (2 * np.pi)
        if a < np.pi / 2 or a > 3 * np.pi / 2:
            robot[self.y - y1 > (self.x - x1) * np.tan(coords[2])] = 0
            robot[self.y - y2 < (self.x - x2) * np.tan(coords[2])] = 0
        else:
            robot[self.y - y1 < (self.x - x1) * np.tan(coords[2])] = 0
            robot[self.y - y2 > (self.x - x2) * np.tan(coords[2])] = 0
        if a < np.pi:
            robot[self.y - y3 < (self.x - x3) * np.tan(np.pi / 2 + coords[2])] = 0
            robot[self.y - y4 > (self.x - x4) * np.tan(np.pi / 2 + coords[2])] = 0
        else:
            robot[self.y - y3 > (self.x - x3) * np.tan(np.pi / 2 + coords[2])] = 0
            robot[self.y - y4 < (self.x - x4) * np.tan(np.pi / 2 + coords[2])] = 0

        return robot

    def our_robot_circle(self, rad, coords):
        # 'occupy' all cells
        robot = np.full(self.permissible_region.shape, self.NUM_PLACES)

        # x, y = np.meshgrid(np.arange(self.permissible_region.shape[1]), np.arange(self.permissible_region.shape[0]))

        coords_prog = np.linspace(np.pi/2+.1, np.pi+.1, 8)

        # upper point
        x1 = (coords[0] - self.x0 - rad * np.sin(coords_prog)) / self.resolution
        y1 = (coords[1] - self.y0 + rad * np.cos(coords_prog)) / self.resolution

        # lower point
        x2 = (coords[0] - self.x0 + rad * np.sin(coords_prog)) / self.resolution
        y2 = (coords[1] - self.y0 - rad * np.cos(coords_prog)) / self.resolution

        # left point
        x3 = (coords[0] - self.x0 - rad * np.cos(coords_prog)) / self.resolution
        y3 = (coords[1] - self.y0 - rad * np.sin(coords_prog)) / self.resolution

        # right point
        x4 = (coords[0] - self.x0 + rad * np.cos(coords_prog)) / self.resolution
        y4 = (coords[1] - self.y0 + rad * np.sin(coords_prog)) / self.resolution

        # 'free' cells outside of each side of the robot
        for i, coord in enumerate(coords_prog):
            if i < len(coords_prog):
                a = coord % (2 * np.pi)
                if a < np.pi / 2 or a > 3 * np.pi / 2:
                    robot[self.y - y1[i] > (self.x - x1[i]) * np.tan(coord)] = 0
                    robot[self.y - y2[i] < (self.x - x2[i]) * np.tan(coord)] = 0
                else:
                    robot[self.y - y1[i] < (self.x - x1[i]) * np.tan(coord)] = 0
                    robot[self.y - y2[i] > (self.x - x2[i]) * np.tan(coord)] = 0
                if a < np.pi:
                    robot[self.y - y3[i] < (self.x - x3[i]) * np.tan(np.pi / 2 + coord)] = 0
                    robot[self.y - y4[i] > (self.x - x4[i]) * np.tan(np.pi / 2 + coord)] = 0
                else:
                    robot[self.y - y3[i] > (self.x - x3[i]) * np.tan(np.pi / 2 + coord)] = 0
                    robot[self.y - y4[i] < (self.x - x4[i]) * np.tan(np.pi / 2 + coord)] = 0

        return robot

    def motion_model(self, center, vel, t_step, goal, prob):
        diff = goal - center[:2]
        dist = np.linalg.norm(vel) * t_step * 0.5
        part = dist/np.linalg.norm(diff)
        new_center = center[:2] + diff * min(1, part)
        c_x = (new_center[0] - self.x0) / self.resolution
        c_y = (new_center[1] - self.y0) / self.resolution
        # sigma_x = sig_x * center[2] * (t_step*.25*abs(np.cos(np.arctan2(vel[1], vel[0])))+1)
        # sigma_y = sig_y * center[2] * (t_step*.25*abs(np.sin(np.arctan2(vel[1], vel[0])))+1)
        sigma_x = self.sigma_x * center[2]/2 * (t_step * np.clip(2 - part, 0, 1) * .25 * abs(np.cos(np.arctan2(diff[1], diff[0]))) + 1)
        sigma_y = self.sigma_y * center[2]/2 * (t_step * np.clip(2 - part, 0, 1) * .25 * abs(np.sin(np.arctan2(diff[1], diff[0]))) + 1)
        const = 100 * 100.0 ** 2 / (self.sigma_x * (t_step * np.clip(2 - part, 0, 1) * .25 *
                                                    abs(np.cos(np.arctan2(diff[1], diff[0]))) + 1) *
                                    self.sigma_y * (t_step * np.clip(2 - part, 0, 1) * .25 *
                                                    abs(np.cos(np.arctan2(diff[1], diff[0]))) + 1) * 2 * np.pi)
        exp = np.exp(-np.power(self.x - c_x, 2) / (2.0 * sigma_x) ** 2 -
                     np.power(self.y - c_y, 2) / (2.0 * sigma_y) ** 2)
        z = const * exp * prob
        return z

    def gaussian(self, center):
        # x, y = np.meshgrid(np.arange(self.shape_map[0]), np.arange(self.shape_map[1]))
        c_x = (center[0] - self.x0) / self.resolution
        c_y = (center[1] - self.y0) / self.resolution
        sigma_x = self.sigma_x * center[2]
        sigma_y = self.sigma_y * center[2]
        const = 150 * 100.0**2/(self.sigma_x * self.sigma_y * 2 * np.pi)
        exp = np.exp(-np.power(self.x - c_x, 2) / (2.0 * sigma_x) ** 2 -
                     np.power(self.y - c_y, 2) / (2.0 * sigma_y) ** 2)
        z = const * exp
        # ax = plt.axes(projection='3d')
        # ax.plot_wireframe(self.x, self.y, z)
        # ax.view_init(75, -90)
        # plt.pause(0.01)
        return z

    def pub_occupancy_grid(self):
        grid = OccupancyGrid()
        grid.header.stamp = rospy.Time.now()
        grid.header.frame_id = "map"
        grid.info.map_load_time = rospy.Time.now()
        grid.info.resolution = self.resolution
        grid.info.width = self.shape_map[0]
        grid.info.height = self.shape_map[1]
        grid.info.origin.position.x = self.x0
        grid.info.origin.position.y = self.y0
        grid.info.origin.orientation.w = 1

        grid.data = self.permissible_region.flatten()

        self.pub_opponent_map.publish(grid)

    def pub_3d_grid(self, rate, velocities):
        grid = OccupancyGrid3D()
        grid.header.stamp = rospy.Time.now()
        grid.header.frame_id = "map"
        grid.info.map_load_time = rospy.Time.now()
        grid.info.resolution = self.resolution
        # grid.info.time_resolutions = np.array(velocities, dtype=np.float32)
        grid.info.time_resolution = rate
        grid.info.depth = self.TIMES
        grid.info.width = self.shape_map[0]
        grid.info.height = self.shape_map[1]
        grid.info.origin.position.x = self.x0
        grid.info.origin.position.y = self.y0
        grid.info.origin.orientation.w = 1
        grid.velocities = np.array(velocities, dtype=np.float32)
        print velocities, self.time_diff
        grid.data = np.minimum(100, self.probability_grid_3d.flatten()).astype(np.float32)
        self.pub_motion_prediction.publish(grid)

    def vis_movement(self, last, current, j):
        arrow = Marker()
        arrow.header.frame_id = "map"
        arrow.header.stamp = rospy.Time.now()
        arrow.id = 7 + j
        arrow.type = 0
        arrow.action = 0
        # arrow.pose.position.x = 1
        # arrow.pose.position.y = 1
        # arrow.pose.orientation.w = movement
        start = Point()
        start.x = last[0]
        start.y = last[1]
        end = Point()
        end.x = (current[0] - last[0]) * 5 + last[0]
        end.y = (current[1] - last[1]) * 5 + last[1]
        arrow.points = [start, end]
        arrow.scale.x = 0.02
        arrow.scale.y = 0.05
        # arrow.scale.z = 1
        arrow.color.a = 1
        arrow.color.r = 1
        arrow.color.g = 0
        arrow.color.b = 0
        self.movement_pub.publish(arrow)

    def vis_poi(self, poi):
        sphere = Marker()
        sphere.header.frame_id = "map"
        sphere.header.stamp = rospy.Time.now()
        sphere.id = 2
        sphere.type = 2
        sphere.action = 0
        sphere.pose.position.x = poi[0]
        sphere.pose.position.y = poi[1]
        sphere.pose.position.z = .1
        # arrow.pose.orientation.w = movement
        # start = Point()
        # start.x = last[0]
        # start.y = last[1]
        # end = Point()
        # end.x = (current[0] - last[0]) * 10 + last[0]
        # end.y = (current[1] - last[1]) * 10 + last[1]
        # arrow.points = [start, end]
        sphere.scale.x = 0.1
        sphere.scale.y = 0.1
        sphere.scale.z = 0.1
        sphere.color.a = 1
        sphere.color.r = 0
        sphere.color.g = 1
        sphere.color.b = 0
        self.poi_pub.publish(sphere)

    def show_animation(self):
        print len(self.img)
        self.stop = True
        ani = animation.ArtistAnimation(self.fig, self.img, interval=500/self.TIMES, blit=False, repeat_delay=0)
        # ani.save("updatingMap.mp4")
        plt.show()


if __name__ == "__main__":
    predictor = PredictOpponentMotion()
rospy.spin()
