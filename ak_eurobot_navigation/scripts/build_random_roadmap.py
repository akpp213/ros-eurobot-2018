#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap
from geometry_msgs.msg import Point, Polygon
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool
from sensor_msgs.msg import PointCloud
import numpy as np
np.set_printoptions(threshold=np.nan)
import tf

from ak_eurobot_navigation.msg import Graph, Edges

from threading import Lock
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class BuildRoadmap:
    def __init__(self):
        rospy.init_node("build_roadmap", anonymous=True, disable_signals=True)

        self.MAX_ITERS = rospy.get_param("~MAX_ITERS", 1000)
        self.NEW_CENTER_RATE = rospy.get_param("~NEW_CENTER_RATE", 15)
        self.ROBOT_NAME = rospy.get_param("~ROBOT_NAME", "main_robot")
        self.SIZE = np.array([rospy.get_param("/" + self.ROBOT_NAME + "/dim_x"),
                              rospy.get_param("/" + self.ROBOT_NAME + "/dim_y")]) / 1000.0
        self.ANGLE_RESOLUTION = rospy.get_param("~ANGLE_RESOLUTION", 0.1)
        self.HEAP_CENTERS = np.array(rospy.get_param("/field/cubes")) / 1000.0
        self.TOWER_CENTERS = np.array(rospy.get_param("/field/towers")) / 1000.0
        self.CUBE_SIZE = rospy.get_param("~CUBE_SIZE", 58) / 1000.0
        self.RAD_CIRCUMSCRIBING_CIRCLE = np.linalg.norm([1.5*self.CUBE_SIZE, 0.5*self.CUBE_SIZE])
        self.HEAP_ELLIPSE = self.SIZE/2 + self.RAD_CIRCUMSCRIBING_CIRCLE
        self.COLOR = rospy.get_param("/field/color")
        self.STEP = rospy.get_param("~STEP", .15)
        self.GAMMA_NEAR = rospy.get_param("~GAMMA_NEAR", 4)
        self.MAX_ANGULAR_RATE = rospy.get_param("~MAX_ANGULAR_RATE", 2)

        self.mutex = Lock()

        self.x0 = -0.2
        self.y0 = -0.2
        self.map_width = None
        self.map_height = None
        self.resolution = None
        self.shape_map = (304, 204)
        self.permissible_region = None
        self.configuration_space = None
        self.map_updated = False
        self.subplots = []
        # self.ax = plt.axes(projection='3d')

        self.nodes = None
        self.points = None
        self.edges = None
        self.edges_list = None
        self.angles = None
        self.node_ind = 0
        self.edge_ind = 0

        self.map_sent = False

        self.visualize = False

        self.pub_roadmap = rospy.Publisher("roadmap", Graph, queue_size=1)
        rospy.Subscriber("/" + self.ROBOT_NAME + "/map", OccupancyGrid, self.update_map, queue_size=3)

    def update_map(self, msg):
        self.mutex.acquire()
        if not self.map_updated:
            self.x0 = msg.info.origin.position.x  # -.2
            self.y0 = msg.info.origin.position.y  # -.2
            self.map_width = msg.info.width * msg.info.resolution
            self.map_height = msg.info.height * msg.info.resolution
            self.resolution = msg.info.resolution
            self.shape_map = (msg.info.width, msg.info.height)
            array255 = np.array(msg.data).reshape((msg.info.height, msg.info.width))
            self.permissible_region = np.ones_like(array255, dtype=bool)
            self.permissible_region[array255 == 100] = 0  # set occupied regions (100) to 0 and unoccupied regions to 1
            self.map_updated = True
            print "Map Initialized"
            if self.visualize:
                self.ax = plt.axes(projection='3d')
            self.add_angle_dimension()
            self.make_configuration_space()
            self.make_roadmap()
            print "Roadmap Complete!"
            self.convert_nodes_to_points()
            self.publish()
            self.write_to_txt()
            self.map_sent = True
            print "Roadmap published"
            if self.visualize:
                self.visualize_roadmap()
        self.mutex.release()
        if self.map_sent:
            rospy.signal_shutdown("Roadmap has been sent. Node is finished")

    # def get_omap(self):
    #     map_service_name = rospy.get_param("~static_map", "static_map")
    #     print "Fetching map from service: ", map_service_name
    #     rospy.wait_for_service(map_service_name)
    #     msg = rospy.ServiceProxy(map_service_name, GetMap)().map
    #     self.x0 = msg.info.origin.position.x  # -.2
    #     self.y0 = msg.info.origin.position.y  # -.2
    #     self.map_width = msg.info.width * msg.info.resolution
    #     self.map_height = msg.info.height * msg.info.resolution
    #     self.resolution = msg.info.resolution
    #     self.shape_map = (msg.info.width, msg.info.height)
    #     array255 = np.array(msg.data).reshape((msg.info.height, msg.info.width))
    #     self.occupancy_grid = np.ones_like(array255, dtype=bool)
    #     self.occupancy_grid[array255 == 100] = 0  # set occupied regions (100) to 0 and unoccupied regions to 1
    #     self.map_updated = True
    #     print "Map Initialized"
    #     self.add_angle_dimension()
    #     self.make_configuration_space()
    #     # self.make_roadmap()

    def add_angle_dimension(self):
        self.configuration_space = np.repeat(self.permissible_region[np.newaxis, :, :],
                                             int(np.pi / self.ANGLE_RESOLUTION), axis=0)

    def make_configuration_space(self):
        self.angles = np.linspace(-np.pi/2, np.pi/2, self.configuration_space.shape[0], endpoint=False)
        # self.angles = [-1.05]
        # X = np.arange(self.shape_map[0])
        # Y = np.arange(self.shape_map[1])
        # X,Y = np.meshgrid(X,Y)
        X = np.array([])
        Y = np.array([])
        Z = np.array([])

        for i, angle in enumerate(self.angles):
            coords = (self.HEAP_CENTERS - np.array([self.x0, self.y0])) / self.resolution
            tower_wall_ellipse_mask = self.add_towers_walls_and_ellipses(self.SIZE, coords, angle)
            treatment_plant_mask = self.add_treatment_plant(self.SIZE, angle)
            # opponent_area_mask = self.add_opponent_area(self.SIZE, angle)
            self.configuration_space[i] &= tower_wall_ellipse_mask & treatment_plant_mask

            if self.visualize:
                edgex = ~self.configuration_space[i] ^ np.roll(self.configuration_space[i], shift=-1, axis=0)
                edgey = ~self.configuration_space[i] ^ np.roll(self.configuration_space[i], shift=-1, axis=1)

                y, x = np.ma.nonzero(~edgex)
                y2, x2 = np.ma.nonzero(~edgey)
                z = np.ones_like(y) * i
                z2 = np.ones_like(y2) * i

                X = np.concatenate((X, x, x2))
                Y = np.concatenate((Y, y, y2))
                Z = np.concatenate((Z, z, z2))

            # plt.imshow(~self.configuration_space[i] & self.permissible_region, origin="lower")
            # plt.show()

            # Each frame in one figure
            # rows = np.round(np.sqrt(self.configuration_space.shape[0]))
            # cols = np.ceil(np.sqrt(self.configuration_space.shape[0]))
            # sub = plt.subplot(rows, cols, i + 1)
            # self.subplots.append(sub)
            # plt.imshow(self.configuration_space[i])

        # 3D Scatter Plot
        if self.visualize:
            self.ax.scatter3D(X, Y, Z, c=Z)
            self.ax.view_init(75, -90)
            # plt.show()

        # plt.pause(0.001)

    def add_ellipses(self, size, coords):
        ellipse = np.full(self.permissible_region.shape, True, dtype='bool')
        x, y = np.meshgrid(np.arange(0, self.permissible_region.shape[1]),
                           np.arange(0, self.permissible_region.shape[0]))

        for coord in coords:
            transformed_x = (x - coord[0]) * np.cos(coord[2]) + (y - coord[1]) * np.sin(coord[2])
            transformed_y = (y - coord[1]) * np.cos(coord[2]) - (x - coord[0]) * np.sin(coord[2])
            ellipse[(transformed_x / size[0])**2 + (transformed_y / size[1])**2 <= 1] = False

        return ellipse

    def add_towers_walls_and_ellipses(self, size, coords, angle):
        mask = np.full(self.permissible_region.shape, True, dtype='bool')
        x, y = np.meshgrid(np.arange(0, self.permissible_region.shape[1]),
                           np.arange(0, self.permissible_region.shape[0]))

        right_left = (np.abs(size[0] * np.cos(angle)) + np.abs(size[1] * np.sin(angle))) / (2 * self.resolution)
        top_bottom = (np.abs(size[1] * np.cos(angle)) + np.abs(size[0] * np.sin(angle))) / (2 * self.resolution)

        # walls
        mask[x >= self.shape_map[0] - right_left - 2] = False
        mask[x <= right_left + 2] = False
        mask[y >= self.shape_map[1] - top_bottom - 2] = False
        mask[y <= top_bottom + 2] = False

        # towers
        towers = (self.TOWER_CENTERS[:, :2] - np.array([self.x0, self.y0])) / self.resolution
        mask[(x <= right_left + towers[0, 0]) &
             (y >= towers[0, 1] + 1.7 - top_bottom) & (y <= towers[0, 1] + 1.7 + top_bottom)] = False
        mask[(y >= towers[1, 1] - top_bottom) &
             (x >= towers[1, 0] - right_left) & (x <= towers[1, 0] + right_left)] = False
        mask[(y >= towers[2, 1] - top_bottom) &
             (x >= towers[2, 0] - right_left) & (x <= towers[2, 0] + right_left)] = False
        mask[(x >= towers[3, 0] - right_left) &
             (y >= towers[3, 1] - 1.7 - top_bottom) & (y <= towers[3, 1] - 1.7 + top_bottom)] = False

        # ellipses
        for coord in coords:
            transformed_x = (x - coord[0]) * np.cos(angle) + (y - coord[1]) * np.sin(angle)
            transformed_y = (y - coord[1]) * np.cos(angle) - (x - coord[0]) * np.sin(angle)
            mask[(transformed_x * self.resolution / (size[0]/2 + self.RAD_CIRCUMSCRIBING_CIRCLE))**2 +
                 (transformed_y * self.resolution / (size[1]/2 + self.RAD_CIRCUMSCRIBING_CIRCLE))**2 <= 1] = False

        return mask

    def add_treatment_plant(self, size, angle):
        mask = np.full(self.permissible_region.shape, False, dtype='bool')
        x, y = np.meshgrid(np.arange(0, self.permissible_region.shape[1]),
                           np.arange(0, self.permissible_region.shape[0]))
        treatment_corners = np.array([[92, 177], [212, 177]])
        robot_corners = np.array([[size[0], size[1]], [-size[0], size[1]],
                                  [size[0], -size[1]], [-size[0], -size[1]]]) / (2*self.resolution)
        transform = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        transformed_corners = np.matmul(robot_corners, transform)
        minkowski_sum = np.concatenate((treatment_corners[0] + transformed_corners,
                                        treatment_corners[1] + transformed_corners))

        minkowski_sum = np.append(minkowski_sum, [[minkowski_sum[np.argsort(minkowski_sum[:, 1])][
            np.argmax(minkowski_sum[np.argsort(minkowski_sum[:, 1])], axis=0)[0]][0], 204]], axis=0)

        sorted_corners = self.find_convex_hull(minkowski_sum, angle)

        for i in xrange(len(sorted_corners)-1):
            slope = (sorted_corners[i+1, 1] - sorted_corners[i, 1]) / (sorted_corners[i+1, 0] - sorted_corners[i, 0])
            mask[y-sorted_corners[i, 1] < (x-sorted_corners[i, 0]) * slope] = True

        mask[0 > (x - sorted_corners[0, 0])] = True
        mask[0 < (x - sorted_corners[-1, 0])] = True

        return mask

    @staticmethod
    def find_convex_hull(corners, angle):
        def rightof(p, q, r):
            if np.all(p == q):
                return True
            mat = [[1, r[0], r[1]], [1, p[0], p[1]], [1, q[0], q[1]]]
            return np.linalg.det(mat) < 0

        pivot = corners[np.argsort(corners[:, 1])][
            np.argmin(corners[np.argsort(corners[:, 1])], axis=0)[0]]
        corners_of_interest = pivot[np.newaxis, :]
        endpoint = corners[0]
        for j in range(1, corners.shape[0]):
            if rightof(pivot, endpoint, corners[j]):
                endpoint = corners[j]
        pivot = endpoint
        while np.any(endpoint != corners[-1]):
            corners_of_interest = np.append(corners_of_interest, [pivot], axis=0)
            endpoint = corners[0]
            for j in range(1, corners.shape[0]):
                if rightof(pivot, endpoint, corners[j]):
                    endpoint = corners[j]
            pivot = endpoint

        sorted_corners = corners_of_interest[corners_of_interest[:, 0].argsort()]
        if angle % (np.pi/2) == 0:
            sorted_corners = sorted_corners[:-1]

        return sorted_corners

    # def add_opponent_area(self, size, angle):
    #     pass

    def make_roadmap(self):
        mid = np.array([self.map_width, self.map_height, 0]) / 2.0
        bbl = np.array([0.47, 0.67, -1.05])
        bbr = np.array([2.57, 0.62, -1.05])
        btl = np.array([0.52, 1.52, -1.05])
        btr = np.array([2.52, 1.57, -1.05])
        tbl = np.array([0.47, 0.62, 1.05])
        tbr = np.array([2.62, 0.67, 1.05])
        ttl = np.array([0.67, 1.52, 1.05])
        ttr = np.array([2.52, 1.52, 1.05])
        self.nodes = np.zeros((self.MAX_ITERS + 9, 3))
        self.edges = np.zeros(((self.MAX_ITERS + 9)**2, 2), dtype=int)
        self.edges_list = [None] * len(self.nodes)
        self.nodes[0] = mid
        self.nodes[1] = bbl
        self.nodes[2] = btl
        self.nodes[3] = btr
        self.nodes[4] = bbr
        self.nodes[5] = tbl
        self.nodes[6] = tbr
        self.nodes[7] = ttl
        self.nodes[8] = ttr
        self.node_ind = 9
        # for node in self.nodes[:self.node_ind]:
        #     self.subplots[int(self.closest_angle(node[2]))].plot((node[0] - self.x0)/self.resolution,
        #                                                          (node[1] - self.y0) / self.resolution, 'og')
        #     plt.pause(0.5)
        # self.nodes = [start]

        for j in xrange(self.MAX_ITERS):
            if not j % 50:
                print j

            rnd_pnt = np.array([self.x0 + self.map_width * np.random.uniform(),
                                self.y0 + self.map_height * np.random.uniform()])
            rnd_angle = -np.pi + 2 * np.pi * np.random.uniform()
            if j % self.NEW_CENTER_RATE == 0 and False:
                closest_angle = self.closest_angle(rnd_angle)
                while not self.configuration_space[closest_angle,
                                                   min(int(np.ceil((rnd_pnt[1] - self.y0) / self.resolution)),
                                                       self.shape_map[1] - 1),
                                                   min(int(np.ceil((rnd_pnt[0] - self.x0) / self.resolution)),
                                                       self.shape_map[0] - 1)]:
                    print "Finding a free spot"
                    rnd_pnt = np.array([self.x0 + self.map_width * np.random.uniform(),
                                        self.y0 + self.map_height * np.random.uniform()])
                    rnd_angle = -np.pi + 2 * np.pi * np.random.uniform()
                    closest_angle = np.argmin(np.abs((rnd_angle - self.angles + np.pi / 2) % np.pi - np.pi / 2))
                self.nodes[self.node_ind, :2] = rnd_pnt
                self.nodes[self.node_ind, 2] = rnd_angle
                self.node_ind += 1

                # self.subplots[int(self.closest_angle(rnd_angle)
                #                   )].plot(int(np.ceil((rnd_pnt[0] - self.x0) / self.resolution)),
                #                           int(np.ceil((rnd_pnt[1] - self.y0) / self.resolution)), 'og')

            else:
                # plt.plot(rnd[0], rnd[1], 'og')
                # find the nearest node
                nn_idx = self.nearest_node(rnd_pnt, rnd_angle)
                near = self.nodes[nn_idx]

                # expand the tree
                new = self.steer_to_node(near, rnd_pnt, rnd_angle)  # goes from x_near to x_new which points towards rnd
                # self.subplots[int(self.closest_angle(new[2]))].plot((new[0] - self.x0) / self.resolution,
                #                                                          (new[1] - self.y0) / self.resolution, 'ob')
                # plt.pause(1)
                if self.configuration_space[self.closest_angle(new[2]),
                                                   min(int(np.ceil((new[1] - self.y0) / self.resolution)),
                                                       self.shape_map[1] - 1),
                                                   min(int(np.ceil((new[0] - self.x0) / self.resolution)),
                                                       self.shape_map[0] - 1)]:
                    # find close nodes
                    near_idxs = self.find_near_nodes(new)
                    # near_idxs = np.arange(len(self.nodes))

                    # self.nodes[self.node_ind] = new
                    # self.edges_list[self.node_ind-9] = []
                    # self.node_ind += 1

                    # self.subplots[int(self.closest_angle(new[2]))].plot(int(np.ceil((new[0] - self.x0) / self.resolution)),
                    #                                                     int(np.ceil((new[1] - self.y0) / self.resolution)),
                    #                                                     'ob')
                    # plt.pause(0.001)
                    edge_added = False
                    for ind in near_idxs:
                        # check for obstacle collisions
                        if not self.line_collision(self.nodes[ind], new):
                            # plt.plot([x_near.x, x_new.x], [x_new.y, x_new.y])
                            self.edges[self.edge_ind] = [ind, self.node_ind-1]
                            self.edge_ind += 1

                            if self.edges_list[ind] is None:
                                self.edges_list[ind] = [self.node_ind]
                            else:
                                self.edges_list[ind].append(self.node_ind)

                            if self.edges_list[self.node_ind] is None:
                                self.edges_list[self.node_ind] = [ind]
                            else:
                                self.edges_list[self.node_ind].append(ind)

                            edge_added = True

                    if edge_added:
                        self.nodes[self.node_ind] = new
                        self.node_ind += 1

                            # self.subplots[time_step].plot([(x_new.x-self.x0) / self.resolution, (self.nodes[x_new.parent].x-self.x0) / self.resolution],
                            #          [(x_new.y-self.y0) / self.resolution, (self.nodes[x_new.parent].y-self.y0) / self.resolution], 'g-')
                            # plt.pause(.1)

    def closest_angle(self, angle):
        return np.argmin(np.abs((angle - self.angles + np.pi / 2) % np.pi - np.pi / 2))

    def theta_diff(self, angle1, angle2):
        dth = self.closest_angle(angle2) - self.closest_angle(angle1)
        back_dth = (self.closest_angle(angle1) - self.closest_angle(angle2)) \
            % (2 * self.configuration_space.shape[0]) - self.configuration_space.shape[0]
        if abs(back_dth) < abs(dth):
            dth = -back_dth

        return dth

    def nearest_node(self, random_point, random_angle):
        """ Finds the index of the nearest node in self.nodes to random_point """
        dists = np.linalg.norm(self.nodes[:self.node_ind, :2] - random_point, axis=1) + \
                np.abs((self.nodes[:self.node_ind, 2] - random_angle + np.pi) % (2*np.pi) - np.pi)
        return np.argmin(dists)

    def steer_to_node(self, near, rnd_pnt, rnd_angle):
        """ Steers from x_near to a point on the line between rnd and x_near """
        dx = rnd_pnt[0] - near[0]
        dy = rnd_pnt[1] - near[1]
        dth = (rnd_angle - near[2] + np.pi) % (2*np.pi) - np.pi
        theta = np.arctan2(dy, dx)
        new_x = near[0] + self.STEP * np.cos(theta)
        new_y = near[1] + self.STEP * np.sin(theta)
        new_th = (near[2] + np.clip(self.STEP/np.linalg.norm([dx, dy]) * dth,
                  -self.MAX_ANGULAR_RATE*self.STEP, self.MAX_ANGULAR_RATE*self.STEP) + np.pi) % (2*np.pi) - np.pi

        return np.array([new_x, new_y, new_th])

    def find_near_nodes(self, new):
        r = max(self.STEP, self.GAMMA_NEAR * np.sqrt((np.log(self.node_ind) / self.node_ind)))
        dists = np.linalg.norm(self.nodes[:self.node_ind, :2] - new[:2], axis=1)
        max_angle = dists * self.MAX_ANGULAR_RATE
        near_idxs = np.nonzero((dists <= r) & (np.absolute((self.nodes[:self.node_ind, 2] - new[2] + np.pi)
                                                           % (2 * np.pi) - np.pi) <= max_angle))[0]
        return near_idxs

    def line_collision(self, start, end, plot=False):
        """ Checks if line collides with obstacles in the map"""

        def correct_method(val, which_delta):
            if which_delta == 2:
                return self.closest_angle(val)
            return int(np.ceil(val / self.resolution))

        def clip(val, which_delta):
            if which_delta == 2:
                return val % self.configuration_space.shape[0]
            return min(np.ceil(val), self.shape_map[which_delta] - 1)

        # self.subplots[int(self.closest_angle(start[2]))].plot((start[0] - self.x0) / self.resolution,
        #                                                       (start[1] - self.y0) / self.resolution, 'ow')
        # self.subplots[int(self.closest_angle(end[2]))].plot((end[0] - self.x0) / self.resolution,
        #                                                     (end[1] - self.y0) / self.resolution, 'ok')
        # plt.pause(1)
        dx = np.ceil((end[0] - start[0]) / self.resolution)
        dy = np.ceil((end[1] - start[1]) / self.resolution)
        dth = self.theta_diff(start[2], end[2])

        deltas = np.array([dx, dy, dth])
        order = np.argsort(-np.absolute(deltas))

        dzero = deltas[order[0]]
        done = deltas[order[1]]
        dtwo = deltas[order[2]]
        skip = 1 if order[0] == 2 else 3

        sign = np.sign(dzero) if dzero != 0.0 else 1
        zero = sign * max(1, abs(dzero))
        deltaerr_one = abs(float(done) / zero)
        deltaerr_two = abs(float(dtwo) / zero)
        error_one = .0
        error_two = .0

        zero_ind = np.arange(correct_method(start[order[0]], order[0]),
                             correct_method(start[order[0]], order[0]) + dzero + sign, sign * skip, dtype=int)
        one_ind = np.empty(zero_ind.shape[0])
        two_ind = np.empty(zero_ind.shape[0])

        one = correct_method(start[order[1]], order[1])
        two = correct_method(start[order[2]], order[2])
        for i in xrange(zero_ind.shape[0]):
            one_ind[i] = one
            two_ind[i] = two
            error_one = error_one + skip * deltaerr_one
            error_two = error_two + skip * deltaerr_two
            while error_one >= 0.5:
                one += np.sign(done)
                error_one -= 1
            while error_two >= 0.5:
                two += np.sign(dtwo)
                error_two -= 1

        offsets = [self.x0, self.y0, 0]
        for i in xrange(len(zero_ind)):
            vals = np.ceil([clip(zero_ind[i] - offsets[order[0]] / self.resolution, order[0]),
                            clip(one_ind[i] - offsets[order[1]] / self.resolution, order[1]),
                            clip(two_ind[i] - offsets[order[2]] / self.resolution, order[2])]).astype(int)
            # print vals[np.where(order == 2)[0][0]]
            if not self.configuration_space[vals[np.where(order == 2)[0][0]], vals[np.where(order == 1)[0][0]],
                                            vals[np.where(order == 0)[0][0]]]:
                # self.subplots[vals[np.where(order == 2)[0][0]]].plot(vals[np.where(order == 0)[0][0]],
                #                                                      vals[np.where(order == 1)[0][0]], 'or')
                # plt.pause(0.05)
                return True
            # color = 'og'
            # if i == len(zero_ind) - 1:
            #     color = 'oc'
            # self.subplots[vals[np.where(order == 2)[0][0]]].plot(vals[np.where(order == 0)[0][0]],
            #                                                      vals[np.where(order == 1)[0][0]], color)
            # plt.pause(0.05)
        return False

    def write_to_txt(self):
        f = open("roadmap.txt", 'w')
        f.write("Nodes:\n")
        f.write(str(self.nodes[:self.node_ind].tolist()))
        f.write("\nEdges:\n")
        f.write(str(self.edges_list[:self.node_ind]))

    def convert_nodes_to_points(self):
        self.points = [None] * self.node_ind
        disconnected = 0
        for i in xrange(self.node_ind):
            p = Point()
            if self.edges_list[i] is None:
                self.edges_list[i] = []
                disconnected += 1
                self.nodes[i][0] = -float('inf')
                self.nodes[i][1] = -float('inf')
                self.nodes[i][2] = -float('inf')
            p.x = self.nodes[i][0]
            p.y = self.nodes[i][1]
            p.z = self.nodes[i][2]
            self.points[i] = p
        print disconnected, "disconnected nodes.", len(self.points), "total nodes"

    def publish(self):
        graph = Graph()
        graph.header.stamp = rospy.Time.now()
        graph.header.frame_id = "map"
        graph.nodes = self.points
        graph.edges = [None] * self.node_ind
        num_edges = 0
        for i in xrange(self.node_ind):
            edges = Edges()
            print self.edges_list[i]
            num_edges += len(self.edges_list[i])
            edges.node_ids = self.edges_list[i]
            graph.edges[i] = edges
        print num_edges, "edges published"
        self.pub_roadmap.publish(graph)

    def visualize_roadmap(self):
        for i in xrange(self.edge_ind):
            # print self.edges[i]
            start = self.nodes[self.edges[i][0]]
            end = self.nodes[self.edges[i][1]]
            dth = self.closest_angle(end[2]) - self.closest_angle(start[2])
            back_dth = (self.closest_angle(start[2]) - self.closest_angle(end[2])) \
                       % (2 * self.configuration_space.shape[0]) - self.configuration_space.shape[0]
            if abs(back_dth) < abs(dth):
                dth = -back_dth
            # z = np.linspace(start[2], start[2] + dth, abs(dth)+1, endpoint=True)
            else:
                z = np.array([self.closest_angle(start[2]), self.closest_angle(end[2])])
                y = (np.array([start[1], end[1]]) - self.y0) / self.resolution
                x = (np.array([start[0], end[0]]) - self.x0) / self.resolution
                self.ax.plot3D(x, y, z)
        plt.show()
        return


if __name__ == "__main__":
    road_map_builder = BuildRoadmap()

rospy.spin()


# Options:
#     1. Time-space configuration space with a priority based de-coupled planning method: https://link.springer.com/content/pdf/10.1007%2FBF01840371.pdf
#         a. More efficient, not complete
#         b. Four dimensions: x,y,th, and time
#         c. Need two separate plans
#     2. Coupled approach using a dynamically created roadmap to decrease the size of the configuration space
#         a. Six dimensions: x,y,th for both robots
#     3. Path coordination (independent planning, then velocity planning to avoid collisions)
#         a. Efficient, more or less existing strategy however
#
# 1. time_step =