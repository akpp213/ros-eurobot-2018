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
        self.ROBOT_NAME = rospy.get_param("robot_name")
        self.SIZE = np.array([rospy.get_param("/" + self.ROBOT_NAME + "/dim_x"),
                              rospy.get_param("/" + self.ROBOT_NAME + "/dim_y")]) / 1000.0
        self.ANGLE_RESOLUTION = rospy.get_param("~ANGLE_RESOLUTION", 0.5)
        self.GRID_RESOLUTION = rospy.get_param("~GRID_RESOLUTION", 0.3)
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
        if self.visualize:
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
        # combined = np.ones(self.configuration_space.shape[1:], dtype=bool)
        # for i in xrange(len(self.angles)):
        #     combined &= self.configuration_space[i]
        # plt.imshow(combined)
        # plt.show()

        # 3D Scatter Plot
        if self.visualize:
            self.ax.scatter3D(X, Y, Z, c=Z)
            self.ax.view_init(90, -90)
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
        x, y, th = np.meshgrid(np.arange(self.x0, self.x0 + self.map_width, self.GRID_RESOLUTION),
                               np.arange(self.y0, self.y0 + self.map_height, self.GRID_RESOLUTION),
                               np.linspace(-np.pi, np.pi, self.angles.shape[0] * 2, endpoint=False))

        self.nodes = np.stack((x.flatten(), y.flatten(), th.flatten()), axis=-1)
        # print self.nodes
        converted_nodes = self.convert(self.nodes)
        # print converted_nodes
        self.nodes = self.nodes[self.configuration_space[tuple(np.flip(np.moveaxis(converted_nodes, -1, 0), 0))]]
        self.edges = np.zeros((self.nodes.shape[0] ** 2, 2), dtype=int)
        self.edges_list = [None] * len(self.nodes)
        # for node in self.nodes:
        #     self.subplots[int(self.closest_angle(node[2]))].plot(int((node[0] - self.x0)/self.resolution),
        #                                                          int((node[1] - self.y0) / self.resolution), 'og')
        #     # plt.pause(0.0001)
        # plt.show()
        # self.nodes = [start]

        for node_ind, node in enumerate(self.nodes):
            if node_ind % 100 == 0:
                print node_ind
            near_idxs = self.find_near_nodes(node)
            # print near_idxs
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
                if not self.line_collision(self.nodes[ind], node):
                    # plt.plot([x_near.x, x_new.x], [x_new.y, x_new.y])
                    self.edges[self.edge_ind] = [ind, node_ind]
                    self.edge_ind += 1

                    if self.edges_list[ind] is None:
                        self.edges_list[ind] = [node_ind]
                    else:
                        self.edges_list[ind].append(node_ind)

                    if self.edges_list[node_ind] is None:
                        self.edges_list[node_ind] = [ind]
                    else:
                        self.edges_list[node_ind].append(ind)

                    edge_added = True

            # if not edge_added:
            #     print "Disconnected Node :("
                # self.nodes[self.node_ind] = new
                # self.node_ind += 1

                    # self.subplots[time_step].plot([(x_new.x-self.x0) / self.resolution, (self.nodes[x_new.parent].x-self.x0) / self.resolution],
                    #          [(x_new.y-self.y0) / self.resolution, (self.nodes[x_new.parent].y-self.y0) / self.resolution], 'g-')
                    # plt.pause(.1)

    def closest_angle(self, angle):
        return np.argmin(np.abs((angle - self.angles + np.pi / 2) % np.pi - np.pi / 2), axis=-1)

    def convert(self, nodes):
        new = np.empty(nodes.shape)
        new[:, :2] = ((nodes[:, :2] - np.array([self.x0, self.y0])) / self.resolution)
        new[:, 2] = self.closest_angle(np.repeat(nodes[:, 2, np.newaxis], self.angles.shape[0], axis=1))
        return new.astype(int)

    def theta_diff(self, angle1, angle2):
        dth = self.closest_angle(angle2) - self.closest_angle(angle1)
        back_dth = (self.closest_angle(angle1) - self.closest_angle(angle2)) \
            % (2 * self.configuration_space.shape[0]) - self.configuration_space.shape[0]
        if abs(back_dth) < abs(dth):
            dth = -back_dth

        return dth

    def find_near_nodes(self, new):
        r = 1.9 * self.GRID_RESOLUTION
        dists = np.linalg.norm(self.nodes[:, :2] - new[:2], axis=1)
        max_angle = dists * self.MAX_ANGULAR_RATE
        near_idxs = np.nonzero((dists != 0) & (dists <= r) &
                               (np.absolute((self.nodes[:, 2] - new[2] + np.pi) % (2 * np.pi) - np.pi) <= max_angle))[0]
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
        f.close()

    def convert_nodes_to_points(self):
        self.points = [None] * self.nodes.shape[0]
        disconnected = 0
        for i in xrange(self.nodes.shape[0]):
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
        graph.edges = [None] * self.nodes.shape[0]
        total_edges = 0
        for i in xrange(self.nodes.shape[0]):
            edges = Edges()
            total_edges += len(self.edges_list[i])
            # print self.edges_list[i]
            edges.node_ids = self.edges_list[i]
            graph.edges[i] = edges
        print total_edges, "edges published"
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