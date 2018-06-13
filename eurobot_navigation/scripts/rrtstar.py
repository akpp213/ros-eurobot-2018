#!/usr/bin/env python
import rospy
import math
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Polygon
from visualization_msgs.msg import Marker
import numpy as np
import matplotlib.pyplot as plt


class Node(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

    def distance(self, random_point):
        distance = np.sqrt((self.x-random_point[0])**2 + (self.y - random_point[1])**2)
        return distance


class RRTStar:
    def __init__(self):
        rospy.init_node("rrtstar", anonymous=True)
        self.MAX_RRT_ITERS = rospy.get_param("motion_planner/MAX_RRT_ITERS")  # 1000
        self.STEP = rospy.get_param("motion_planner/STEP")  # 0.1
        self.GAMMA_RRT = rospy.get_param("motion_planner/GAMMA_RRT")  # 3
        self.EPSILON_GOAL = rospy.get_param("motion_planner/EPSILON_GOAL")  # 0.2
        self.PATH_WIDTH = rospy.get_param("motion_planner/PATH_WIDTH")  # 0.1
        self.GOAL_SAMPLE_RATE = rospy.get_param("motion_planner/GOAL_SAMPLE_RATE")
        self.coords = np.zeros(3)
        self.goal = np.zeros(3)

        self.nodes = []
        # self.nodes_secondary = None
        self.path = []
        self.turn_angle = None

        self.x0 = -0.2
        self.y0 = -0.2
        self.map_width = None
        self.map_height = None
        self.resolution = None
        self.shape_map = (304, 204)
        self.permissible_region = None
        self.map_updated = False

        self.new_path = rospy.Publisher('rrt_path', Polygon, queue_size=1)
        rospy.Subscriber('current_coords', Point, self.update_coords, queue_size=1)
        rospy.Subscriber('new_goal_loc', Point, self.find_path_rrtstar, queue_size=1)
        rospy.Subscriber("/main_robot/map", OccupancyGrid, self.update_map, queue_size=1)
        rospy.Subscriber("/secondary_robot/map", OccupancyGrid, self.update_map, queue_size=1)
        # rospy.Subscriber("lookahead_pnts", Point, self.add_look, queue_size=1)

    def update_coords(self, msg):
        self.path = []
        self.coords[0] = msg.x
        self.coords[1] = msg.y
        self.coords[2] = msg.z
        print (self.coords[:2] - [self.x0, self.y0])/self.resolution, "coooooooords"
        print (1.0 - self.x0)/self.resolution, (1.0 - self.y0)/self.resolution, "end pntttt"

    def update_map(self, msg):
        # print "Updating Map"
        self.x0 = msg.info.origin.position.x
        self.y0 = msg.info.origin.position.y
        self.map_width = msg.info.width * msg.info.resolution
        self.map_height = msg.info.height * msg.info.resolution
        self.resolution = msg.info.resolution
        self.shape_map = (msg.info.width, msg.info.height)
        array255 = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.permissible_region = np.ones_like(array255, dtype=bool)
        self.permissible_region[array255 == 100] = 0  # setting occupied regions (100) to 0. Unoccupied regions are a 1
        # self.permissible_region[34:74, 0:34] = 1
        # self.permissible_region[82:122, 82:122] = 1
        # self.permissible_region[53:93, 34:74] = 1
        self.map_updated = True
        # print self.permissible_region
        # self.permissible_region

    # def add_look(self, msg):
    #     print "adding new follow pnt"
    #     plt.plot(msg.x, msg.y, 'og')

    def find_path_rrtstar(self, msg):
        self.goal[0] = msg.x
        self.goal[1] = msg.y
        self.goal[2] = msg.z

        plt.plot(self.coords[0], self.coords[1], 'ob')
        plt.plot(self.goal[0], self.goal[1], 'or')
        plt.pause(.001)
        x, y, th = self.coords
        start_node = Node(x, y)
        self.path = []
        self.nodes = [start_node]
        j = 0
        path_found = False
        plt.axis([self.x0, self.x0 + self.map_width, self.y0 + self.map_height, self.y0])
        while not path_found and (j < self.MAX_RRT_ITERS):

            # random sample
            if j % self.GOAL_SAMPLE_RATE == 0:
                # sample goal location every 10 iterations
                rnd = [self.goal[0], self.goal[1]]
            else:
                rnd = [self.x0 + self.map_width * np.random.uniform(), self.y0 + self.map_height * np.random.uniform()]

            # plt.plot(rnd[0], rnd[1], 'og')
            # find the nearest node
            nn_idx = self.nearest_node(rnd)
            x_near = self.nodes[nn_idx]

            # expand the tree
            x_new, theta = self.steer_to_node(x_near, rnd)  # goes from x_near to x_new which points towards rnd
            x_new.parent = nn_idx

            # print "length nodes"
            # print j
            # print ""
            if not j % 50:
                print j

            # check for obstacle collisions
            if self.obstacle_free(x_near, x_new, theta):
                # plt.plot([x_near.x, x_new.x], [x_new.y, x_new.y])

                # find close nodes
                near_idxs = self.find_near_nodes(x_new)

                # find the best parent for the node
                x_new = self.choose_parent(x_new, near_idxs)
                # plt.plot([x_new.x, self.nodes[x_new.parent].x], [x_new.y, self.nodes[x_new.parent].y])
                # plt.pause(.0001)

                # add new node and rewire
                self.nodes.append(x_new)
                self.rewire(x_new, near_idxs)

                # check if sample point is in goal region
                dx = x_new.x - self.goal[0]
                dy = x_new.y - self.goal[1]
                d = np.sqrt(dx ** 2 + dy ** 2)
                # print self.EPSILON_GOAL
                if d <= self.EPSILON_GOAL:
                    print "found goal!"
                    # construct path
                    last_idx = self.get_last_idx()
                    if last_idx is None:
                        return
                    path = self.get_path(last_idx)
                    self.path = np.array(path)
                    self.new_path.publish(self.to_poly(self.path))
                    self.visualize()
                    return

                    # for point in self.path:
                    # 	pt_obj = Point()
                    # 	pt_obj.x = point[0]
                    # 	pt_obj.y = point[1]

                    # 	self.trajectory.addPoint(pt_obj)
                    # self.publish_trajectory()

                    # return self.path

            j += 1

        self.visualize()

    def nearest_node(self, random_point):
        """ Finds the index of the nearest node in self.nodes to random_point """
        # index of nearest node
        nn_idx = 0
        # loops through all nodes and finds the closest one
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            nearest_node = self.nodes[nn_idx]
            if node.distance(random_point) < nearest_node.distance(random_point):
                nn_idx = i
        return nn_idx

    def steer_to_node(self, x_near, rnd):
        """ Steers from x_near to a point on the line between rnd and x_near """

        theta = math.atan2(rnd[1] - x_near.y, rnd[0] - x_near.x)
        new_x = x_near.x + self.STEP * math.cos(theta)
        new_y = x_near.y + self.STEP * math.sin(theta)

        x_new = Node(new_x, new_y)
        x_new.cost += self.STEP

        return x_new, theta

    def obstacle_free(self, nearest_node, new_node, theta):
        """ Checks if the path from x_near to x_new is obstacle free """

        dx = math.sin(theta) * self.PATH_WIDTH
        dy = math.cos(theta) * self.PATH_WIDTH

        if nearest_node.x < new_node.x:
            bound0 = ((nearest_node.x, nearest_node.y), (new_node.x, new_node.y))
            bound1 = ((nearest_node.x + dx, nearest_node.y - dy), (new_node.x + dx, new_node.y - dy))
            bound2 = ((nearest_node.x - dx, nearest_node.y + dy), (new_node.x - dx, new_node.y + dy))
            dx*=.5
            dy*=.5
            bound3 = ((nearest_node.x + dx, nearest_node.y - dy), (new_node.x + dx, new_node.y - dy))
            bound4 = ((nearest_node.x - dx, nearest_node.y + dy), (new_node.x - dx, new_node.y + dy))
        else:
            bound0 = ((new_node.x, new_node.y), (nearest_node.x, nearest_node.y))
            bound1 = ((new_node.x + dx, new_node.y - dy), (nearest_node.x + dx, nearest_node.y - dy))
            bound2 = ((new_node.x - dx, new_node.y + dy), (nearest_node.x - dx, nearest_node.y + dy))
            dx*=.5
            dy*=.5
            bound3 = ((new_node.x + dx, new_node.y - dy), (nearest_node.x + dx, nearest_node.y - dy))
            bound4 = ((new_node.x - dx, new_node.y + dy), (nearest_node.x - dx, nearest_node.y + dy))

        if self.line_collision(bound0) or self.line_collision(bound1) or self.line_collision(bound2) or\
                self.line_collision(bound3) or self.line_collision(bound4):
            return False
        else:
            return True

    def line_collision(self, line):
        """ Checks if line collides with obstacles in the map"""

        # discretize values of x and y along line according to map using Bresemham's alg

        x_ind = np.arange(np.ceil((line[0][0]) / self.resolution), np.ceil((line[1][0]) / self.resolution + 1), dtype=int)
        y_ind = []

        dx = max(1, np.ceil(line[1][0] / self.resolution) - np.ceil(line[0][0] / self.resolution))
        dy = np.ceil(line[1][1] / self.resolution) - np.ceil(line[0][1] / self.resolution)
        deltaerr = abs(dy / dx)
        error = .0

        y = int(np.ceil((line[0][1]) / self.resolution))
        for _ in x_ind:
            y_ind.append(y)
            error = error + deltaerr
            while error >= 0.5:
                y += np.sign(dy) * 1
                error += -1

        y_ind = np.array(y_ind)
        # check if any cell along the line contains an obstacle
        # plt.plot(x_ind, y_ind)
        # plt.pause(0.001)
        for i in range(len(x_ind)):
            row = min([int(-self.y0 / self.resolution + y_ind[i]), self.shape_map[1] - 1])
            column = min([int(-self.x0 / self.resolution + x_ind[i]), self.shape_map[0] - 1])
            # print row, column, "rowcol"
            # print self.map_width, self.map_height
            if not self.permissible_region[row, column]:
                # print "returning true for a line collision"
                return True
        return False

    def find_near_nodes(self, x_new):
        length_nodes = len(self.nodes)
        r = self.GAMMA_RRT * math.sqrt((math.log(length_nodes) / length_nodes))
        near_idxs = []
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            if node.distance((x_new.x, x_new.y)) <= r:
                near_idxs.append(i)
        return near_idxs

    def choose_parent(self, x_new, near_idxs):
        if len(near_idxs) == 0:
            return x_new

        distances = []
        for i in near_idxs:
            node = self.nodes[i]
            d = node.distance((x_new.x, x_new.y))
            dx = x_new.x - node.x
            dy = x_new.y - node.y
            theta = math.atan2(dy, dx)

            if self.obstacle_free(node, x_new, theta):
                distances.append(d)
            else:
                distances.append(float("inf"))
        mincost = min(distances)
        minind = near_idxs[distances.index(mincost)]

        if mincost == float("inf"):
            return x_new

        x_new.cost = mincost
        x_new.parent = minind

        return x_new

    def rewire(self, x_new, near_idxs):
        n = len(self.nodes)
        for i in near_idxs:
            node = self.nodes[i]
            d = node.distance((x_new.x, x_new.y))
            scost = x_new.cost + d

            if node.cost > scost:
                dx = x_new.x - node.x
                dy = x_new.y - node.y
                theta = math.atan2(dy, dx)
                if self.obstacle_free(node, x_new, theta):
                    node.parent = n - 1
                    node.cost = scost

    def get_last_idx(self):
        dlist = []
        for node in self.nodes:
            d = node.distance((self.goal[0], self.goal[1]))
            dlist.append(d)
        goal_idxs = [dlist.index(i) for i in dlist if i <= self.STEP]

        if len(goal_idxs) == 0:
            return None

        mincost = min([self.nodes[i].cost for i in goal_idxs])

        for i in goal_idxs:
            if self.nodes[i].cost == mincost:
                return i

        return None

    def get_path(self, last_idx):
        path = [(self.goal[0], self.goal[1])]
        while self.nodes[last_idx].parent is not None and (self.nodes[last_idx].x, self.nodes[last_idx].y) not in path:
            node = self.nodes[last_idx]
            path.append((node.x, node.y))
            last_idx = node.parent
        path.append((self.coords[0], self.coords[1]))
        path.reverse()
        return path

    @staticmethod
    def to_poly(path):
        poly = Polygon()
        for pt in path:
            point = Point()
            point.x = pt[0]
            point.y = pt[1]
            point.z = 0.0
            poly.points.append(point)
        return poly

    def visualize(self):
        if self.nodes:
            line_strip = Marker()
            line_strip.type = line_strip.LINE_STRIP
            line_strip.action = line_strip.ADD
            line_strip.header.frame_id = "/map"

            line_strip.scale.x = 0.05

            line_strip.color.a = 1.0
            line_strip.color.r = 1.0
            line_strip.color.g = 0.0
            line_strip.color.b = 0.0

            # marker orientaiton
            line_strip.pose.orientation.x = 0.0
            line_strip.pose.orientation.y = 0.0
            line_strip.pose.orientation.z = 0.0
            line_strip.pose.orientation.w = 1.0

            # marker position
            line_strip.pose.position.x = 0.0
            line_strip.pose.position.y = 0.0
            line_strip.pose.position.z = 0.0

            # marker line points
            line_strip.points = []

            for node in self.nodes:
                if node.parent is not None:
                    parent = self.nodes[node.parent]

                    plt.plot([node.x, parent.x], [node.y, parent.y], "-g")

                    theta = math.atan2(node.y - parent.y, node.x - parent.x)

                    dx = math.sin(theta) * self.PATH_WIDTH
                    dy = math.cos(theta) * self.PATH_WIDTH

                    plt.plot([parent.x + dx, node.x + dx], [parent.y - dy, node.y - dy], "-g")
                    plt.plot([parent.x - dx, node.x - dx], [parent.y + dy, node.y + dy], "-g")

            x, y = self.coords[:2]
            plt.plot(x, y, "ob")
            x, y = self.goal[:2]
            plt.plot(x, y, "or")
            plt.axis([self.x0, self.x0 + self.map_width, self.y0 + self.map_height, self.y0])

            plt.imshow(self.permissible_region,
                       extent=[self.x0, self.x0 + self.map_width, self.y0 + self.map_height, self.y0])
            plt.pause(0.01)

            for node in self.path:
                new_point = Point()
                new_point.x = node[0]
                new_point.y = node[1]
                new_point.z = 0.0
                line_strip.points.append(new_point)

            # Publish the Marker
            # self.viz_pub.publish(line_strip)

            # visualizes the computed path in RVIZ
            # self.visualize_path()

    # @staticmethod
    # def visualize_path():
    #     line_strip = Marker()
    #     line_strip.type = line_strip.LINE_STRIP
    #     line_strip.action = line_strip.ADD
    #     line_strip.header.frame_id = "/map"
    #
    #     line_strip.ns = "path"


if __name__ == "__main__":
    rrtstar = RRTStar()

rospy.spin()
