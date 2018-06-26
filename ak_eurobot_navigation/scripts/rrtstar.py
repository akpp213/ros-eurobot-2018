#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Polygon
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool
from sensor_msgs.msg import PointCloud
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt


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

        self.RESOLUTION = 0.05

        self.new_path = rospy.Publisher('rrt_path', Polygon, queue_size=1)
        self.viz_pub = rospy.Publisher("visualization_msgs/Marker", Marker, queue_size=3)
        self.viz_pub2 = rospy.Publisher("visualization_msgs/Marker2", Marker, queue_size=3)
        self.no_path = rospy.Publisher("no_path_found", Bool, queue_size=1)
        rospy.Subscriber('current_coords', Point, self.update_coords, queue_size=1)
        rospy.Subscriber('new_goal_loc', Point, self.find_path_rrtstar, queue_size=1)
        rospy.Subscriber("/main_robot/map", OccupancyGrid, self.update_map, queue_size=3)
        rospy.Subscriber("/secondary_robot/map", OccupancyGrid, self.update_map, queue_size=3)
        rospy.Subscriber("/map_server/opponent_robots", PointCloud, self.detected_robots_callback, queue_size=1)
        # rospy.Subscriber("lookahead_pnts", Point, self.add_look, queue_size=1)

    def update_coords(self, msg):
        self.path = []
        self.coords[0] = msg.x
        self.coords[1] = msg.y
        self.coords[2] = msg.z
        self.map_updated = False

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
            self.permissible_region = np.ones_like(array255, dtype=bool)
            self.permissible_region[array255 == 100] = 0  # setting occupied regions (100) to 0. Unoccupied regions are a 1
            self.map_updated = True
            print "Map Updated"

    def detected_robots_callback(self, data):
        if len(data.points) == 0:
            self.opponent_robots = np.array([])
        else:
            self.opponent_robots = np.array([[robot.x, robot.y] for robot in data.points])
        self.robots_upd_time = data.header.stamp

    def find_path_rrtstar(self, msg):
        while not self.map_updated:
            pass
            # print "MAP NOT UPDATED YET"
        self.goal[0] = msg.x
        self.goal[1] = msg.y
        self.goal[2] = msg.z

        # plt.plot((self.coords[0]-self.x0) / self.resolution, (self.coords[1]-self.y0) / self.resolution, 'ob')
        # plt.plot((self.goal[0]-self.x0) / self.resolution, (self.goal[1]-self.y0) / self.resolution, 'or')
        # plt.pause(.001)
        # plt.imshow(self.permissible_region)
        # plt.pause(0.001)

        x, y, th = self.coords
        start_node = Node(x, y)
        self.path = []
        self.nodes = [start_node]
        j = 0
        path_found = False
        #plt.axis([self.x0, self.x0 + self.map_width, self.y0 + self.map_height, self.y0])
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
                # near_idxs = self.find_near_nodes(x_new)

                # find the best parent for the node
                # x_new = self.choose_parent(x_new, near_idxs)

                # plt.plot([(x_new.x-self.x0) / self.resolution, (self.nodes[x_new.parent].x-self.x0) / self.resolution],
                #          [(x_new.y-self.y0) / self.resolution, (self.nodes[x_new.parent].y-self.y0) / self.resolution])
                # plt.pause(.0001)

                # add new node and rewire
                self.nodes.append(x_new)
                # self.rewire(x_new, near_idxs)

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
                    self.path = pd.unique(np.array(path))
                    print self.path, "path"
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
        self.no_path.publish(True)

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

        theta = np.arctan2(rnd[1] - x_near.y, rnd[0] - x_near.x)
        new_x = x_near.x + self.STEP * np.cos(theta)
        new_y = x_near.y + self.STEP * np.sin(theta)

        x_new = Node(new_x, new_y)
        x_new.cost += self.STEP

        return x_new, theta

    def obstacle_free(self, nearest_node, new_node, theta):
        """ Checks if the path from x_near to x_new is obstacle free """

        dx = np.sin(theta) * self.PATH_WIDTH
        dy = np.cos(theta) * self.PATH_WIDTH

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

        if self.line_collision(bound0) or self.line_collision(bound1) or self.line_collision(bound2):# or self.line_collision(bound3) or self.line_collision(bound4):
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
        # plt.plot(x_ind-self.x0/self.resolution, y_ind-self.y0/self.resolution)
        # plt.pause(.001)
        for i in range(len(x_ind)):
            row = min([int(-self.y0 / self.resolution + y_ind[i]), self.shape_map[1] - 1])
            column = min([int(-self.x0 / self.resolution + x_ind[i]), self.shape_map[0] - 1])
            # plt.plot(column, row, 'or')
            # print row, column, "rowcol"
            # print self.map_width, self.map_height
            if not self.permissible_region[row, column]:
                # print "returning true for a line collision"
                return True
        return False

    def find_near_nodes(self, x_new):
        length_nodes = len(self.nodes)
        r = self.GAMMA_RRT * np.sqrt((np.log(length_nodes) / length_nodes))
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
            theta = np.arctan2(dy, dx)

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
                theta = np.arctan2(dy, dx)
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
        path = np.array(path)
        self.visualize(2)
        result = self.bezier_path(path[0], path[path.shape[0]/2], path[-1])
        interval = int(np.ceil(path.shape[0]/2.0))-1
        while result is None and interval > 2:
            print interval, "interval"
            for i in xrange(0, path.shape[0] - interval, interval):
                inter = interval
                if i+inter >= path.shape[0] - interval:
                    inter = path.shape[0] - i - 1
                new_result = self.bezier_path(path[i], path[(i+inter)/2], path[i+inter])
                if new_result is None:
                    result = None
                    interval /= 2
                    break
                else:
                    # print new_result, "new result"
                    if i == 0:
                        result = new_result
                    else:
                        result = np.concatenate((result[:i], new_result))
        if result is not None:
            return result
        print "No Bezier Path Found"
        return path

    def bezier_path(self, P0, P1, P2):
        max_len = int(((np.linalg.norm(P1 - P0) + np.linalg.norm(P2 - P1))) / self.RESOLUTION)
        if not max_len:
            return None
        # path = np.zeros((max_len, 2))
        t = 0
        dt = 1.0 / (5 * max_len)
        # path[0] = P0
        path = np.array([P0])
        i = 1

        while t < 1:
            t += dt
            potential_point = self.bezier_curve(P0, P1, P2, t)
            if np.linalg.norm(path[i - 1] - potential_point) >= self.RESOLUTION:
                theta = np.arctan2(potential_point[1] - path[i-1, 1], potential_point[0] - path[i-1, 0])
                if self.nodeless_obstacle_free(path[i-1], potential_point, theta):
                    path = np.append(path, [potential_point], axis=0)
                    i += 1
                    if np.linalg.norm(P2 - potential_point) < self.RESOLUTION:
                        break
                else:
                    return None
        path = np.append(path, [P2], axis=0)
        n = i + 1
        # path[:n, 2] = np.linspace(P0[2], P2[2], n)

        return path[:n]

    @staticmethod
    def bezier_curve(P0, P1, P2, t):
        return (1 - t) ** 2 * P0[:2] + 2 * t * (1 - t) * P1[:2] + t ** 2 * P2[:2]

    def nodeless_obstacle_free(self, nearest_node, new_node, theta):
        """ Checks if the path from x_near to x_new is obstacle free """

        dx = np.sin(theta) * self.PATH_WIDTH
        dy = np.cos(theta) * self.PATH_WIDTH
        # plt.plot([(new_node[0] - self.x0) / self.resolution, (nearest_node[0] - self.x0) / self.resolution],
        #          [(new_node[1] - self.y0) / self.resolution, (nearest_node[1] - self.y0) / self.resolution])
        # plt.pause(.001)

        if nearest_node[0] < new_node[0]:
            bound0 = ((nearest_node[0], nearest_node[1]), (new_node[0], new_node[1]))
            bound1 = ((nearest_node[0] + dx, nearest_node[1] - dy), (new_node[0] + dx, new_node[1] - dy))
            bound2 = ((nearest_node[0] - dx, nearest_node[1] + dy), (new_node[0] - dx, new_node[1] + dy))
            dx*=.5
            dy*=.5
            bound3 = ((nearest_node[0] + dx, nearest_node[1] - dy), (new_node[0] + dx, new_node[1] - dy))
            bound4 = ((nearest_node[0] - dx, nearest_node[1] + dy), (new_node[0] - dx, new_node[1] + dy))
        else:
            bound0 = ((new_node[0], new_node[1]), (nearest_node[0], nearest_node[1]))
            bound1 = ((new_node[0] + dx, new_node[1] - dy), (nearest_node[0] + dx, nearest_node[1] - dy))
            bound2 = ((new_node[0] - dx, new_node[1] + dy), (nearest_node[0] - dx, nearest_node[1] + dy))
            dx*=.5
            dy*=.5
            bound3 = ((new_node[0] + dx, new_node[1] - dy), (nearest_node[0] + dx, nearest_node[1] - dy))
            bound4 = ((new_node[0] - dx, new_node[1] + dy), (nearest_node[0] - dx, nearest_node[1] + dy))

        if self.line_collision(bound0) or self.line_collision(bound1) or self.line_collision(bound2):# or self.line_collision(bound3) or self.line_collision(bound4):
            return False
        else:
            return True

    @staticmethod
    def to_poly(path):
        poly = Polygon()
        poly.points = []
        if path is None:
            return poly
        try:
            for pt in path:
                point = Point()
                point.x = pt[0]
                point.y = pt[1]
                point.z = 0.0
                poly.points.append(point)
            return poly
        except TypeError:
            return poly

    def visualize(self, topic=1):
        if self.nodes:
            line_strip = Marker()
            line_strip.type = line_strip.LINE_STRIP
            line_strip.action = line_strip.ADD
            line_strip.header.frame_id = "/map"

            line_strip.scale.x = 0.05

            line_strip.color.a = 1.0
            line_strip.color.r = 0.0
            line_strip.color.g = 0.0
            line_strip.color.b = 1.0

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

            # for node in self.nodes:
                # if node.parent is not None:
                    # parent = self.nodes[node.parent]

                    # plt.plot([node.x, parent.x], [node.y, parent.y], "-g")

                    # theta = np.arctan2(node.y - parent.y, node.x - parent.x)

                    # dx = np.sin(theta) * self.PATH_WIDTH
                    # dy = np.cos(theta) * self.PATH_WIDTH

                    # plt.plot([parent.x + dx, node.x + dx], [parent.y - dy, node.y - dy], "-g")
                    # plt.plot([parent.x - dx, node.x - dx], [parent.y + dy, node.y + dy], "-g")

            # x, y = self.coords[:2]
            # plt.plot(x, y, "ob")
            # x, y = self.goal[:2]
            # plt.plot(x, y, "or")
            # plt.axis([self.x0, self.x0 + self.map_width, self.y0 + self.map_height, self.y0])

            # plt.imshow(self.permissible_region, extent=[self.x0, self.x0 + self.map_width, self.y0 + self.map_height, self.y0])
            # plt.pause(0.01)

            for node in self.path:
                new_point = Point()
                new_point.x = node[0]
                new_point.y = node[1]
                new_point.z = 0.0
                line_strip.points.append(new_point)

            # Publish the Marker
            if topic == 2:
                print "SHOWING TWO PATHS"
                line_strip.color.r = 1.0
                self.viz_pub2.publish(line_strip)
            self.viz_pub.publish(line_strip)


if __name__ == "__main__":
    rrtstar = RRTStar()

rospy.spin()
