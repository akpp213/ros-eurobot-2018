#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Polygon
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool
from sensor_msgs.msg import PointCloud
import numpy as np
# import pandas as pd
from threading import Lock
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
        self.STEP_MIN = rospy.get_param("motion_planner/STEP_MIN")
        self.STEP_MAX = rospy.get_param("motion_planner/STEP_MAX")
        self.GAMMA_RRT = rospy.get_param("motion_planner/GAMMA_RRT")  # 3
        self.EPSILON_GOAL = rospy.get_param("motion_planner/EPSILON_GOAL")  # 0.2
        self.PATH_WIDTH = rospy.get_param("motion_planner/PATH_WIDTH")  # 0.1
        self.PATH_WIDTH_SMALL = rospy.get_param("motion_planner/PATH_WIDTH_SMALL")
        self.PATH_WIDTH_LARGE = rospy.get_param("motion_planner/PATH_WIDTH_LARGE")
        self.PATH_WIDTH_MAX = rospy.get_param("motion_planner/PATH_WIDTH_MAX")
        self.robot_name = rospy.get_param("robot_name")
        self.path_width = self.PATH_WIDTH_SMALL if self.robot_name == "main_robot" else self.PATH_WIDTH_LARGE
        print self.path_width, "path width"
        self.GOAL_SAMPLE_RATE = rospy.get_param("motion_planner/GOAL_SAMPLE_RATE")
        self.coords = np.zeros(3)
        self.goal = None

        self.mutex = Lock()

        self.find_bezier = False

        self.nodes = []
        # self.nodes_secondary = None
        self.path = []
        self.turn_angle = None
        self.find_bezier = False

        self.opponent_robots = None
        self.robots_upd_time = None

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
        rospy.Subscriber('new_goal_loc', Point, self.update_goal, queue_size=1)
        # rospy.Subscriber("/main_robot/map", OccupancyGrid, self.update_map, queue_size=3)
        rospy.Subscriber("map", OccupancyGrid, self.update_map, queue_size=3)
        rospy.Subscriber("/map_server/opponent_robots", PointCloud, self.detected_robots_callback, queue_size=1)
        # rospy.Subscriber("lookahead_pnts", Point, self.add_look, queue_size=1)

    def update_coords(self, msg):
        self.path = []
        self.coords[0] = msg.x
        self.coords[1] = msg.y
        self.coords[2] = msg.z
        # self.goal = None
        self.map_updated = False

    def update_goal(self, msg):
        if msg.x == -1 and msg.y == -1 and msg.z == -1:
            self.goal = None
        else:
            self.goal = np.zeros(3)
            self.goal[0] = msg.x
            self.goal[1] = msg.y
            self.goal[2] = msg.z
            if self.map_updated and (self.path is None or not len(self.path)):
                self.find_path_rrtstar()

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
            self.permissible_region[array255 == 100] = 0  # set occupied regions (100) to 0 and unoccupied regions to 1
            self.map_updated = True
            print "Map Updated"
            if self.goal is not None and (self.path is None or not len(self.path)):
                self.find_path_rrtstar()

    def detected_robots_callback(self, data):
        if len(data.points) == 0:
            self.opponent_robots = np.array([])
        else:
            self.opponent_robots = np.array([[robot.x, robot.y] for robot in data.points])
        self.robots_upd_time = data.header.stamp

    def find_path_rrtstar(self):
        self.mutex.acquire()
        if self.goal is None:
            print "No Goal"
            self.mutex.release()
            return
        print "Planning New Path"
        # plt.clf()
        # plt.plot((self.coords[0]-self.x0) / self.resolution, (self.coords[1]-self.y0) / self.resolution, 'ob')
        # plt.plot((self.goal[0]-self.x0) / self.resolution, (self.goal[1]-self.y0) / self.resolution, 'or')
        # plt.imshow(self.permissible_region)
        # plt.pause(0.001)

        x, y, th = self.coords
        start_node = Node(x, y)
        self.path = []
        self.nodes = [start_node]
        j = 0
        path_found = False
        # plt.axis([self.x0, self.x0 + self.map_width, self.y0 + self.map_height, self.y0])
        row = min([int(-self.y0 / self.resolution + self.goal[1]), self.shape_map[1] - 1])
        column = min([int(-self.x0 / self.resolution + self.goal[0]), self.shape_map[0] - 1])
        if self.permissible_region[row, column]:
            while not path_found and (j < self.MAX_RRT_ITERS):

                # random sample
                if j % self.GOAL_SAMPLE_RATE == (self.GOAL_SAMPLE_RATE-1):
                    # sample goal location every 10 iterations
                    rnd = [self.goal[0], self.goal[1]]
                else:
                    rnd = [self.x0 + self.map_width * np.random.uniform(),
                           self.y0 + self.map_height * np.random.uniform()]

                # plt.plot(rnd[0], rnd[1], 'og')
                # find the nearest node
                nn_idx = self.nearest_node(rnd)
                x_near = self.nodes[nn_idx]

                # calculate step size based on distance from goal
                dist = np.linalg.norm([self.goal[0] - x_near.x, self.goal[1] - x_near.y])
                step = max(min(dist/2, self.STEP_MAX), self.STEP_MIN)

                # expand the tree
                x_new, theta = self.steer_to_node(x_near, rnd, step)  # goes from x_near to x_new and points towards rnd
                x_new.parent = nn_idx

                # print "length nodes"
                # print j
                # print ""
                if not j % 50:
                    print j

                # check for obstacle collisions
                if x_near.parent is None:
                    angle_fraction = 2*abs((self.coords[2] - theta + np.pi/2) % np.pi - np.pi/2)/np.pi
                else:
                    prev_node = self.nodes[x_near.parent]
                    old_theta = np.arctan2(x_near.y - prev_node.y, x_near.x - prev_node.x)
                    angle_fraction = 2*abs((old_theta - theta + np.pi/2) % np.pi - np.pi/2)/np.pi
                start_width = (self.PATH_WIDTH_MAX - self.PATH_WIDTH_SMALL) * angle_fraction + self.PATH_WIDTH_SMALL
                # print start_path_width, "START PATH WIDTH"
                if self.obstacle_free(x_near, x_new, theta, start_width, step):
                    # plt.plot([x_near.x, x_new.x], [x_new.y, x_new.y])

                    # find close nodes
                    near_idxs = self.find_near_nodes(x_new, step*2)
                    # near_idxs = np.arange(len(self.nodes))

                    # find the best parent for the node
                    x_new = self.choose_parent(x_new, near_idxs, step)

                    # plt.plot([(x_new.x-self.x0) / self.resolution, (self.nodes[x_new.parent].x-self.x0)
                    #           / self.resolution],
                    #          [(x_new.y-self.y0) / self.resolution, (self.nodes[x_new.parent].y-self.y0)
                    #           / self.resolution], 'g-')
                    # plt.pause(.0001)

                    # add new node and rewire
                    self.nodes.append(x_new)
                    # self.rewire(x_new, near_idxs, step)

                    # check if sample point is in goal region
                    dx = x_new.x - self.goal[0]
                    dy = x_new.y - self.goal[1]
                    d = np.sqrt(dx ** 2 + dy ** 2)
                    # print self.EPSILON_GOAL
                    if d <= self.EPSILON_GOAL:
                        print "found goal! took", j, "iterations"
                        # construct path
                        # last_idx = self.get_last_idx()
                        last_idx = -1
                        if last_idx is None:
                            self.mutex.release()
                            return
                        # path = self.get_path(last_idx)
                        # self.path = pd.unique(np.array(path))
                        self.path = self.get_path(last_idx)
                        print self.path, "path"
                        self.new_path.publish(self.to_poly(self.path))
                        self.visualize()
                        self.mutex.release()
                        return

                        # for point in self.path:
                        # 	pt_obj = Point()
                        # 	pt_obj.x = point[0]
                        # 	pt_obj.y = point[1]

                        # 	self.trajectory.addPoint(pt_obj)
                        # self.publish_trajectory()

                        # return self.path

                j += 1
        else:
            print "GOAL in an Obstacle"

        self.visualize()
        print "NO PATH FOUND IN", self.MAX_RRT_ITERS, "ITERATIONS"
        self.new_path.publish(self.to_poly(self.path))
        # self.no_path.publish(True)
        self.mutex.release()

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

    @staticmethod
    def steer_to_node(x_near, rnd, step):
        """ Steers from x_near to a point on the line between rnd and x_near """

        theta = np.arctan2(rnd[1] - x_near.y, rnd[0] - x_near.x)
        new_x = x_near.x + step * np.cos(theta)
        new_y = x_near.y + step * np.sin(theta)

        x_new = Node(new_x, new_y)
        x_new.cost += step

        return x_new, theta

    def obstacle_free(self, nearest_node, new_node, theta, start_path_width, step, plot=False):
        """ Checks if the path from x_near to x_new is obstacle free """

        dx_end = np.sin(theta) * self.path_width
        dy_end = np.cos(theta) * self.path_width
        dx_start = np.sin(theta) * start_path_width
        dy_start = np.cos(theta) * start_path_width

        mult = .5 * step / self.STEP_MAX

        bound0 = ((nearest_node.x, nearest_node.y), (new_node.x, new_node.y))
        bound1 = ((nearest_node.x + dx_start - dy_start*mult, nearest_node.y - dy_start - dx_start*mult),
                  (new_node.x + dx_end + dy_end*mult, new_node.y - dy_end + dx_end*mult))
        bound2 = ((nearest_node.x - dx_start - dy_start*mult, nearest_node.y + dy_start - dx_start*mult),
                  (new_node.x - dx_end + dy_end*mult, new_node.y + dy_end + dx_end*mult))
        if self.line_collision(bound0, plot) or self.line_collision(bound1, plot) or self.line_collision(bound2, plot):
            return False
        if start_path_width > .15:
            dx_start *= .5
            dy_start *= .5
            dx_end *= .5
            dy_end *= .5
            bound3 = ((nearest_node.x + dx_start - dy_start*mult, nearest_node.y - dy_start - dx_start*mult),
                      (new_node.x + dx_end + dy_end*mult, new_node.y - dy_end + dx_end*mult))
            bound4 = ((nearest_node.x - dx_start - dy_start*mult, nearest_node.y + dy_start - dx_start*mult),
                      (new_node.x - dx_end + dy_end*mult, new_node.y + dy_end + dx_end*mult))
            if self.line_collision(bound3, plot) or self.line_collision(bound4, plot):
                return False
        return True

    def line_collision(self, line, plot=False):
        """ Checks if line collides with obstacles in the map"""

        # discretize values of x and y along line according to map using Bresemham's alg
        if abs(line[1][0] - line[0][0]) > abs(line[1][1] - line[0][1]):
            dx = np.ceil(line[1][0] / self.resolution) - np.ceil(line[0][0] / self.resolution)
            sign = np.sign(dx) if dx != 0.0 else 1
            dx = sign*max(1, abs(dx))
            dy = np.ceil(line[1][1] / self.resolution) - np.ceil(line[0][1] / self.resolution)
            deltaerr = abs(float(dy) / dx)
            error = .0

            x_ind = np.arange(np.ceil((line[0][0]) / self.resolution),
                              np.ceil((line[1][0]) / self.resolution + sign * 1), sign * 2, dtype=int)
            y_ind = np.zeros(x_ind.shape[0])

            y = int(np.ceil((line[0][1]) / self.resolution))
            for i in xrange(x_ind.shape[0]):
                y_ind[i] = y
                error = error + 2*deltaerr
                while error >= 0.5:
                    y += np.sign(dy) * 1
                    error -= 1
            # y_ind = np.array(y_ind)
            # check if any cell along the line contains an obstacle
            for i in range(len(x_ind)):
                row = min([int(-self.y0 / self.resolution + y_ind[i]), self.shape_map[1] - 1])
                column = min([int(-self.x0 / self.resolution + x_ind[i]), self.shape_map[0] - 1])
                if plot:
                    plt.plot(column, row, 'go')
                    plt.pause(.05)
                # print self.map_width, self.map_height
                if not self.permissible_region[row, column]:
                    if plot:
                        plt.plot(column, row, 'ro')
                        plt.pause(.05)
                    # print "returning true for a line collision"
                    # plt.plot(x_ind - self.x0 / self.resolution, y_ind - self.y0 / self.resolution, 'r-')
                    # plt.pause(.0001)
                    return True
            if plot:
                plt.plot(x_ind - self.x0 / self.resolution, y_ind - self.y0 / self.resolution)
                plt.pause(.1)

            return False

        else:
            dy = np.ceil(line[1][1] / self.resolution) - np.ceil(line[0][1] / self.resolution)
            sign = np.sign(dy) if dy != 0.0 else 1
            dy = sign*max(1, abs(dy))
            dx = np.ceil(line[1][0] / self.resolution) - np.ceil(line[0][0] / self.resolution)
            deltaerr = abs(float(dx) / dy)
            error = .0

            y_ind = np.arange(np.ceil((line[0][1]) / self.resolution),
                              np.ceil((line[1][1]) / self.resolution + sign * 1), sign * 2, dtype=int)
            x_ind = np.zeros(y_ind.shape[0])

            x = int(np.ceil((line[0][0]) / self.resolution))
            for i in xrange(y_ind.shape[0]):
                x_ind[i] = x
                error = error + 2 * deltaerr
                while error >= 0.5:
                    x += np.sign(dx) * 1
                    error -= 1
            # x_ind = np.array(x_ind)
            # check if any cell along the line contains an obstacle
            for i in range(len(y_ind)):
                row = min([int(-self.y0 / self.resolution + y_ind[i]), self.shape_map[1] - 1])
                column = min([int(-self.x0 / self.resolution + x_ind[i]), self.shape_map[0] - 1])
                if plot:
                    plt.plot(column, row, 'go')
                    plt.pause(.05)
                # print self.map_width, self.map_height
                if not self.permissible_region[row, column]:
                    if plot:
                        plt.plot(column, row, 'ro')
                        plt.pause(.05)
                    # print "returning true for a line collision"
                    # plt.plot(x_ind - self.x0 / self.resolution, y_ind - self.y0 / self.resolution, 'r-')
                    # plt.pause(.0001)
                    return True
            if plot:
                plt.plot(x_ind - self.x0 / self.resolution, y_ind - self.y0 / self.resolution)
                plt.pause(.1)

            return False

    def find_near_nodes(self, x_new, step):
        # length_nodes = len(self.nodes)
        # r = self.GAMMA_RRT * np.sqrt((np.log(length_nodes) / length_nodes))
        near_idxs = []
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            # if node.distance((x_new.x, x_new.y)) <= r:
            if node.distance((x_new.x, x_new.y)) <= step:
                near_idxs.append(i)
        return near_idxs

    def choose_parent(self, x_new, near_idxs, step):
        if len(near_idxs) == 0 or len(self.nodes) == 1:
            return x_new

        # distances = []
        # for i in near_idxs:
        #     node = self.nodes[i]
        #     d = node.distance((x_new.x, x_new.y))
        #     dx = x_new.x - node.x
        #     dy = x_new.y - node.y
        #     theta = np.arctan2(dy, dx)

            # if node.parent is None:
            #     angle_fraction = 2*abs((self.coords[2] - theta + np.pi/2) % np.pi - np.pi/2)/np.pi
            # else:
            #     prev_node = self.nodes[node.parent]
            #     old_theta = np.arctan2(node.y - prev_node.y, node.x - prev_node.x)
            #     angle_fraction = 2*abs((old_theta - theta + np.pi/2) % np.pi - np.pi/2)/np.pi
            # start_path_width = (self.PATH_WIDTH_MAX - self.PATH_WIDTH_SMALL) * angle_fraction + self.PATH_WIDTH_SMALL
            # print start_path_width, "START PATH WIDTH"
        #
        #     if self.obstacle_free(node, x_new, theta):
        #         distances.append(d)
        #     else:
        #         distances.append(float("inf"))
        # min_cost = min(distances)
        # min_ind = near_idxs[distances.index(min_cost)]
        #
        # if min_cost == float("inf"):
        #     return x_new
        #
        # x_new.cost = min_cost
        # x_new.parent = min_ind

        d_angles = []
        for i in near_idxs:
            node = self.nodes[i]
            dx = x_new.x - node.x
            dy = x_new.y - node.y
            theta_new = np.arctan2(dy, dx)

            if node.parent is None:
                # d_angles.append(float("inf"))
                # continue
                delta = abs((self.coords[2] - theta_new + np.pi/2) % np.pi - np.pi/2)
                angle_fraction = 2*delta/np.pi
                # delta = abs((self.coords[2] - theta_new + np.pi) % (2 * np.pi) - np.pi)
            else:
                prev_node = self.nodes[node.parent]
                old_theta = np.arctan2(node.y - prev_node.y, node.x - prev_node.x)
                delta = abs((old_theta - theta_new + np.pi) % (2 * np.pi) - np.pi)
                angle_fraction = 2*abs((old_theta - theta_new + np.pi/2) % np.pi - np.pi/2)/np.pi
            start_path_width = (self.PATH_WIDTH_MAX - self.PATH_WIDTH_SMALL) * angle_fraction + self.PATH_WIDTH_SMALL
            # print start_path_width, "START PATH WIDTH"

            if self.obstacle_free(node, x_new, theta_new, start_path_width, step):
                d_angles.append(delta)
            else:
                d_angles.append(float("inf"))

        min_delta_angle = min(d_angles)
        min_ind = near_idxs[d_angles.index(min_delta_angle)]

        if min_delta_angle == float("inf"):
            return x_new

        x_new.cost = self.nodes[min_ind].distance((x_new.x, x_new.y))
        x_new.parent = min_ind

        return x_new

    def rewire(self, x_new, near_idxs, step):
        n = len(self.nodes)
        for i in near_idxs:
            node = self.nodes[i]
            d = node.distance((x_new.x, x_new.y))
            scost = x_new.cost + d

            if node.cost > scost:
                dx = x_new.x - node.x
                dy = x_new.y - node.y
                theta = np.arctan2(dy, dx)

                if node.parent is None:
                    angle_fraction = 2 * abs((self.coords[2] - theta + np.pi / 2) % np.pi - np.pi / 2) / np.pi
                else:
                    prev_node = self.nodes[node.parent]
                    old_theta = np.arctan2(node.y - prev_node.y, node.x - prev_node.x)
                    angle_fraction = 2 * abs((old_theta - theta + np.pi / 2) % np.pi - np.pi / 2) / np.pi
                start_width = (self.PATH_WIDTH_MAX - self.PATH_WIDTH_SMALL) * angle_fraction + self.PATH_WIDTH_SMALL
                # print start_path_width, "START PATH WIDTH"

                if self.obstacle_free(node, x_new, theta, start_width, step, True):
                    node.parent = n - 1
                    node.cost = scost
                    plt.plot((np.array([node.x, self.nodes[n-1].x]) - self.x0) / self.resolution,
                             (np.array([node.y, self.nodes[n-1].y]) - self.y0) / self.resolution, 'r')
                    plt.pause(.1)

    def get_last_idx(self):
        dlist = []
        for node in self.nodes:
            d = node.distance((self.goal[0], self.goal[1]))
            dlist.append(d)
        goal_idxs = [dlist.index(i) for i in dlist if i <= self.STEP]

        if len(goal_idxs) == 0:
            return None

        min_cost = min([self.nodes[i].cost for i in goal_idxs])

        for i in goal_idxs:
            if self.nodes[i].cost == min_cost:
                return i

        return None

    def get_path(self, last_idx):
        path = [(self.goal[0], self.goal[1])]
        while self.nodes[last_idx].parent is not None and (self.nodes[last_idx].x, self.nodes[last_idx].y) not in path:
            node = self.nodes[last_idx]
            path.append((node.x, node.y))
            last_idx = node.parent
        if (self.coords[0], self.coords[1]) not in path:
            path.append((self.coords[0], self.coords[1]))
        path.reverse()
        path = np.array(path)
        if self.find_bezier:
            # self.visualize(2)
            result = self.bezier_path(path[0], path[path.shape[0]/2], path[-1], self.path_width)
            interval = int(np.ceil(path.shape[0]/2.0))-1
            while result is None and interval > 2:
                print interval, "interval"
                for i in xrange(0, path.shape[0] - interval, interval):
                    inter = interval
                    if i+inter >= path.shape[0] - interval:
                        inter = path.shape[0] - i - 1
                    new_result = self.bezier_path(path[i], path[(i+inter)/2], path[i+inter], self.path_width)
                    if new_result is None:
                        result = None
                        interval /= 2
                        break
                    else:
                        # print new_result, "new result"
                        if i == 0:
                            result = new_result
                        else:
                            result = np.concatenate((result, new_result))
            if result is not None:
                print "Found bezier path"
                return result
            print "No Bezier Path Found"
        return path

    # def update_path_width(self, direct):
    #     pass
        # self.path_width = new_pw

    def bezier_path(self, p0, p1, p2, step):
        max_len = int((np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1)) / self.RESOLUTION)
        if not max_len:
            return None
        # path = np.zeros((max_len, 2))
        t = 0
        dt = 1.0 / (5 * max_len)
        # path[0] = P0
        path = np.array([p0])
        i = 1

        while t < 1:
            t += dt
            potential_point = self.bezier_curve(p0, p1, p2, t)
            if np.linalg.norm(path[i - 1] - potential_point) >= self.RESOLUTION:
                theta = np.arctan2(potential_point[1] - path[i-1, 1], potential_point[0] - path[i-1, 0])
                if self.nodeless_obstacle_free(path[i-1], potential_point, theta, self.path_width, step, True):
                    path = np.append(path, [potential_point], axis=0)
                    i += 1
                    if np.linalg.norm(p2 - potential_point) < self.RESOLUTION:
                        break
                else:
                    return None
        path = np.append(path, [p2], axis=0)
        n = i + 1
        # path[:n, 2] = np.linspace(P0[2], P2[2], n)

        return path[:n]

    @staticmethod
    def bezier_curve(p0, p1, p2, t):
        return (1 - t) ** 2 * p0[:2] + 2 * t * (1 - t) * p1[:2] + t ** 2 * p2[:2]

    def nodeless_obstacle_free(self, nearest_node, new_node, theta, start_path_width, step, plot=False):
        """ Checks if the path from x_near to x_new is obstacle free """

        dx_end = np.sin(theta) * self.path_width
        dy_end = np.cos(theta) * self.path_width
        dx_start = np.sin(theta) * start_path_width
        dy_start = np.cos(theta) * start_path_width

        mult = .5 * step / self.STEP_MAX

        bound0 = ((nearest_node[0], nearest_node[1]), (new_node[0], new_node[1]))
        bound1 = ((nearest_node[0] + dx_start - dy_start*mult, nearest_node[1] - dy_start - dx_start*mult),
                  (new_node[0] + dx_end + dy_end*mult, new_node[1] - dy_end + dx_end*mult))
        bound2 = ((nearest_node[0] - dx_start - dy_start*mult, nearest_node[1] + dy_start - dx_start*mult),
                  (new_node[0] - dx_end + dy_end*mult, new_node[1] + dy_end + dx_end*mult))
        if self.line_collision(bound0, plot) or self.line_collision(bound1, plot) or self.line_collision(bound2, plot):
            return False
        if start_path_width > .15:
            dx_start *= .5
            dy_start *= .5
            dx_end *= .5
            dy_end *= .5
            bound3 = ((nearest_node[0] + dx_start - dy_start*mult, nearest_node[1] - dy_start - dx_start*mult),
                      (new_node[0] + dx_end + dy_end*mult, new_node[1] - dy_end + dx_end*mult))
            bound4 = ((nearest_node[0] - dx_start - dy_start*mult, nearest_node[1] + dy_start - dx_start*mult),
                      (new_node[0] - dx_end + dy_end*mult, new_node[1] + dy_end + dx_end*mult))
            if self.line_collision(bound3, plot) or self.line_collision(bound4, plot):
                return False
        return True

    @staticmethod
    def to_poly(path):
        poly = Polygon()
        poly.points = []
        if path is None or not len(path):
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

    def visualize(self):
        if self.nodes:
            line_strip = Marker()
            line_strip.type = line_strip.LINE_STRIP
            line_strip.action = line_strip.ADD
            line_strip.header.frame_id = "/map"

            line_strip.scale.x = 0.05

            line_strip.color.a = 1.0
            line_strip.color.r = 0.0
            line_strip.color.g = 1.0 if self.robot_name == "main_robot" else 0.0
            line_strip.color.b = 0.0 if self.robot_name == "main_robot" else 1.0

            # marker orientation
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

            '''
            for node in self.nodes:
                if node.parent is not None:
                    parent = self.nodes[node.parent]

                    plt.plot([node.x, parent.x], [node.y, parent.y], "-g")

                    theta = np.arctan2(node.y - parent.y, node.x - parent.x)

                    dx = np.sin(theta) * self.path_width
                    dy = np.cos(theta) * self.path_width

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
            '''

            for node in self.path:
                new_point = Point()
                new_point.x = node[0]
                new_point.y = node[1]
                new_point.z = 0.0
                line_strip.points.append(new_point)

            self.viz_pub.publish(line_strip)


if __name__ == "__main__":
    rrtstar = RRTStar()

rospy.spin()
