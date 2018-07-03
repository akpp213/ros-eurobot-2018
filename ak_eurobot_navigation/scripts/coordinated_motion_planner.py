#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Polygon
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool
from sensor_msgs.msg import PointCloud
import numpy as np
import pandas as pd
from threading import Lock
import matplotlib.pyplot as plt


class Node(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.time_step = 0
        self.cost = 0.0
        self.total_cost = 0.0

    def distance(self, random_point):
        distance = np.sqrt((self.x - random_point[0]) ** 2 + (self.y - random_point[1]) ** 2)
        return distance


class CoordinatedMotionPlanner:
    def __init__(self):
        rospy.init_node("rrtstar", anonymous=True)
        self.MAX_RRT_ITERS = rospy.get_param("coordinated_motion_planner/MAX_RRT_ITERS")  # 1000
        self.STEP = rospy.get_param("coordinated_motion_planner/STEP")  # 0.1
        self.STEP_MIN = rospy.get_param("coordinated_motion_planner/STEP_MIN")
        self.STEP_MAX = rospy.get_param("coordinated_motion_planner/STEP_MAX")
        self.GAMMA_RRT = rospy.get_param("coordinated_motion_planner/GAMMA_RRT")  # 3
        self.EPSILON_GOAL = rospy.get_param("coordinated_motion_planner/EPSILON_GOAL")  # 0.2
        self.PATH_WIDTH = rospy.get_param("coordinated_motion_planner/PATH_WIDTH")  # 0.1
        self.PATH_WIDTH_SMALL = rospy.get_param("coordinated_motion_planner/PATH_WIDTH_SMALL")
        self.PATH_WIDTH_LARGE = rospy.get_param("coordinated_motion_planner/PATH_WIDTH_LARGE")
        self.PATH_WIDTH_MAX = rospy.get_param("coordinated_motion_planner/PATH_WIDTH_MAX")
        self.path_width = self.PATH_WIDTH_SMALL
        self.GOAL_SAMPLE_RATE = rospy.get_param("coordinated_motion_planner/GOAL_SAMPLE_RATE")
        self.TIME_RESOLUTION = rospy.get_param("coordinated_motion_planner/TIME_RESOLUTION")
        self.coords = np.zeros(3)
        self.goal = None
        self.other_goal = np.zeros(3)

        self.color = rospy.get_param("/field/color")
        self.size_main = np.array([rospy.get_param('/main_robot/dim_x'), rospy.get_param('/main_robot/dim_y')]) / 1000
        self.radius_main = rospy.get_param('/main_robot/dim_r')
        self.size_secondary = np.array([rospy.get_param('/secondary_robot/dim_x'), rospy.get_param('/secondary_robot/dim_y')]) / 1000
        self.radius_secondary = rospy.get_param('/secondary_robot/dim_r')

        self.UNCERTAINTY_BUFFER = 1.5#rospy.get_param("motion_planner/UNCERTAINTY_BUFFER")
        self.robot_name = rospy.get_param("robot_name")
        self.other_robot_name = "secondary_robot" if self.robot_name == "main_robot" else "main_robot"
        self.other_dims = np.array([rospy.get_param("/"+self.other_robot_name+"/dim_x"), rospy.get_param("/"+self.other_robot_name+"/dim_y")])/1000.0
        self.other_coords = rospy.get_param("/"+self.other_robot_name+"/start_"+self.color)
        self.OTHER_DESIRED_DIRECS = np.array(rospy.get_param("/"+self.other_robot_name+"/motion_planner/DESIRED_DIRECTIONS_OF_TRAVEL"))
        # self.other_desired_direction = 0.0
        self.other_path_found = False
        self.other_path = []
        self.other_angles = None
        self.other_times = None

        self.mutex = Lock()
        self.map_mutex = Lock()

        self.find_bezier = False

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
        self.map3d_updated = False
        self.subplots = []

        self.RESOLUTION = 0.05

        self.new_path = rospy.Publisher('rrt_path', Polygon, queue_size=1)
        self.viz_pub = rospy.Publisher("visualization_msgs/Marker", Marker, queue_size=3)
        self.viz_pub2 = rospy.Publisher("visualization_msgs/Marker2", Marker, queue_size=3)
        self.no_path = rospy.Publisher("no_path_found", Bool, queue_size=1)
        rospy.Subscriber("current_coords", Point, self.update_coords, queue_size=1)
        rospy.Subscriber("/"+self.other_robot_name+"/current_coords", Point, self.update_other_coords, queue_size=1)
        rospy.Subscriber("new_goal_loc", Point, self.update_goal, queue_size=1)
        rospy.Subscriber("/"+self.other_robot_name+"/new_goal_loc", Point, self.update_other_goal, queue_size=1)
        rospy.Subscriber("map", OccupancyGrid, self.update_map, queue_size=3)
        # rospy.Subscriber("/secondary_robot/map", OccupancyGrid, self.update_map, queue_size=3)
        rospy.Subscriber("/map_server/opponent_robots", PointCloud, self.detected_robots_callback, queue_size=1)
        rospy.Subscriber("/"+self.other_robot_name+"/rrt_path", Polygon, self.rrt_found, queue_size=1)
        # rospy.Subscriber("lookahead_pnts", Point, self.add_look, queue_size=1)

    def update_coords(self, msg):
        self.path = []
        self.coords[0] = msg.x
        self.coords[1] = msg.y
        self.coords[2] = msg.z
        self.map_updated = False

    def update_goal(self, msg):
        if msg.x == -1 and msg.y == -1 and msg.z == -1:
            print "Setting goal to none"
            self.goal = None
        else:
            self.goal = np.zeros(3)
            self.goal[0] = msg.x
            self.goal[1] = msg.y
            self.goal[2] = msg.z
            # if self.map3d_updated:
            #     self.find_path_rrtstar()

    def update_other_coords(self, msg):
        # self.path = []
        self.other_coords[0] = msg.x
        self.other_coords[1] = msg.y
        self.other_coords[2] = msg.z
        print "Know other robot's start pose"

    def update_other_goal(self, msg):
        if msg.x == -1 and msg.y == -1 and msg.z == -1:
            print "Setting goal to none"
            self.other_goal = None
        else:
            self.other_goal[0] = msg.x
            self.other_goal[1] = msg.y
            self.other_goal[2] = msg.z
            self.map_updated = False
            self.map3d_updated = False
            self.other_path = []
            self.other_path_found = False
            print "Know other robot's goal"

    def rrt_found(self, msg):
        pnts = self.poly_to_list(msg.points)
        self.other_path_found = True
        if len(pnts):
            self.other_path = np.array(pd.unique(pnts).tolist())
            res = self.TIME_RESOLUTION
        else:
            print "NO PATH FOR OTHER ROBOT"
            self.other_path = np.array([self.other_coords[:2], self.other_goal[:2]])
            res = int(np.linalg.norm(self.other_path[1] - self.other_path[0])/0.1)

        self.other_angles, self.other_times = self.calc_angle_and_time_progression()
        print "Found time, angle progressions"
        while not self.map_updated:
            pass
        print "Making", (len(self.other_path)-1) * res + 1, "time slices"
        if len(self.permissible_region.shape) != 2:
            self.permissible_region = self.permissible_region[0]
        self.permissible_region = np.repeat(self.permissible_region[np.newaxis, :, :],
                                            (len(self.other_path) - 1) * res + 1, axis=0)
        self.mutex.acquire()
        self.update_time_slices(res)
        self.mutex.release()
        self.map3d_updated = True
        if self.goal is not None:
            self.find_path_rrtstar(res)

    def update_map(self, msg):
        if not self.map_updated:
            self.mutex.acquire()
            # self.map_mutex.acquire()
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
            self.map3d_updated = False
            print "Map Updated"
            # plt.imshow(self.permissible_region)
            # plt.pause(1)
            # if self.other_path_found:
            #     print "Making", self.other_path.shape[0], "time sliceS"
            #     self.permissible_region = np.repeat(self.permissible_region[np.newaxis, :, :],
            #                                         (self.other_path.shape[0]-1) * self.TIME_RESOLUTION + 1, axis=0)
            #     self.update_time_slices()
            #     self.map3d_updated = True
            #     if self.goal is not None:
            #         self.find_path_rrtstar()

            # self.map_mutex.release()
            self.mutex.release()

    def update_time_slices(self, res=None):
        # robot_loc = [self.other_coords[0]/1000.0, self.other_coords[1]/1000.0, self.other_coords[2]]
        if res is None:
            res = self.TIME_RESOLUTION
        print "STARTED TIME MAP"
        # plt.figure()
        for i, pt in enumerate(self.other_path):
            # print i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i
            robot_loc = [self.other_path[i, 0], self.other_path[i, 1], self.other_angles[i]]
            if i == 0:
                start_loc = robot_loc[:]
            if i < self.other_path.shape[0] - 1:
                delta_angle = ((self.other_angles[i+1] - self.other_angles[i] + np.pi) % (2*np.pi) - np.pi) / float(res)
                delta_d = (self.other_path[i+1] - self.other_path[i]) / float(res)
                iters = res
            else:
                delta_angle = 0
                delta_d = np.zeros(2)
                iters = 1
            for j in xrange(iters):
                size = self.size_main if self.robot_name == "secondary_robot" else self.size_secondary
                if i+j > 0:
                    self.permissible_region[i*res + j, self.our_robot(size*1.1/self.resolution, start_loc)] = 1
                self.permissible_region[i*res + j, self.our_robot(size*1.1/self.resolution, robot_loc)] = 0
                # # print robot_loc, "ROBOT LOCATION"
                # top_left, bottom_left, top_right, bottom_right = self.calc_bounding_vals(robot_loc, self.other_dims)
                # # print top_left, bottom_left, top_right, bottom_right, "corner coords"
                # if abs(robot_loc[2]) <= np.pi / 4 or abs(robot_loc[2]) >= 3 * np.pi / 4:
                #     x_ind_top, y_ind_top = self.get_map_line(top_left, top_right)
                #     dy = bottom_left[1] - top_left[1]
                #     sign = int(np.sign(dy)) if dy != 0.0 else 1
                #     dx = bottom_left[0] - top_left[0]
                #     y_ind = np.arange(0, bottom_left[1] - top_left[1] + sign * 1, sign, dtype=np.intp)
                # else:
                #     x_ind_top, y_ind_top = self.get_map_line(bottom_left, top_left)
                #     dy = bottom_right[1] - bottom_left[1]
                #     sign = int(np.sign(dy)) if dy != 0.0 else 1
                #     dx = bottom_right[0] - bottom_left[0]
                #     y_ind = np.arange(0, bottom_right[1] - bottom_left[1] + sign * 1, sign, dtype=np.intp)
                #
                # dy = sign * max(1, abs(dy))
                # deltaerr = abs(float(dx) / dy)
                # error = 0.0
                #
                # for k in xrange(y_ind.shape[0]):
                #     self.permissible_region[i*res + j, y_ind_top + y_ind[k], x_ind_top] = 0
                #     error += deltaerr
                #     while error >= 0.5:
                #         x_ind_top += int(np.sign(dx))
                #         error -= 1

                robot_loc[:2] += delta_d
                robot_loc[2] += delta_angle

                # rows = np.floor(np.sqrt(self.permissible_region.shape[0]))
                # cols = np.ceil(np.sqrt(self.permissible_region.shape[0]))
                # sub = plt.subplot(rows, cols, i*res + j + 1)
                # self.subplots.append(sub)
                # plt.imshow(self.permissible_region[i*res + j])

                # plt.figure()
                # plt.imshow(self.permissible_region[i*j + j])
                # plt.plot(top_left[0], top_left[1], 'ro')
                # plt.plot(top_right[0], top_right[1], 'bo')
                # plt.plot(bottom_right[0], bottom_right[1], 'go')
                # plt.plot(bottom_left[0], bottom_left[1], 'yo')
                # plt.show()

            # plt.figure()
            # for j in xrange(iters):
            #     plt.subplot(2,3,j+1)
            #     plt.imshow(self.permissible_region[i*res + j])
            # plt.show()
        # plt.pause(10)

        print "TIME MAP FINISHED WITH SHAPE:", self.permissible_region.shape

    def calc_angle_and_time_progression(self):
        goal_delta_angle = np.arctan2(self.other_path[-1][1] - self.other_coords[1], self.other_path[-1][0] - self.other_coords[0]) - self.other_goal[2]
        other_desired_direction = self.OTHER_DESIRED_DIRECS[
            np.abs((self.OTHER_DESIRED_DIRECS - goal_delta_angle + np.pi) % (2 * np.pi) - np.pi).argmin()]

        other_path_len = self.other_path.shape[0]
        angles = np.zeros(other_path_len)
        times = np.zeros(other_path_len)
        angles[0] = (self.other_coords[2] + np.pi) % (2*np.pi) - np.pi
        times[0] = 0
        for i in xrange(1, other_path_len):
            # travel = ???
            # turn_rate = (self.other_desired_direction - travel + np.pi) % (2 * np.pi) - np.pi

            angles[i] = (np.arctan2(self.other_path[i][1] - self.other_path[i-1][1], self.other_path[i][0] - self.other_path[i-1][0]) + np.pi + other_desired_direction) % (2*np.pi) - np.pi
            times[i] = times[i-1] + np.linalg.norm(self.other_path[i] - self.other_path[i-1])
        angles[-1] = (self.other_goal[2] + np.pi) % (2*np.pi) - np.pi
        # print "ANGLE PROGRESSION FOUND"
        # print angles*180/np.pi
        return angles, times

    def calc_bounding_vals(self, robot_loc, robot_dims):
        center = np.array([robot_loc[0] - self.x0, robot_loc[1] - self.y0]) / self.resolution
        # print center, "robot_center"
        left_right_sides = np.array([-robot_dims[0] / 2, robot_dims[0] / 2]) / self.resolution
        bottom_top_sides = np.array([-robot_dims[1] / 2, robot_dims[1] / 2]) / self.resolution
        # print left_right_sides, bottom_top_sides, "left right top bottom in robot coords"
        top_left = (self.rotate([left_right_sides[0], bottom_top_sides[1]], robot_loc[2]) + center).astype(np.intp)
        bottom_left = (self.rotate([left_right_sides[0], bottom_top_sides[0]], robot_loc[2]) + center).astype(np.intp)
        top_right = (self.rotate([left_right_sides[1], bottom_top_sides[1]], robot_loc[2]) + center).astype(np.intp)
        bottom_right = (self.rotate([left_right_sides[1], bottom_top_sides[0]], robot_loc[2]) + center).astype(np.intp)
        return top_left, bottom_left, top_right, bottom_right

    def our_robot(self, size, coords):
        # 'occupy' all cells
        robot = np.full(self.permissible_region.shape[1:], True, dtype='bool')

        x, y = np.meshgrid(np.arange(0, self.permissible_region.shape[2]), np.arange(0, self.permissible_region.shape[1]))

        # upper point
        x1 = coords[0] / self.resolution - size[1] / 2 * np.sin(coords[2])
        y1 = coords[1] / self.resolution + size[1] / 2 * np.cos(coords[2])

        # lower point
        x2 = coords[0] / self.resolution + size[1] / 2 * np.sin(coords[2])
        y2 = coords[1] / self.resolution - size[1] / 2 * np.cos(coords[2])

        # left point
        x3 = coords[0] / self.resolution - size[0] / 2 * np.cos(coords[2])
        y3 = coords[1] / self.resolution - size[0] / 2 * np.sin(coords[2])

        # right point
        x4 = coords[0] / self.resolution + size[0] / 2 * np.cos(coords[2])
        y4 = coords[1] / self.resolution + size[0] / 2 * np.sin(coords[2])

        # 'free' cells outside of each side of the robot
        a = coords[2] % (2 * np.pi)
        if a < np.pi / 2 or a > 3 * np.pi / 2:
            robot[y - y1 > (x - x1) * np.tan(coords[2])] = False
            robot[y - y2 < (x - x2) * np.tan(coords[2])] = False
        else:
            robot[y - y1 < (x - x1) * np.tan(coords[2])] = False
            robot[y - y2 > (x - x2) * np.tan(coords[2])] = False
        if a < np.pi:
            robot[y - y3 < (x - x3) * np.tan(np.pi / 2 + coords[2])] = False
            robot[y - y4 > (x - x4) * np.tan(np.pi / 2 + coords[2])] = False
        else:
            robot[y - y3 > (x - x3) * np.tan(np.pi / 2 + coords[2])] = False
            robot[y - y4 < (x - x4) * np.tan(np.pi / 2 + coords[2])] = False

        return robot

    @staticmethod
    def get_map_line(start, end):
        dx = end[0] - start[0]
        sign = np.sign(dx) if dx != 0.0 else 1
        dx = sign * max(1, abs(dx))
        dy = end[1] - start[1]
        deltaerr = abs(float(dy) / dx)
        error = .0

        x_ind = np.arange(start[0], end[0] + sign*1, sign, dtype=np.intp)
        y_ind = np.zeros(x_ind.shape[0], dtype=np.intp)

        y = start[1]
        for i in xrange(x_ind.shape[0]):
            y_ind[i] = y
            error = error + deltaerr
            while error >= 0.5:
                y += int(np.sign(dy))
                error -= 1

        return x_ind, y_ind

    def detected_robots_callback(self, data):
        if len(data.points) == 0:
            self.opponent_robots = np.array([])
        else:
            self.opponent_robots = np.array([[robot.x, robot.y] for robot in data.points])
        self.robots_upd_time = data.header.stamp

    def find_path_rrtstar(self, res):
        self.mutex.acquire()
        if not self.map3d_updated:
            self.mutex.release()
            print "Stopping because no 3d max exists"
            return
        # plt.clf()
        # self.subplots[0].plot((self.coords[0]-self.x0) / self.resolution, (self.coords[1]-self.y0) / self.resolution, 'ob')
        # self.subplots[-1].plot((self.goal[0]-self.x0) / self.resolution, (self.goal[1]-self.y0) / self.resolution, 'or')
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
            if j % self.GOAL_SAMPLE_RATE == (self.GOAL_SAMPLE_RATE-1):
                # sample goal location every 10 iterations
                rnd = [self.goal[0], self.goal[1]]
            else:
                rnd = [self.x0 + self.map_width * np.random.uniform(), self.y0 + self.map_height * np.random.uniform()]

            # plt.plot(rnd[0], rnd[1], 'og')
            # find the nearest node
            nn_idx = self.nearest_node(rnd)
            x_near = self.nodes[nn_idx]

            # calculate step size based on distance from goal
            dist = np.linalg.norm([self.goal[0] - x_near.x, self.goal[1] - x_near.y])
            step = max(min(dist/2, self.STEP_MAX), self.STEP_MIN)

            # expand the tree
            x_new, theta = self.steer_to_node(x_near, rnd, step)  # goes from x_near to x_new which points towards rnd
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
            start_path_width = (self.PATH_WIDTH_MAX - self.PATH_WIDTH_SMALL) * angle_fraction + self.PATH_WIDTH_SMALL
            # print start_path_width, "START PATH WIDTH"
            if self.obstacle_free(x_near, x_new, theta, start_path_width, step, res):
                # plt.plot([x_near.x, x_new.x], [x_new.y, x_new.y])

                # find close nodes
                # near_idxs = self.find_near_nodes(x_new, step*2)
                # near_idxs = np.arange(len(self.nodes))

                # find the best parent for the node
                # x_new = self.choose_parent(x_new, near_idxs, step, res)

                time_step = np.argmax(x_new.total_cost <= self.other_times)-1
                if x_new.total_cost > 0:
                    interval = (self.other_times[time_step + 1] - self.other_times[time_step]) / float(
                        res)
                    intermediate_step = np.arange(self.other_times[time_step], self.other_times[time_step + 1],
                                                  interval)
                    inter = (np.abs(intermediate_step - x_new.total_cost)).argmin()
                    time_step = res * time_step + inter
                else:
                    time_step = 0

                # self.subplots[time_step].plot([(x_new.x-self.x0) / self.resolution, (self.nodes[x_new.parent].x-self.x0) / self.resolution],
                #          [(x_new.y-self.y0) / self.resolution, (self.nodes[x_new.parent].y-self.y0) / self.resolution], 'g-')
                # plt.pause(.1)

                # add new node and rewire
                self.nodes.append(x_new)
                # self.rewire(x_new, near_idxs, step)

                # check if sample point is in goal region
                dx = x_new.x - self.goal[0]
                dy = x_new.y - self.goal[1]
                d = np.sqrt(dx ** 2 + dy ** 2)
                # print self.EPSILON_GOAL
                if d <= self.EPSILON_GOAL:
                    # self.subplots[time_step].plot((self.goal[0] - self.x0) / self.resolution,
                    #                        (self.goal[1] - self.y0) / self.resolution, 'or')
                    # plt.pause(0.001)
                    print "found goal! took", j, "iterations"
                    # construct path
                    # last_idx = self.get_last_idx()
                    last_idx = -1
                    if last_idx is None:
                        self.mutex.release()
                        return
                    path = self.get_path(last_idx)
                    self.path = pd.unique(np.array(path))
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

        self.visualize()
        print "NO PATH FOUND IN", self.MAX_RRT_ITERS, "ITERATIONS"
        self.no_path.publish(True)
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

    def steer_to_node(self, x_near, rnd, step):
        """ Steers from x_near to a point on the line between rnd and x_near """

        theta = np.arctan2(rnd[1] - x_near.y, rnd[0] - x_near.x)
        new_x = x_near.x + step * np.cos(theta)
        new_y = x_near.y + step * np.sin(theta)

        x_new = Node(new_x, new_y)
        x_new.cost += step
        x_new.total_cost = step + x_near.total_cost

        return x_new, theta

    def obstacle_free(self, nearest_node, new_node, theta, start_path_width, step, res, plot=False):
        """ Checks if the path from x_near to x_new is obstacle free """

        dx_end = np.sin(theta) * self.path_width
        dy_end = np.cos(theta) * self.path_width
        dx_start = np.sin(theta) * start_path_width
        dy_start = np.cos(theta) * start_path_width

        mult = .5 * step / self.STEP_MAX

        time_step = np.argmax(new_node.total_cost <= self.other_times)-1
        # if new_node.total_cost > 0 and time_step == -1:
        #     time_step = -1
        if new_node.total_cost > 0:
            interval = (self.other_times[time_step + 1] - self.other_times[time_step]) / float(res)
            intermediate_step = np.arange(self.other_times[time_step], self.other_times[time_step + 1], interval)
            inter = (np.abs(intermediate_step - new_node.total_cost)).argmin()
            time_step = res * time_step + inter
        else:
            time_step = 0
        time_step = min(time_step, self.permissible_region.shape[0]-1)

        bound0 = ((nearest_node.x, nearest_node.y), (new_node.x, new_node.y))
        bound1 = ((nearest_node.x + dx_start - dy_start*mult, nearest_node.y - dy_start - dx_start*mult), (new_node.x + dx_end + dy_end*mult, new_node.y - dy_end + dx_end*mult))
        bound2 = ((nearest_node.x - dx_start - dy_start*mult, nearest_node.y + dy_start - dx_start*mult), (new_node.x - dx_end + dy_end*mult, new_node.y + dy_end + dx_end*mult))
        if self.line_collision(bound0, time_step, plot) or self.line_collision(bound1, time_step, plot) or self.line_collision(bound2, time_step, plot):
            return False
        if start_path_width > .15:
            dx_start *= .5
            dy_start *= .5
            dx_end *= .5
            dy_end *= .5
            bound3 = ((nearest_node.x + dx_start - dy_start*mult, nearest_node.y - dy_start - dx_start*mult), (new_node.x + dx_end + dy_end*mult, new_node.y - dy_end + dx_end*mult))
            bound4 = ((nearest_node.x - dx_start - dy_start*mult, nearest_node.y + dy_start - dx_start*mult), (new_node.x - dx_end + dy_end*mult, new_node.y + dy_end + dx_end*mult))
            if self.line_collision(bound3, time_step, plot) or self.line_collision(bound4, time_step, plot):
                return False
        return True

    def line_collision(self, line, time, plot=False):
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
                              np.ceil((line[1][0]) / self.resolution + sign * 1), sign * 5, dtype=int)
            y_ind = np.zeros(x_ind.shape[0])

            y = int(np.ceil((line[0][1]) / self.resolution))
            for i in xrange(x_ind.shape[0]):
                y_ind[i] = y
                error = error + 5*deltaerr
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
                # print row, column, "rowcol"
                # print self.map_width, self.map_height
                if self.permissible_region.shape[1:] != (204, 304):
                    print self.permissible_region.shape
                if not self.permissible_region[time, row, column]:
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
                              np.ceil((line[1][1]) / self.resolution + sign * 1), sign * 5, dtype=int)
            x_ind = np.zeros(y_ind.shape[0])

            x = int(np.ceil((line[0][0]) / self.resolution))
            for i in xrange(y_ind.shape[0]):
                x_ind[i] = x
                error = error + 5 * deltaerr
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
                # print row, column, "rowcol"
                # print self.map_width, self.map_height
                if self.permissible_region.shape[1:] != (204, 304):
                    print self.permissible_region.shape
                if not self.permissible_region[time, row, column]:
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
        length_nodes = len(self.nodes)
        r = self.GAMMA_RRT * np.sqrt((np.log(length_nodes) / length_nodes))
        near_idxs = []
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            # if node.distance((x_new.x, x_new.y)) <= r:
            if node.distance((x_new.x, x_new.y)) <= step:
                near_idxs.append(i)
        return near_idxs

    def choose_parent(self, x_new, near_idxs, step, res):
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
        # mincost = min(distances)
        # minind = near_idxs[distances.index(mincost)]
        #
        # if mincost == float("inf"):
        #     return x_new
        #
        # x_new.cost = mincost
        # x_new.parent = minind

        d_angles = []
        for i in near_idxs:
            node = self.nodes[i]
            dx = x_new.x - node.x
            dy = x_new.y - node.y
            theta_new = np.arctan2(dy, dx)

            if node.parent is None:
                continue
                angle_fraction = 2*abs((self.coords[2] - theta_new + np.pi/2) % np.pi - np.pi/2)/np.pi
                delta = abs((self.coords[2] - theta_new + np.pi) % (2 * np.pi) - np.pi)
            else:
                prev_node = self.nodes[node.parent]
                old_theta = np.arctan2(node.y - prev_node.y, node.x - prev_node.x)
                delta = abs((old_theta - theta_new + np.pi) % (2 * np.pi) - np.pi)
                angle_fraction = 2*abs((old_theta - theta_new + np.pi/2) % np.pi - np.pi/2)/np.pi
            start_path_width = (self.PATH_WIDTH_MAX - self.PATH_WIDTH_SMALL) * angle_fraction + self.PATH_WIDTH_SMALL
            # print start_path_width, "START PATH WIDTH"

            if self.obstacle_free(node, x_new, theta_new, start_path_width, step, res):
                d_angles.append(delta)
            else:
                d_angles.append(float("inf"))

        mindeltaangle = min(d_angles)
        minind = near_idxs[d_angles.index(mindeltaangle)]

        if mindeltaangle == float("inf"):
            return x_new

        x_new.cost = self.nodes[minind].distance((x_new.x, x_new.y))
        x_new.total_cost = x_new.cost + self.nodes[minind].total_cost
        x_new.parent = minind

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
                start_path_width = (self.PATH_WIDTH_MAX - self.PATH_WIDTH_SMALL) * angle_fraction + self.PATH_WIDTH_SMALL
                # print start_path_width, "START PATH WIDTH"

                if self.obstacle_free(node, x_new, theta, start_path_width, step, True):
                    node.parent = n - 1
                    node.cost = scost
                    node.total_cost = scost + self.nodes[n-1].total_cost
                    plt.plot((np.array([node.x, self.nodes[n-1].x]) - self.x0) / self.resolution, (np.array([node.y, self.nodes[n-1].y]) - self.y0) / self.resolution, 'r')
                    plt.pause(.1)

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
        if self.find_bezier:
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

    def update_path_width(self, dir):
        pass
        # self.path_width = new_pw

    def bezier_path(self, P0, P1, P2, step):
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
                if self.nodeless_obstacle_free(path[i-1], potential_point, theta, self.path_width, step):
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

    def nodeless_obstacle_free(self, nearest_node, new_node, theta, start_path_width):
        """ Checks if the path from x_near to x_new is obstacle free """

        dx_start = np.sin(theta) * start_path_width
        dy_start = np.cos(theta) * start_path_width
        dx_end = np.sin(theta) * self.path_width
        dy_end = np.cos(theta) * self.path_width
        # plt.plot([(new_node[0] - self.x0) / self.resolution, (nearest_node[0] - self.x0) / self.resolution],
        #          [(new_node[1] - self.y0) / self.resolution, (nearest_node[1] - self.y0) / self.resolution])
        # plt.pause(.001)

        if nearest_node[0] < new_node[0]:
            bound0 = ((nearest_node[0], nearest_node[1]), (new_node[0], new_node[1]))
            bound1 = ((nearest_node[0] + dx_start, nearest_node[1] - dy_start), (new_node[0] + dx_end, new_node[1] - dy_end))
            bound2 = ((nearest_node[0] - dx_start, nearest_node[1] + dy_start), (new_node[0] - dx_end, new_node[1] + dy_end))
            # dx*=.5
            # dy*=.5
            # bound3 = ((nearest_node[0] + dx, nearest_node[1] - dy), (new_node[0] + dx, new_node[1] - dy))
            # bound4 = ((nearest_node[0] - dx, nearest_node[1] + dy), (new_node[0] - dx, new_node[1] + dy))
        else:
            bound0 = ((new_node[0], new_node[1]), (nearest_node[0], nearest_node[1]))
            bound1 = ((new_node[0] + dx_end, new_node[1] - dy_end), (nearest_node[0] + dx_start, nearest_node[1] - dy_start))
            bound2 = ((new_node[0] - dx_end, new_node[1] + dy_end), (nearest_node[0] - dx_start, nearest_node[1] + dy_start))
            # dx*=.5
            # dy*=.5
            # bound3 = ((new_node[0] + dx, new_node[1] - dy), (nearest_node[0] + dx, nearest_node[1] - dy))
            # bound4 = ((new_node[0] - dx, new_node[1] + dy), (nearest_node[0] - dx, nearest_node[1] + dy))

        if self.line_collision(bound0) or self.line_collision(bound1) or self.line_collision(bound2):# or self.line_collision(bound3) or self.line_collision(bound4):
            return False
        else:
            return True

    @staticmethod
    def bot_to_global_transform(pnt, bot_pose):
        c, s = np.cos(bot_pose[2]), np.sin(bot_pose[2])
        dx = pnt[0] * c - pnt[1] * s
        dy = pnt[0] * s + pnt[1] * c
        return np.array([dx + bot_pose[0], dy + bot_pose[1]])

    @staticmethod
    def rotate(pnt, angle):
        c, s = np.cos(angle), np.sin(angle)
        dx = pnt[0] * c + pnt[1] * s
        dy = pnt[0] * s - pnt[1] * c
        return np.array([dx, dy])

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

    @staticmethod
    def poly_to_list(points):
        if points == []:
            return []
        pnts = []
        for pt in points:
            pnts.append((pt.x, pt.y))
        return np.array(pnts)

    def visualize(self, topic=1):
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

                    # dx = np.sin(theta) * self.path_width
                    # dy = np.cos(theta) * self.path_width

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

            self.viz_pub.publish(line_strip)


if __name__ == "__main__":
    coordinated_motion_planner = CoordinatedMotionPlanner()

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