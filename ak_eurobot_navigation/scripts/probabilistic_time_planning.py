#!/usr/bin/env python
import rospy
from rospy.numpy_msg import numpy_msg
from geometry_msgs.msg import Point, Polygon
from visualization_msgs.msg import Marker
import numpy as np

from ak_eurobot_navigation.msg import Graph
from ak_eurobot_navigation.msg import OccupancyGrid3D

import tf
import time

from threading import Lock

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d


class ProbabilisticTimePlanner:
    def __init__(self):
        rospy.init_node("probabilistic_time_planning", anonymous=True)
        self.robot_name = rospy.get_param("robot_name", "main_robot")
        self.V_MAX = rospy.get_param("/" + self.robot_name + "/motion_planner/V_MAX", .25)
        self.REPLAN_TOLERANCE = rospy.get_param("~REPLAN_TOLERANCE", .5)
        self.MAP_SCALEDOWN = rospy.get_param("~MAP_SCALEDOWN", 8)

        self.teammate_name = "secondary_robot" if self.robot_name == "main_robot" else "main_robot"
        self.teammate_size = np.array([rospy.get_param("/" + self.teammate_name + "/dim_x"),
                                       rospy.get_param("/" + self.teammate_name + "/dim_y")]) / 1000.0
        self.teammate_speed = rospy.get_param("/" + self.teammate_name + "/motion_planner/V_MAX", self.V_MAX)

        self.A = None
        self.nodes = None
        self.goal = None
        self.path = None
        self.coords = np.zeros(3)

        self.teammate_path = None

        self.opponent_map_initialized = False
        self.teammate_map_initialized = False
        self.x0 = -0.02
        self.y0 = -0.02
        self.x = None
        self.y = None
        self.map_depth = None
        self.map_width = None
        self.map_height = None
        self.resolution = 0.01
        self.time_resolution = None
        self.opponent_velocities = None
        self.shape_map = None
        self.probability_map = None
        self.prob_map_small = None
        self.teammate_map = None
        self.team_map_small = None
        self.opponent_map = None
        self.opp_map_small = None

        self.path_info = []

        self.mutex = Lock()

        self.listener = tf.TransformListener()

        self.visualize = "no_robot"

        if self.visualize == self.robot_name:
            self.fig = plt.figure()
            self.img = []
            rospy.on_shutdown(self.show_animation)
        else:
            rospy.on_shutdown(self.record_path_info)

        self.viz_pub = rospy.Publisher("/probabilistic_time_planning/visualization_msgs/Marker", Marker, queue_size=1)
        self.path_pub = rospy.Publisher("new_probability_path", Polygon, queue_size=1)
        rospy.Subscriber("/" + self.robot_name + "/new_goal_loc", Point, self.update_goal, queue_size=1)
        rospy.Subscriber("roadmap", Graph, self.add_roadmap, queue_size=1)
        rospy.Subscriber("/motion_prediction", numpy_msg(OccupancyGrid3D), self.update_probability_map, queue_size=3)
        rospy.Subscriber("/" + self.teammate_name + "/new_probability_path", Polygon, self.add_teammate_path, queue_size=3)
        print "Finished init"
        rospy.Timer(rospy.Duration(1.5), self.find_path)

    def update_goal(self, msg):
        if msg.x == -1 and msg.y == -1 and msg.z == -1:
            print "Setting goal to none"
            self.goal = None
        else:
            print "updating goal"
            self.path = []
            self.goal = np.zeros(3)
            self.goal[0] = msg.x
            self.goal[1] = msg.y
            self.goal[2] = (msg.z + np.pi) % (2 * np.pi) - np.pi
            if self.A is not None:
                self.find_path(None)

    def update_probability_map(self, msg):
        # self.mutex.acquire()
        if not self.opponent_map_initialized:
            self.x0 = msg.info.origin.position.x  # -.2
            self.y0 = msg.info.origin.position.y  # -.2
            self.x, self.y = np.meshgrid(np.arange(msg.info.width), np.arange(msg.info.height))
            self.map_depth = msg.info.depth * msg.info.time_resolution
            self.map_width = msg.info.width * msg.info.resolution
            self.map_height = msg.info.height * msg.info.resolution
            self.resolution = msg.info.resolution
            self.shape_map = (msg.info.width, msg.info.height, msg.info.depth)
            # self.probability_map = np.array(msg.data).reshape((msg.info.depth, msg.info.height, msg.info.width))
            # self.permissible_region = np.ones_like(array255, dtype=bool)
            # self.permissible_region[array255 == 100] = 0  # set occupied regions (100) to 0 and unoccupied regions to 1
            self.opponent_map_initialized = True
            print "Probability Map Initialized"
        print "Probability Map Received"
        self.time_resolution = msg.info.time_resolution
        self.opponent_velocities = msg.velocities
        self.opponent_map = np.array(msg.data).reshape((msg.info.depth, msg.info.height, msg.info.width))
        self.opp_map_small = self.get_small_map(self.opponent_map, self.MAP_SCALEDOWN)
        self.prob_map_small = self.opp_map_small
        # ax = plt.axes(projection='3d')
        # frame1 = ax.plot_wireframe(self.x, self.y, self.prob_map_small[0])
        # plt.show()
        team_map_small = self.find_teammate()
        # ax = plt.axes(projection='3d')
        # x_small, y_small = np.meshgrid(np.arange(np.ceil(msg.info.width / float(self.MAP_SCALEDOWN))),
        #                              np.arange(np.ceil(msg.info.height / float(self.MAP_SCALEDOWN))))
        # ax.plot_wireframe(x_small, y_small, team_map_small)
        # ax.view_init(75, -90)
        # plt.show()
        if team_map_small is not None:
            self.prob_map_small = self.opp_map_small + team_map_small
            self.team_map_small = team_map_small
            if self.visualize == self.robot_name:
                for frame in self.prob_map_small:
                    frame[-1, -1] = 100
                    self.img.append([plt.imshow(frame)])
        elif self.team_map_small is not None:
            self.prob_map_small = self.opp_map_small + self.team_map_small
            if self.visualize == self.robot_name:
                for frame in self.prob_map_small:
                    frame[-1, -1] = 100
                    self.img.append([plt.imshow(frame)])
        # ax = plt.axes(projection='3d')
        # frame1 = ax.plot_wireframe(self.x, self.y, self.prob_map_small[0])
        # plt.show()
        # self.mutex.release()

    def add_teammate_path(self, msg):
        if True or self.robot_name == "secondary_robot":
            print "Received teammate path"
            self.teammate_path = self.poly_to_list(msg.points)
            self.team_map_small = self.find_teammate()
            if self.opponent_map_initialized:
                self.prob_map_small = self.opp_map_small + self.team_map_small
            else:
                self.prob_map_small = self.team_map_small
            if self.visualize == self.robot_name:
                for frame in self.prob_map_small:
                    frame[-1, -1] = 100
                    self.img.append([plt.imshow(frame)])

    def find_teammate(self):
        try:
            (trans, rot) = self.listener.lookupTransform('/map', '/' + self.teammate_name, rospy.Time(0))
            yaw = tf.transformations.euler_from_quaternion(rot)[2]
            # self.coords[:2] = trans[:2]
            # self.coords[2] = yaw
            team_coords = np.empty(3)
            team_coords[:2] = trans[:2]
            team_coords[2] = yaw
            # empty_map = np.zeros_like(self.opp_map_small)
            if self.teammate_path is None or not self.opponent_map_initialized:
                gauss = self.gaussian(team_coords, 2*self.teammate_size)
                return self.get_small_map(gauss, self.MAP_SCALEDOWN)
            teammate_probability_map = self.make_team_prob_map(team_coords)
            return self.get_small_map(teammate_probability_map, self.MAP_SCALEDOWN)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo("Simulator failed to lookup tf for robot")
            return None

    def gaussian(self, coords, size, sigma=10, time_step=0, total_steps=10):
        # x, y = np.meshgrid(np.arange(self.shape_map[0]), np.arange(self.shape_map[1]))
        c_x = (coords[0] - self.x0) / self.resolution
        c_y = (coords[1] - self.y0) / self.resolution
        sigma *= (1 + .25*time_step/total_steps)
        sigma_x = sigma * size[0]
        sigma_y = sigma * size[1]
        const = 50000.0/(sigma * sigma * 2 * np.pi)
        x_diff = (self.x - c_x) * np.cos(coords[2]) + (self.y - c_y) * np.sin(coords[2])
        y_diff = (self.y - c_y) * np.cos(coords[2]) - (self.x - c_x) * np.sin(coords[2])
        exp = np.exp(-np.power(x_diff, 2) / (2.0 * sigma_x) ** 2 -
                     np.power(y_diff, 2) / (2.0 * sigma_y) ** 2)
        z = const * exp
        # ax = plt.axes(projection='3d')
        # ax.plot_wireframe(self.x, self.y, z)
        # ax.set_ylim(0, 304)
        # ax.view_init(45, -90)
        # # plt.pause(0.01)
        # plt.pause(1)
        return z

    def make_team_prob_map(self, coords):
        teammate_map = np.zeros_like(self.opponent_map)
        for t in xrange(len(teammate_map)):
            gauss = self.gaussian(coords, 2*self.teammate_size, 5, t, len(teammate_map))
            teammate_map[t] = gauss
            dist_traveled = self.teammate_speed*.5
            next_point_ind = self.find_closest_segment(coords[:2]) + 1
            prev_point = coords
            if next_point_ind < self.teammate_path.shape[0]:
                next_point = self.teammate_path[next_point_ind]
            else:
                next_point = self.teammate_path[-1]
            dist_to_next_point = np.linalg.norm(next_point[:2] - prev_point[:2])
            while dist_to_next_point < dist_traveled:
                dist_traveled -= dist_to_next_point
                if next_point_ind < self.teammate_path.shape[0] - 1:
                    next_point_ind += 1
                    prev_point = next_point
                    next_point = self.teammate_path[next_point_ind]
                    dist_to_next_point = np.linalg.norm(prev_point[:2] - next_point[:2])
                else:
                    dist_to_next_point = np.linalg.norm(self.teammate_path[-1][:2] - next_point[:2])
                    break
            if dist_to_next_point == 0:
                remaining = 1
            else:
                remaining = min(1, dist_traveled / float(dist_to_next_point))
            coords = prev_point + (next_point - prev_point) * remaining
            # if self.visualize == self.robot_name:
            #     self.img.append([plt.imshow(gauss)])
        return teammate_map

    def find_closest_segment(self, pos):
        # Finds the closest line segment to the bot from anywhere along the trajectory

        x, y = pos
        mps = ([x, y])*np.ones((len(self.teammate_path)-1, 2))

        # finds the closest point to the robot in each line segment
        min_dists = self.find_min_dist(self.teammate_path[:-1, :2], self.teammate_path[1:, :2], mps)

        closest_seg_index = np.argmin(min_dists)
        return closest_seg_index

    @staticmethod
    def find_min_dist(p1, p2, myPose):
        # given 2 lists of points and the robot's position,
        # finds the shortest distance to each line segment from the robot.
        # line segments are formed from the two points at the same index in each list

        # finds the distance squared between the two points
        l2 = np.sum(np.abs(p1 - p2) ** 2, axis=-1, dtype=np.float32)
        # finds where along the line segment is closest to myPose. t is a fraction of the line segment,
        # so it must be between 0 and 1
        if np.any(l2 == 0):
            print "l2", l2, "l2"
        t = np.maximum(0, np.minimum(1, np.einsum('ij,ij->i', myPose - p1, p2 - p1) / l2))
        # uses this fraction to get the actual location of the point closest to myPose
        projection = p1 + np.repeat(t, 2).reshape((p2.shape[0], 2)) * (p2 - p1)
        # returns the distance between the closest point and myPose
        return np.sum(np.abs(myPose - projection) ** 2, axis=-1)

    @staticmethod
    def poly_to_list(points):
        if points == []:
            return []
        pnts = np.empty((len(points), 3))
        for i in xrange(len(points)):
            pnts[i] = [points[i].x, points[i].y, points[i].z]
        return pnts

    @staticmethod
    def get_small_map(a, skip):
        # a = a.astype(np.int16)
        if len(a.shape) == 3:
            return a[:, ::skip, ::skip] + a[:, ::skip, 1::skip] + a[:, 1::skip, ::skip] + a[:, 1::skip, 1::skip]
        return a[::skip, ::skip] + a[::skip, 1::skip] + a[1::skip, ::skip] + a[1::skip, 1::skip]

    def add_roadmap(self, msg):
        if self.A is None:
            # self.mutex.acquire()
            print "received graph msg"
            self.nodes = self.pts_to_arr(msg.nodes)
            print self.nodes
            self.A = msg.edges
            # self.mutex.release()
            if self.goal is not None and (self.path is None or self.path == []):
                self.find_path(None)

    def find_path(self, _):
        self.mutex.acquire()
        if self.goal is not None and self.A is not None and \
                np.linalg.norm(self.coords[:2]-self.goal[:2]) > self.REPLAN_TOLERANCE:
            euclidean_dist = np.linalg.norm(self.coords[:2]-self.goal[:2])
            print euclidean_dist, "dist from goal"
            print "=================================================================="
            print "Finding start and goal on map"
            s, sd = self.find_start_ind()
            g, gd = self.find_goal_ind()
            print s, g
            if s is not None and g is not None:
                print "Finding Path!"
                start = time.clock()
                d, parents, dists = self.dijkstra(s, g, sd)
                # print np.sort(dists).tolist(), "Distance from start for each node"
                total_time = time.clock()-start
                total_dist = dists[g] + gd
                self.path_info.append([euclidean_dist, total_dist, total_time])
                print "took", total_time, "seconds to find path"
                prob_collision = d[g]
                print prob_collision, "total weight"
                self.path = self.get_path(s, g, parents, dists)
                # self.mutex.release()
                print "Path Found!"
                print self.path
                self.pub_path()
                self.vis_path()
            self.mutex.release()
        elif self.goal is not None and np.linalg.norm(self.coords - self.goal) <= self.REPLAN_TOLERANCE:
            self.goal = None
            self.path = None
            self.mutex.release()
        else:
            self.mutex.release()

    def find_start_ind(self):
        try:
            (trans, rot) = self.listener.lookupTransform('/map', '/'+self.robot_name, rospy.Time(0))
            yaw = tf.transformations.euler_from_quaternion(rot)[2]
            self.coords[:2] = trans[:2]
            self.coords[2] = yaw
            error = np.linalg.norm(self.goal - self.coords)
            if self.nodes is not None and error > self.REPLAN_TOLERANCE:
                diffs = np.linalg.norm(self.nodes - self.coords, axis=1)
                min_node = np.argmin(diffs)
                attempts = 0
                while self.A[min_node].node_ids == []:
                    print "Picked an unconnected node! Trying again"
                    attempts += 1
                    min_node = np.argpartition(diffs, attempts)[attempts]
                return int(min_node), diffs[min_node]
            return None, None

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo("Simulator failed to lookup tf for robot")
            print "recursing"
            s = self.find_start_ind()
            return s

    def find_goal_ind(self):
        if self.goal is not None:
            diffs = np.linalg.norm(self.nodes - self.goal, axis=1)
            min_node = np.argmin(diffs)
            attempts = 0
            while self.A[min_node].node_ids == []:
                print "Picked an unconnected node! Trying again"
                attempts += 1
                min_node = np.argpartition(diffs, attempts)[attempts]
            return int(min_node), diffs[min_node]
        return None, None

    def dijkstra(self, s, g, sd):
        d = [float('inf')] * len(self.A)
        parent = [None] * len(self.A)
        d[s], parent[s] = 0, s
        # Q = [float('inf')] * len(self.A)
        Q = np.ones(len(self.A)) * float('inf')
        dists = np.ones(len(self.A)) * float('inf')
        Q[s] = 0
        dists[s] = sd
        for _ in xrange(len(self.A)):
            # u = 0
            # for i in xrange(len(self.A)):
            #     if (Q[u] is None or Q[i] < Q[u]) and Q[i] is not None:
            #         u = i
            u = np.nanargmin(Q)
            if u == g:
                print "found goal!"
                return d, parent, dists
            Q[u] = np.NaN
            for v in self.A[u].node_ids:
                if not np.isnan(Q[v]):
                    self.relax(d, dists, parent, u, v)
                    if Q[v] != d[v]:
                        Q[v] = d[v]
        return d, parent, dists

    def relax(self, d, dists, parent, u, v):
        weight, dist = self.weight(u, v, dists[u], parent[u])
        if d[v] > d[u] + weight:
            dists[v] = dists[u] + dist
            d[v] = d[u] + weight
            parent[v] = u

    def weight(self, u, v, prior_dist, prev_node=None):
        n1 = .125*(self.nodes[u][:2] - np.array([self.x0, self.y0])) / self.resolution
        n2 = .125*(self.nodes[v][:2] - np.array([self.x0, self.y0])) / self.resolution
        diff = np.empty(3)
        diff[:2] = 8 * (n1 - n2)
        diff[2] = (self.nodes[u][2] - self.nodes[v][2] + np.pi) % (2 * np.pi) - np.pi
        dist_from_goal = self.nodes[v] - self.goal
        dist_from_goal[2] = (dist_from_goal[2] + np.pi) % (2 * np.pi) - np.pi
        move_weight = (np.linalg.norm(diff * self.resolution) + np.linalg.norm(dist_from_goal))
        dist = np.linalg.norm(diff[:2] * self.resolution)
        if self.prob_map_small is None or not self.opponent_map_initialized:
            # print "ignoring collisions"
            return dist, dist

        th_start = min(self.prob_map_small.shape[0] - 1, int(np.round(prior_dist*.5 / self.V_MAX)))
        th_end = min(self.prob_map_small.shape[0] - 1, int(np.round((prior_dist + dist)*.5 / self.V_MAX)))

        # print th_start, th_end, "time slice"

        neighbors_n1_x = np.clip(np.arange(5) - 2 + int(n1[0]), 0, self.prob_map_small.shape[2] - 1)
        neighbors_n1_y = np.clip(np.arange(5) - 2 + int(n1[1]), 0, self.prob_map_small.shape[1] - 1)
        n1_x, n1_y = np.meshgrid(neighbors_n1_x, neighbors_n1_y)

        neighbors_n2_x = np.clip(np.arange(5) - 2 + int(n2[0]), 0, self.prob_map_small.shape[2] - 1)
        neighbors_n2_y = np.clip(np.arange(5) - 2 + int(n2[1]), 0, self.prob_map_small.shape[1] - 1)
        n2_x, n2_y = np.meshgrid(neighbors_n2_x, neighbors_n2_y)

        start_inds = np.moveaxis(np.stack((n1_y.flatten(), n1_x.flatten()), axis=-1), -1, 0)
        end_inds = np.moveaxis(np.stack((n2_y.flatten(), n2_x.flatten()), axis=-1), -1, 0)

        start_weight = np.sum(self.prob_map_small[(th_start,) + tuple(start_inds)])

        end_weight = np.sum(self.prob_map_small[(th_end,) + tuple(end_inds)])

        angle_change = 0
        if prev_node is not None:
            n0 = .125*(self.nodes[prev_node][:2] - np.array([self.x0, self.y0])) / self.resolution
            angle_change = (np.arctan2(n2[1] - n1[1], n2[0] - n1[0]) - np.arctan2(n1[1] - n0[1], n1[0] - n0[0])
                            + np.pi) % (2 * np.pi) - np.pi

        return start_weight**2 + end_weight**2 + move_weight**2 + abs(angle_change), dist

        # thresh = 3
        # # th = min(self.probability_map.shape[0] - 1, int(np.ceil(.2*prior_dist/self.time_resolution)))
        # th = min(self.probability_map.shape[0] - 1, int(np.ceil(prior_dist/self.V_MAX)))
        # print th, "time slice"
        # l_bound = self.y >= (min(n1[1], n2[1]) - thresh)
        # u_bound = self.y <= (max(n1[1], n2[1]) + thresh)
        # if diff[0] == 0:
        #     return np.sum(self.prob_map_small[th, (self.x >= (n1[0] - thresh)) & (self.x <= (n1[0] + thresh)) &
        #                                       l_bound & u_bound]) + move_weight, dist
        #
        # slope = diff[1] / float(diff[0])
        # l_eq = self.y - n1[1] >= (self.x - n1[0]) * slope - thresh
        # u_eq = self.y - n1[1] <= (self.x - n1[0]) * slope + thresh
        # if abs(slope) > 1:
        #     return np.sum(self.prob_map_small[th, l_eq & u_eq & l_bound & u_bound]) + move_weight, dist
        #
        # return np.sum(self.prob_map_small[th, l_eq & u_eq & (self.x >= (min(n1[0], n2[0]) - thresh)) &
        #                                   (self.x <= (max(n1[0], n2[0]) + thresh))]) + move_weight, dist

        # def clip(val, which_delta):
        #     return min(int(np.ceil(val)), int(np.ceil(self.shape_map[which_delta]/8.0)) - 1)
        #
        # total = 0
        # th = 0
        # dx = np.ceil(n2[0] - n1[0])
        # dy = np.ceil(n2[1] - n1[1])
        #
        # deltas = np.array([dx, dy])
        # order = np.argsort(-np.absolute(deltas))
        #
        # dzero = deltas[order[0]]
        # done = deltas[order[1]]
        #
        # sign = np.sign(dzero) if dzero != 0.0 else 1
        # zero = sign * max(1, abs(dzero))
        # deltaerr_one = abs(float(done) / zero)
        # error_one = .0
        #
        # zero_ind = np.arange(int(np.ceil(n1[order[0]])),
        #                      int(np.ceil(n1[order[0]])) + dzero + sign, sign * 1, dtype=int)
        # one_ind = np.empty(zero_ind.shape[0])
        #
        # one = int(np.ceil(n1[order[1]]))
        # for i in xrange(zero_ind.shape[0]):
        #     one_ind[i] = one
        #     error_one = error_one + 1 * deltaerr_one
        #     while error_one >= 0.5:
        #         one += np.sign(done)
        #         error_one -= 1
        #
        # offsets = [self.x0, self.y0]
        # for i in xrange(len(zero_ind)):
        #     vals = [clip(zero_ind[i] - offsets[order[0]] / self.resolution, order[0]),
        #             clip(one_ind[i] - offsets[order[1]] / self.resolution, order[1])]
        #     # print vals[np.where(order == 2)[0][0]]
        #     total += self.prob_map_small[th, vals[np.where(order == 1)[0][0]], vals[np.where(order == 0)[0][0]]]
        #
        # return total + dist, dist

    def a_star(self, s, sd, g, gd):
        current = [s, sd, 0]
        visited = [s]
        neighbors = self.find_neighbor_weights(current, g)
        while current != g:
            neighbors = self.A[current].node_ids
            if neighbors == []:
                print "no path..."
                break
            # for neighbor in self.A[current].node_ids:
            current = self.find_next(current, neighbors, g)

    def find_neighbor_weights(self, current, goal):
        neighbors = self.A[current[0]].node_ids
        dists = np.linalg.norm(self.nodes[neighbors] - self.nodes[goal], axis=1)
        collision_probs = self.find_collision_prob()
        weights = collision_probs + dists + current[1]
        return np.stack((neighbors, weights), axis=-1)

    def find_next(self, current, neighbors, goal):
        dists = np.linalg.norm(self.nodes[neighbors] - self.nodes[goal], axis=1)
        weights = current  # TODO
        ind = min(dists+weights)
        return neighbors[ind]

    def get_path(self, start, goal, parents, dists):
        path = [(self.goal[0], self.goal[1], self.goal[2])]
        last_idx = goal
        while parents[last_idx] != start:
            node_ind = parents[last_idx]
            prev_node = parents[node_ind]
            print self.weight(node_ind, last_idx, dists[node_ind], prev_node), "WEIGHT"
            node = self.nodes[node_ind]
            if (node[0], node[1], node[2]) in path:
                print "Duplicate!!! Not good"
            path.append((node[0], node[1], node[2]))
            last_idx = node_ind
        if (self.coords[0], self.coords[1], self.coords[2]) not in path:
            path.append((self.coords[0], self.coords[1], self.coords[2]))
        path.reverse()
        # path = np.array(path)
        return path

    def pub_path(self):
        self.path_pub.publish(self.to_poly(self.path))

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
                point.z = pt[2]
                poly.points.append(point)
            return poly
        except TypeError:
            return poly

    @staticmethod
    def pts_to_arr(pnts):
        arr = np.empty((len(pnts), 3))
        for p in xrange(len(pnts)):
            arr[p] = [pnts[p].x, pnts[p].y, pnts[p].z]
        return arr

    def vis_path(self):
        if self.path is not None:
            print "visualizing path"
            line_strip = Marker()
            line_strip.type = line_strip.LINE_STRIP
            line_strip.action = line_strip.ADD
            line_strip.header.frame_id = "/map"

            line_strip.id = 1 if self.robot_name == "main_robot" else 2

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

            for node in self.path:
                new_point = Point()
                new_point.x = node[0]
                new_point.y = node[1]
                new_point.z = 0.0
                line_strip.points.append(new_point)

            self.viz_pub.publish(line_strip)

    def show_animation(self):
        print len(self.img)
        # self.stop = True
        ani = animation.ArtistAnimation(self.fig, self.img, interval=200, blit=False, repeat_delay=0)
        # ani.save("updatingMap.mp4")
        plt.show()

    def record_path_info(self):
        if len(self.path_info):
            f = open("path_info.txt", "a")
            info = np.array(self.path_info)
            print info.tolist()
            f.write("Euclidean Distances:\n")
            f.write(str(info[:, 0]))
            f.write("\nTotal Distances:\n")
            f.write(str(info[:, 1]))
            f.write("\nCalculation Times:\n")
            f.write(str(info[:, 2]))
            f.close()
            for i in range(10000):
                continue


if __name__ == "__main__":
    probabilistic_time_planner = ProbabilisticTimePlanner()

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