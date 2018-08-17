#!/usr/bin/env python
import rospy
import numpy as np
import tf2_ros
import tf
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist, Point, Polygon
from visualization_msgs.msg import Marker
from std_msgs.msg import String, Bool
from threading import Lock
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import LaserScan, PointCloud
from nav_msgs.msg import OccupancyGrid

import matplotlib.pyplot as plt


def rotate_vel(start_vel, angle):
    new_vel = np.array([start_vel[0] * np.cos(angle) - start_vel[1] * np.sin(angle),
                        start_vel[0] * np.sin(angle) + start_vel[1] * np.cos(angle)])
    print new_vel, "new_vel"
    return new_vel.copy()


class MotionPlanner:
    def __init__(self):
        rospy.init_node("motion_planner", anonymous=True)

        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.listener = tf.TransformListener()

        self.robot_name = rospy.get_param("robot_name")
        self.team_color = rospy.get_param("/field/color")
        self.coords = np.array(rospy.get_param('start_' + self.team_color))
        self.coords[:2] /= 1000.0
        self.vel = np.zeros(3)
        self.last_valid_vel = np.zeros(3)

        self.RATE = rospy.get_param("motion_planner/RATE")
        self.REPLAN_RATE = rospy.get_param("motion_planner/REPLAN_RATE")
        self.XY_GOAL_TOLERANCE = rospy.get_param("motion_planner/XY_GOAL_TOLERANCE")
        self.YAW_GOAL_TOLERANCE = rospy.get_param("motion_planner/YAW_GOAL_TOLERANCE")
        self.V_MAX = rospy.get_param("motion_planner/V_MAX")
        self.W_MAX = rospy.get_param("motion_planner/W_MAX")
        self.C_A_MAX = rospy.get_param("motion_planner/COLLISION_AVOID_MAX")
        self.ACCELERATION = rospy.get_param("motion_planner/ACCELERATION")
        self.D_DECELERATION = rospy.get_param("motion_planner/D_DECELERATION")
        self.GAMMA = rospy.get_param("motion_planner/GAMMA")
        # for movements to water towers or cube heaps
        self.XY_ACCURATE_GOAL_TOLERANCE = rospy.get_param("motion_planner/XY_ACCURATE_GOAL_TOLERANCE")
        self.YAW_ACCURATE_GOAL_TOLERANCE = rospy.get_param("motion_planner/YAW_ACCURATE_GOAL_TOLERANCE")
        self.D_ACCURATE_DECELERATION = rospy.get_param("motion_planner/D_ACCURATE_DECELERATION")
        self.NUM_RANGEFINDERS = rospy.get_param("motion_planner/NUM_RANGEFINDERS")
        self.RANGEFINDER_ANGLES = rospy.get_param("motion_planner/RANGEFINDER_ANGLES")
        self.RF_DISTS = rospy.get_param("motion_planner/RANGEFINDER_DISTANCES")
        # self.RF_Y_DISTS = rospy.get_param("motion_planner/RANGEFINDER_Y_DISTANCES")
        self.COLLISION_STOP_DISTANCE = rospy.get_param("motion_planner/COLLISION_STOP_DISTANCE")
        self.COLLISION_AVOIDANCE_COEFFICIENT = rospy.get_param("motion_planner/COLLISION_AVOIDANCE_COEFFICIENT")
        self.COLLISION_STOP_NEIGHBOUR_DISTANCE = rospy.get_param("motion_planner/COLLISION_STOP_NEIGHBOUR_DISTANCE")
        self.COLLISION_GAMMA = rospy.get_param("motion_planner/COLLISION_GAMMA")
        self.PATH_WIDTH = rospy.get_param("motion_planner/PATH_WIDTH")
        # for pure pursuit path following
        self.LOOKAHEAD = rospy.get_param("motion_planner/LOOKAHEAD")  # 0.25
        self.STEP = rospy.get_param("motion_planner/STEP")
        self.DESIRED_DIRECS = np.array(rospy.get_param("motion_planner/DESIRED_DIRECTIONS_OF_TRAVEL"))
        self.desired_direction = 0.0
        self.look = self.LOOKAHEAD

        self.LIDAR_C_A = rospy.get_param("motion_planner/LIDAR_COLLISION_STOP_DISTANCE")
        self.MAX_LIDAR_DIST = rospy.get_param("motion_planner/MAX_LIDAR_DIST")
        self.CIRCLE_REPLAN_RATE = rospy.get_param("motion_planner/CIRCLE_REPLAN_RATE")

        self.lidar_dist = self.LIDAR_C_A
        self.iters_since_last_collision = 16
        self.color = rospy.get_param("/field/color")
        self.robot_name = rospy.get_param("robot_name")
        self.lidar_point = np.array([rospy.get_param("lidar_x"), rospy.get_param("lidar_y"), rospy.get_param("lidar_a")])
        self.scan = None
        self.ranges = None
        self.intensities = None
        self.min_intensity = rospy.get_param("~MIN_LIDAR_INTENSITY", 3500)
        self.angles = None
        self.fourth_closest = None
        self.fourth_angle = None
        self.travel_direc = None
        self.need_to_make_circle = False
        self.disable_circle = False
        self.time_since_last_circle = self.CIRCLE_REPLAN_RATE
        self.indexes = None
        self.scan_mid_ind = None

        if self.robot_name == "main_robot":
            # get initial cube heap coordinates
            self.heap_coords = np.zeros((6, 2))
            for n in range(6):
                self.heap_coords[n, 0] = rospy.get_param("/field/cube" + str(n + 1) + "c_x") / 1000
                self.heap_coords[n, 1] = rospy.get_param("/field/cube" + str(n + 1) + "c_y") / 1000
        else:
            # get water tower coordinates
            self.towers = np.array(rospy.get_param("/field/towers"))
            self.towers[:, :2] /= 1000.0
            self.tower_approaching_vectors = np.array(rospy.get_param("/field/tower_approaching_vectors"))
            self.tower_approaching_vectors[:, :2] /= 1000.0

        self.mutex = Lock()

        # self.o_map = None
        # self.secondary_o_map = None
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
        self.prev_permissible_region = None
        self.map_updated = False
        self.path_found = False
        self.stuck = False
        self.checking_if_stuck = 0
        self.stuck_loc = np.zeros(2)

        self.cmd_id = None
        self.t_prev = None
        self.goal = None
        self.mode = None
        self.other_robot_name = "main_robot" if self.robot_name == "secondary_robot" else "secondary_robot"
        self.other_robot_coords = np.array(rospy.get_param("/" + self.other_robot_name + "/start_" + self.team_color))
        self.other_robot_coords[:2] /= 1000.0
        self.opponent_robots = np.array([])
        self.rangefinder_data = np.zeros(self.NUM_RANGEFINDERS)
        self.rangefinder_status = np.zeros(self.NUM_RANGEFINDERS)
        self.active_rangefinder_zones = np.ones(3, dtype="int")
        self.avoid_direc = None
        self.avoid = False
        self.avoid_vel = np.zeros(3)

        self.cmd_stop_robot_id = None
        self.robot_stopped = None
        self.stop_id = 0

        self.pub_twist = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.pub_response = rospy.Publisher("response", String, queue_size=1)
        self.pub_cmd = rospy.Publisher("stm_command", String, queue_size=1)
        self.path_plan_pub = rospy.Publisher("new_goal_loc", Point, queue_size=1)
        self.pub_current_coords = rospy.Publisher("current_coords", Point, queue_size=1)
        # self.lookahead_pnts_pub = rospy.Publisher("lookahead_pnts", Point, queue_size=1)
        self.pub_rangefinders = rospy.Publisher("/rangefinder_data", Marker, queue_size=3)
        self.path_viz_pub = rospy.Publisher("visualization_msgs/Marker", Marker, queue_size=3)
        rospy.Subscriber("move_command", String, self.cmd_callback, queue_size=1)
        rospy.Subscriber("response", String, self.response_callback, queue_size=1)
        rospy.Subscriber("barrier_rangefinders_data", Int32MultiArray, self.rangefinder_data_callback, queue_size=3)
        # rospy.Subscriber("/map_server/opponent_robots", PointCloud, self.detected_robots_callback, queue_size=1)
        rospy.Subscriber("/map_w_opponent", OccupancyGrid, self.update_with_opponents, queue_size=1)
        rospy.Subscriber("rrt_path", Polygon, self.rrt_found, queue_size=1)
        rospy.Subscriber("new_probability_path", Polygon, self.prm_path_found, queue_size=1)
        # rospy.Subscriber("/" + self.robot_name + "/scan", LaserScan, self.scan_callback, queue_size=5)
        rospy.Subscriber("/"+self.other_robot_name+"/no_path_found", Bool, self.no_path, queue_size=1)
        rospy.Subscriber("/main_robot/map", OccupancyGrid, self.update_map, queue_size=3)
        rospy.Subscriber("/secondary_robot/map", OccupancyGrid, self.update_map, queue_size=3)
        # start the main timer that will follow given goal points
        rospy.Timer(rospy.Duration(1.0 / self.RATE), self.plan)
        print "Finished Setup"

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
            # plt.imshow(self.permissible_region)
            # plt.pause(1)
            print "Map Updated"

    def update_with_opponents(self, msg):
        if self.map_updated:
            array255 = np.array(msg.data).reshape(self.shape_map[::-1])
            # plt.imshow(self.permissible_region)
            # plt.pause(.0001)
            self.permissible_region[self.prev_permissible_region == 0] = 1
            self.permissible_region[array255 == 0] = 0
            self.prev_permissible_region = array255

    def plan(self, event):
        self.mutex.acquire()
        # self.lidar_dist = self.LIDAR_C_A

        if self.cmd_id is None:
            self.mutex.release()
            return

        # rospy.loginfo("-------NEW MOTION PLANNING ITERATION-------")

        if not self.update_coords():
            self.set_speed(np.zeros(3))
            rospy.loginfo('Stopping because of tf2 lookup failure.')
            self.mutex.release()
            return

        # print self.desired_direction, "desired direction of travel"
        # current linear and angular goal distance
        goal_distance = np.zeros(3)
        goal_distance = self.distance(self.coords, self.goal)
        # rospy.loginfo('Goal distance:\t' + str(goal_distance))
        goal_d = np.linalg.norm(goal_distance[:2])
        # rospy.loginfo('Goal d:\t' + str(goal_d))

        # stop and publish response if we have reached the goal with the given tolerance
        if ((self.mode == 'move' or self.mode == 'move_fast') and goal_d < self.XY_GOAL_TOLERANCE and goal_distance[2] < self.YAW_GOAL_TOLERANCE) or (self.mode == 'move_heap' and goal_d < self.XY_ACCURATE_GOAL_TOLERANCE and goal_distance[2] < self.YAW_ACCURATE_GOAL_TOLERANCE):
            rospy.loginfo(self.cmd_id + " finished, reached the goal")
            self.terminate_following()
            self.mutex.release()
            return

        # # CHOOSE VELOCITY COMMAND.

        if self.avoid_direc:
            # rospy.loginfo('AVOIDING COLLISION: picking new direction to move.', string(self.avoid_direc))
            print "AVOIDING COLLISION: picking new direction to move.", self.avoid_direc
            self.iters_since_last_collision += 1
            vel = self.avoid_vel
            if self.need_to_make_circle and self.path is not None and self.time_since_last_circle >= self.CIRCLE_REPLAN_RATE:
                self.update_path()
                self.need_to_make_circle = False
                self.time_since_last_circle = 0
        else:
            if self.need_to_make_circle and self.path is not None and self.time_since_last_circle >= self.CIRCLE_REPLAN_RATE:
                self.update_path()
                self.need_to_make_circle = False
                self.time_since_last_circle = 0

            active_rangefinders, stop_ranges = self.choose_active_rangefinders()

            if self.robot_name == "secondary_robot":
                a, b, c = active_rangefinders == 0, active_rangefinders == 2, active_rangefinders == 6
                stop_ranges[a] += 50
                stop_ranges[b] += 25
                stop_ranges[c] += 25

            t = rospy.get_time()
            dt = t - self.t_prev
            self.t_prev = t

            # set speed limit
            speed_limit_acs = min(self.V_MAX, np.linalg.norm(self.vel[:2]) + self.ACCELERATION * dt)
            # rospy.loginfo('Acceleration Speed Limit:\t' + str(speed_limit_acs))

            speed_limit_dec = (goal_d / (self.D_ACCURATE_DECELERATION if self.mode == "move_heap" else self.D_DECELERATION)) ** self.GAMMA * self.V_MAX
            # rospy.loginfo('Deceleration Speed Limit:\t' + str(speed_limit_dec))

            if active_rangefinders.shape[0] != 0:
                speed_limit_collision = []
                for i in range(active_rangefinders.shape[0]):
                    if self.rangefinder_data[active_rangefinders[i]] > 0:
                        if self.rangefinder_data[active_rangefinders[i]] < stop_ranges[i]:
                            speed_limit_collision.append(0)
                        else:
                            speed_limit_collision.append(min(self.distance_to_closest_robot() * self.COLLISION_AVOIDANCE_COEFFICIENT, ((self.rangefinder_data[active_rangefinders[i]] - stop_ranges[i]) / (255.0 - stop_ranges[i])) ** self.COLLISION_GAMMA * self.V_MAX))
                    else:
                        speed_limit_collision.append(self.distance_to_closest_robot() * self.COLLISION_AVOIDANCE_COEFFICIENT)
                # rospy.loginfo('Collision Avoidance  Speed Limit:\t' + str(speed_limit_collision))
            else:
                speed_limit_collision = [self.V_MAX]
            # speed_limit_collision = [self.V_MAX]
            speed_limit = min(speed_limit_dec, speed_limit_acs, *speed_limit_collision)
            # rospy.loginfo('Final Speed Limit:\t' + str(speed_limit))

            # Find a path and then follow it
            if self.path == [] or self.stuck:
                self.set_speed(np.zeros(3))
                rospy.loginfo('Stopping because the path has not been found yet')
                self.mutex.release()
                return
            # self.path = None #uncomment later
            vel = None
            wrong_coords = False
            # self.path = None
            if self.path is not None:
                # print "following path"
                vel = self.follow_path()
                vel[0] *= goal_d*3
                vel[1] *= goal_d*3
                # if self.path[1, 2] == 0:
                vel[2] = -self.W_MAX * self.find_rot(np.arctan2(vel[1], vel[0])) * 3
                # else:
                #     vel[2] *= goal_d*3
                # vel[2] = self.W_MAX * goal_distance[2] / goal_d
                if np.linalg.norm(vel[:2]) < self.V_MAX/2:
                # if abs(vel[0]) < .05 and abs(vel[1] < .05):
                    vel = None
            if not vel:
                # maximum speed in goal distance proportion
                vel = self.V_MAX * goal_distance / goal_d
                wrong_coords = True

            vel = np.array(vel)

            if abs(vel[2]) > self.W_MAX:
                vel *= self.W_MAX / abs(vel[2])
            # rospy.loginfo('Vel before speed limit\t:' + str(vel))

            # apply speed limit
            v_abs = np.linalg.norm(vel[:2])
            # rospy.loginfo('Vel abs before speed limit\t:' + str(vel))
            if v_abs > speed_limit:
                vel *= speed_limit / v_abs
            # rospy.loginfo('Vel after speed limit\t:' + str(vel))

            # vel to robot frame
            if wrong_coords:
                vel = self.rotation_transform(vel, -self.coords[2])

            self.last_valid_vel = vel

        active_rangefinders, stop_ranges = self.choose_active_rangefinders()

        a, b, c = active_rangefinders == 0, active_rangefinders == 2, active_rangefinders == 6
        stop_ranges[a] += 50
        stop_ranges[b] += 25
        stop_ranges[c] += 25

        if np.any(self.rangefinder_data[active_rangefinders][self.rangefinder_data[active_rangefinders] > 0] < stop_ranges[self.rangefinder_data[active_rangefinders] > 0]):
            # collision avoidance
            rospy.loginfo('EMERGENCY STOP: collision avoidance. Active rangefinder error.')
            vel[:2] = rotate_vel(self.last_valid_vel, np.pi + np.pi/8)
            vel[2] = 0
            self.iters_since_last_collision = 0
            self.lidar_dist = min(self.MAX_LIDAR_DIST, self.lidar_dist + 10)
            # self.look += .1
            print "INCREASING LIDAR DIST TO: ", self.lidar_dist, "AND LOOKAHEAD DIST TO: ", self.look

            if self.time_since_last_circle >= self.CIRCLE_REPLAN_RATE and self.path != []:# and self.fourth_closest < 500:
                self.need_to_make_circle = True

        elif not self.avoid_direc:
            if self.iters_since_last_collision > 15:
                self.lidar_dist = self.LIDAR_C_A
                # self.look = self.LOOKAHEAD
            self.iters_since_last_collision += 2

        # send cmd: vel in robot frame
        self.set_speed(vel)
        # rospy.loginfo('Vel cmd\t:' + str(vel))
        self.time_since_last_circle += 1
        self.check_if_stuck()

        self.mutex.release()

    def scan_callback(self, scan):
        self.ranges = np.array(scan.ranges) * 1000
        self.intensities = np.array(scan.intensities)
        self.angles = np.arange(scan.angle_max - scan.angle_increment, scan.angle_min - scan.angle_increment, -scan.angle_increment)
        self.angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
        motion_angle = np.arctan2(self.last_valid_vel[1], self.last_valid_vel[0])
        m_a_lidar_frame = motion_angle - self.lidar_point[2]
        rel_angles = (self.angles - m_a_lidar_frame + np.pi) % (2*np.pi) - np.pi

        # self.indexes = np.arange(self.ranges.shape[0])
        # self.scan_mid_ind = self.indexes.shape[0] // 2
        # self.ranges = scan.ranges
        # print scan.range_min, scan.range_max
        # print self.ranges
        self.ranges[self.ranges < scan.range_min*1000] = 0
        self.ranges[self.ranges > scan.range_max*1000] = 0
        # self.ranges[self.intensities < self.min_intensity] = 0
        # forward_ranges = self.ranges[~((rel_angles < np.pi/3)-(rel_angles > -np.pi/3)) & (self.ranges > 0)]
        forward_ranges = self.ranges[(rel_angles < np.pi / 3) & (rel_angles > -np.pi / 3) & (self.ranges > 0)]
        self.fourth_closest = np.sort(forward_ranges)[0]
        self.fourth_angle = self.angles[np.where(self.ranges == self.fourth_closest)][np.argmin(np.abs(self.angles[np.where(self.ranges == self.fourth_closest)]))]
        self.travel_direc = motion_angle
        # print self.ranges
        collision_pnts = self.ranges[0 < self.ranges][self.ranges[0 < self.ranges] < self.lidar_dist]
        if collision_pnts.shape[0] > 0:
            print "something in range"
            collision_angs = rel_angles[0 < self.ranges][self.ranges[0 < self.ranges] < self.lidar_dist]
            # collision_inds = self.indexes[0 < self.ranges][self.ranges[0 < self.ranges] < self.lidar_dist]
            side = collision_angs >= 0
            left_obst = collision_angs >= np.pi/3
            right_obst = collision_angs <= -np.pi/3
            self.avoid_direc = "back"
            if side.all():
                print "need to move right"
                self.avoid_direc = "right forward"
                self.avoid_vel[:2] = rotate_vel(self.last_valid_vel[:2], -np.pi/3)
                self.avoid_vel[2] = self.last_valid_vel[2]
            elif not side.any():
                print "need to move left"
                self.avoid_direc = "left forward"
                self.avoid_vel[:2] = rotate_vel(self.last_valid_vel[:2], np.pi/3)
                self.avoid_vel[2] = self.last_valid_vel[2]
            elif not left_obst.any():
                print "can move left"
                self.avoid_direc = "left"
                self.avoid_vel[:2] = rotate_vel(self.last_valid_vel[:2], np.pi/2)
                self.avoid_vel[2] = self.last_valid_vel[2]
            elif not right_obst.any():
                print "can move right"
                self.avoid_direc = "right"
                self.avoid_vel[:2] = rotate_vel(self.last_valid_vel[:2], -np.pi/2)
                self.avoid_vel[2] = self.last_valid_vel[2]
            else:
                self.avoid_vel[:2] = -self.last_valid_vel[:2]
                self.lidar_dist = min(self.MAX_LIDAR_DIST, self.lidar_dist+10)
                # self.look += .1
                print "need to back up"
                print "increased lidar dist to: ", self.lidar_dist, "and lookahead dist to: ", self.look
            if self.avoid_vel[0] > self.C_A_MAX and self.avoid_vel[0] > self.avoid_vel[1]:
                self.avoid_vel[1] = self.avoid_vel[1]/float(self.avoid_vel[0]) * self.C_A_MAX
                self.avoid_vel[0] = self.C_A_MAX
            elif self.avoid_vel[1] > self.C_A_MAX:
                self.avoid_vel[0] = self.avoid_vel[0]/float(self.avoid_vel[1]) * self.C_A_MAX
                self.avoid_vel[1] = self.C_A_MAX
        else:
            self.avoid_direc = None

    def rrt_found(self, msg):
        self.mutex.acquire()
        pnts = self.poly_to_list(msg.points)
        if not len(pnts):
            self.path = None
        else:
            self.path_found = True
            # self.path = np.array(pd.unique(pnts).tolist())
            self.path = pnts
        self.mutex.release()

    def prm_path_found(self, msg):
        self.mutex.acquire()
        print "prm path found"
        pnts = self.poly_to_list(msg.points)
        if not len(pnts):
            self.path = None
        else:
            self.path_found = True
            # self.path = np.array(pd.unique(pnts).tolist())
            self.path = pnts
        self.mutex.release()

    def update_path(self):
        self.disable_circle = True
        dist = self.fourth_closest / 800.0
        print "ADDING CIRCLE TO PATH WITH RADIUS:", dist
        center_local = np.array([dist*np.cos(self.fourth_angle+self.travel_direc), dist*np.sin(self.fourth_angle+self.travel_direc)])
        center_global = self.bot_to_global_transform(center_local, self.coords)
        print "4angle:", self.fourth_angle, "travel dir:", self.travel_direc
        print "local:", center_local, "global:", center_global, "pose:", self.coords
        ind = self.find_closest_segment(center_global)
        index = ind
        ret_type = None
        rad = dist
        while ret_type != 2 and ret_type != 3:  # types 2, 3 imply either 2 intersections or 1 in desired direction
            seg = [self.path[index, :2], self.path[index + 1, :2]]
            ret_type, closest_pnt, second_intersect = self.goal_point(center_global, rad, seg)
            if ret_type == 3:
                first_intersect = closest_pnt
                first_ind = index + 1
            if index < self.path.shape[0] - 2:
                index += 1
            elif ret_type == 1:
                second_intersect = self.path[-1]
                ret_type = 2
            else:
                self.time_since_last_circle = self.CIRCLE_REPLAN_RATE
                return
        second_ind = index
        # rad1 = rad
        # rad = dist
        # index = ind
        # while ret_type != 1 and ret_type != 3:
        #     seg = [self.path[index], self.path[index + 1]]
        #     ret_type, closest_pnt, first_intersect = self.goal_point(center_global, rad, seg)
        #     if index > 0:
        #         index -= 1
        #     elif ret_type == 2:
        #         first_intersect = self.path[0]
        #         ret_type = 1
        #     else:
        #         rad *= 1.5
        #         index = ind
        # if ret_type != 3:
        #     first_ind = index + 1
        first_ind = self.find_closest_segment(self.coords[:2])

        circle = self.make_circle(center_global, rad, self.path[first_ind], self.path[second_ind])
        if circle is None:
            self.time_since_last_circle = self.CIRCLE_REPLAN_RATE
        else:
            # self.path = np.array(pd.unique(np.concatenate((self.path[:min(first_ind, second_ind)], circle, self.path[second_ind:]))).tolist())

            self.path = np.concatenate((self.path[:first_ind], circle, self.path[second_ind:]))
            if np.all(self.path[first_ind - 1] == self.path[first_ind] and self.path[second_ind - 1] == self.path[second_ind]):
                self.path = np.delete(self.path, [first_ind, second_ind], 0)
            elif np.all(self.path[first_ind - 1] == self.path[first_ind]):
                self.path = np.delete(self.path, first_ind, 0)
            elif np.all(self.path[second_ind - 1] == self.path[second_ind]):
                self.path = np.delete(self.path, second_ind, 0)

            print circle
            self.visualize_path()

    def make_circle(self, center, r, start, stop):
        start_angle = np.arctan2(start[1] - center[1], center[0] - start[0])
        stop_angle = np.arctan2(stop[1] - center[1], center[0] - stop[0])
        angle_diff = (stop_angle - start_angle) % (2*np.pi)
        angle_increment = self.STEP / r
        iters = int(angle_diff / angle_increment)
        th_increment = (stop[2] - start[2]) / iters
        print iters, "iters"
        circle = np.array([(start[0], start[1], start[2])])
        for i in xrange(iters):
            new_point = (center[0] - np.cos(start_angle)*r, center[1] + np.sin(start_angle)*r, start[2] + i*th_increment)
            theta = np.arctan2(new_point[1] - circle[i, 1], new_point[0] - circle[i, 0])
            if self.nodeless_obstacle_free(circle[i], new_point, theta):
                circle = np.append(circle, [new_point], axis=0)
                start_angle += angle_increment
            else:
                print "CIRCLE DID NOT MAKE A OBSTACLE FREE PATH"
                return None
        return circle

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

    def check_if_stuck(self):
        if not self.checking_if_stuck:
            self.stuck_loc = self.coords[:2]
            self.checking_if_stuck += 1
        elif self.checking_if_stuck <= self.REPLAN_RATE:
            if np.linalg.norm(self.coords[:2] - self.stuck_loc) <= 0.3:
                self.checking_if_stuck += 1
            else:
                self.checking_if_stuck = 0
        else:
            self.checking_if_stuck = 0
            # self.stuck = True
            self.path = []
            print "++++++++++++++++++++++++++++++++++++++++++++++++++++"
            print "STUCK. PLANNING NEW PATH"
            print "++++++++++++++++++++++++++++++++++++++++++++++++++++"
            goal = Point()
            goal.x = self.goal[0]
            goal.y = self.goal[1]
            goal.z = self.goal[2]
            start = Point()
            start.x = self.coords[0]
            start.y = self.coords[1]
            start.z = self.coords[2]
            # self.pub_current_coords.publish(start)
            self.path_plan_pub.publish(goal)

    @staticmethod
    def poly_to_list(points):
        if points == []:
            return []
        pnts = np.empty((len(points), 3))
        for i in xrange(len(points)):
            pnts[i] = [points[i].x, points[i].y, points[i].z]
        return pnts

    def no_path(self, msg):
        self.path = []
        print "++++++++++++++++++++++++++++++++++++++++++++++++++++"
        print "OTHER ROBOT STUCK. PLANNING NEW PATH"
        print "++++++++++++++++++++++++++++++++++++++++++++++++++++"
        goal = Point()
        goal.x = self.goal[0]
        goal.y = self.goal[1]
        goal.z = self.goal[2]
        start = Point()
        start.x = self.coords[0]
        start.y = self.coords[1]
        start.z = self.coords[2]
        # self.pub_current_coords.publish(start)
        self.path_plan_pub.publish(goal)

    def find_rot(self, travel):
        # desired_direction = self.DESIRED_DIRECS[np.abs(self.goal[2] - self.DESIRED_DIRECS).argmin()]
        # desired_direction = self.DESIRED_DIRECS[np.abs(travel - self.DESIRED_DIRECS).argmin()]
        # desired_direction = self.DESIRED_DIRECS[np.abs((self.DESIRED_DIRECS - travel + np.pi) % (2*np.pi) - np.pi).argmin()]
        # return desired_direction - travel
        turn_rate = (self.desired_direction - travel + np.pi) % (2*np.pi) - np.pi
        # print turn_rate, "tr"
        return turn_rate

    def follow_path(self):
        if self.path is not None and self.path != []:
            ind = self.find_closest_segment(self.coords[:2])
            index = ind
            intersect_pnt = None
            ret_type = None
            look = self.LOOKAHEAD
            while ret_type != 2 and ret_type != 3:  # types 2, 3 imply either 2 intersections or 1 in desired direction
                seg = [self.path[index, :2], self.path[index+1, :2]]
                ret_type, closest_pnt, intersect_pnt = self.goal_point(self.coords[:2], look, seg)
                if index < self.path.shape[0]-2:
                    index += 1
                elif ret_type == 1 or ret_type == -1:
                    intersect_pnt = self.path[-1, :2]
                    ret_type = 2
                else:
                    look *= 1.5
                    index = ind
                # if ret_type != 2 and ret_type != 3:
                #     self.set_speed(np.zeros(3))
                #     rospy.loginfo('Stopping because path was lost.')

            # transformed_pt = self.global_to_bot_transform(intersect_pnt,myPose)
            transformed_pt = self.rotation_transform(intersect_pnt, -self.coords[2])
            trans_pt = self.global_to_bot_transform(intersect_pnt, self.coords)
            # print trans_pt, "lookahead goal loc"
            trans = Point()
            trans.x = transformed_pt[0]
            trans.y = transformed_pt[1]
            # self.lookahead_pnts_pub.publish(trans)

            if abs(trans_pt[1]) < abs(trans_pt[0]):
                if trans_pt[0] > 0:
                    v_x = self.V_MAX
                else:
                    v_x = -self.V_MAX
                v_y = float(trans_pt[1]) / trans_pt[0] * v_x
            else:
                if trans_pt[1] > 0:
                    v_y = self.V_MAX
                else:
                    v_y = -self.V_MAX
                v_x = float(trans_pt[0]) / trans_pt[1] * v_y
            dist = np.linalg.norm(self.coords[:2] - self.path[ind+1, :2])
            diff = (self.path[ind+1, 2] - self.coords[2] + np.pi) % (2 * np.pi) - np.pi
            v_w = diff/dist * np.linalg.norm([v_x, v_y])

            vel = [v_x, v_y, v_w]
            # print vel, "VELOCITY"
            return vel

        return None

    def find_closest_segment(self, pos):
        # Finds the closest line segment to the bot from anywhere along the trajectory

        x, y = pos
        mps = ([x, y])*np.ones((len(self.path)-1, 2))

        # finds the closest point to the robot in each line segment
        min_dists = self.find_min_dist(self.path[:-1, :2], self.path[1:, :2], mps)

        closest_seg_index = np.argmin(min_dists)
        return closest_seg_index

    @staticmethod
    def find_min_dist(p1, p2, myPose):
        # given 2 lists of points and the robot's position,
        # finds the shortest distance to each line segment from the robot.
        # line segments are formed from the two points at the same index in each list

        # finds the distance squared between the two points
        l2 = np.sum(np.abs(p1-p2)**2, axis=-1, dtype=np.float32)
        # finds where along the line segment is closest to myPose. t is a fraction of the line segment,
        # so it must be between 0 and 1
        if np.any(l2 == 0):
            print "l2", l2, "l2"
        t = np.maximum(0, np.minimum(1, np.einsum('ij,ij->i', myPose - p1, p2 - p1) / l2))
        # uses this fraction to get the actual location of the point closest to myPose
        projection = p1 + np.repeat(t, 2).reshape((p2.shape[0], 2)) * (p2 - p1)
        # returns the distance between the closest point and myPose
        return np.sum(np.abs(myPose-projection)**2, axis=-1)

    @staticmethod
    def goal_point(bot_pos, lookahead, constraint):
        # Finds the closest point on line segment to center of a circle if line segment passes through the circle
        # Returns a Boolean stating if there was an intersection, and where the closest intersection is
        #
        # Inputs: bot_pos = [x,y, orientation]  Point that states the center of the circle
        #         lookahead = radius        Scalar value of the radius of the circle
        #         constraint = 2x2 Vector   End points of path line segment
        #
        #
        # Outputs: Return Type				Which case is returned
        #
        # 		   Closest Point            Vector of the coords of point on
        #                                   line segment closest to circle center
        #
        #          Goal Point               Vector of the coords of point on
        #                                   line segment that is the goal point
        #
        #          t                        Percentage of way along the trajectory
        #                                   the closest point is
        #
        #          goal_t                   Percentage of way along the trajectory
        #                                   the goal point is

        Q = [bot_pos[0], bot_pos[1]]

        r = lookahead                       # Radius of the circle
        P1 = np.array(constraint[0])                # Start of line segment
        v = np.subtract(np.array(constraint[1]), P1)            # Vector along line segment
        a = float(np.dot(v, v))
        if not a:
            return -1, None, None
        b = float(2*np.dot(v, (P1 - Q)))
        try:
            c = np.dot(P1, P1) + np.dot(Q, Q) - 2*np.dot(P1, Q) - r**2
        except OverflowError:
            return 0, None, None

        disc = b**2-4*a*c
        if disc < 0:
            # Line segment is outside of the circle, need to extend circle radius
            return 0, None, None

        sqrt_disc = np.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2*a)
        t2 = (-b - sqrt_disc) / (2*a)

        # Closest point on extended line to center of circle is P1 + t*V where t is listed below
        t = max(min(1.0, -b / (2*a)), 0.0)

        if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
            # Line segment is inside circle but too short, need to shrink circle radius
            return -1, None, None

        elif not (0 <= t1 <= 1):
            # t2 intersects, but not t1. return the closest point and goal point
            return 1, P1 + t*v, P1 + t2*v

        elif not (0 <= t2 <= 1):
            # t1 intersects, but not t2, return the closest point and goal point
            return 2, P1 + t*v, P1 + t1*v

        else:
            goal_t = max((t1-t), (t2-t)) + t
            other_t = min((t1-t), (t2-t)) + t
            return 3, P1 + other_t*v, P1 + goal_t*v

    def choose_active_rangefinders(self):
        if self.robot_name == "secondary_robot":
            d_map_frame = self.goal[:2] - self.coords[:2]
            d_robot_frame = self.rotation_transform(d_map_frame, -self.coords[2])
            # rospy.loginfo("Choosing active rangefinders")
            goal_angle = (np.arctan2(d_robot_frame[1], d_robot_frame[0]) % (2 * np.pi))
            move_angle = (np.arctan2(self.last_valid_vel[1], self.last_valid_vel[0])) % (2 * np.pi)
            # rospy.loginfo("Goal angle in robot frame:\t" + str(goal_angle))
            n = int(round((np.pi / 2 - move_angle) / (np.pi / 4))) % 8
            # rospy.loginfo("Closest rangefinder: " + str(n))
            return np.array([n - 1, n, n + 1])[np.where(self.active_rangefinder_zones)] % 8, np.array([self.COLLISION_STOP_NEIGHBOUR_DISTANCE, self.COLLISION_STOP_DISTANCE, self.COLLISION_STOP_NEIGHBOUR_DISTANCE])[np.where(self.active_rangefinder_zones)]
        else:
            d_map_frame = self.goal[:2] - self.coords[:2]
            d_robot_frame = self.rotation_transform(d_map_frame, -self.coords[2])
            # rospy.loginfo("Choosing active rangefinders")
            goal_angle = (np.arctan2(d_robot_frame[1], d_robot_frame[0]) % (2 * np.pi))
            k = int(round((-np.pi / 2 + goal_angle) / (np.pi / 4))) % 8
            # rospy.loginfo("Goal angle in robot frame:\t" + str(goal_angle) + "\t; k = " + str(k))
            if k == 0:
                # rospy.loginfo("Closest rangefinders: 9, 0")
                rf = np.array([8, 9, 0, 1])
            elif k < 4:
                n = k
                # rospy.loginfo("Closest rangefinder: " + str(n))
                rf = np.array([n - 1, n, n + 1]) % 10
            elif k == 4:
                # rospy.loginfo("Closest rangefinders: 4, 5")
                rf = np.array([3, 4, 5, 6])
            elif k > 4:
                n = k + 1
                # rospy.loginfo("Closest rangefinder: " + str(n))
                rf = np.array([n - 1, n, n + 1]) % 10

            active_rangefinders = []
            stop_ranges = []
            if self.active_rangefinder_zones[0]:
                active_rangefinders.append(rf[0])
                stop_ranges.append(self.COLLISION_STOP_NEIGHBOUR_DISTANCE)
            if self.active_rangefinder_zones[1]:
                active_rangefinders.append(rf[1])
                stop_ranges.append(self.COLLISION_STOP_DISTANCE)
                if rf.shape[0] == 4:
                    active_rangefinders.append(rf[2])
                    stop_ranges.append(self.COLLISION_STOP_DISTANCE)
            if self.active_rangefinder_zones[2]:
                active_rangefinders.append(rf[-1])
                stop_ranges.append(self.COLLISION_STOP_NEIGHBOUR_DISTANCE)

            return np.array(active_rangefinders), np.array(stop_ranges)

    @staticmethod
    def distance(coords1, coords2):
        ans = coords2 - coords1
        if abs(coords1[2] - coords2[2]) > np.pi:
            if coords2[2] > coords1[2]:
                ans[2] -= 2 * np.pi
            else:
                ans[2] += 2 * np.pi
        return ans

    @staticmethod
    def rotation_transform(vec, angle):
        M = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        ans = vec.copy()
        ans[:2] = np.matmul(M, ans[:2].reshape((2, 1))).reshape(2)
        return ans

    @staticmethod
    def global_to_bot_transform(pnt, bot_pose):
        new_pnt = pnt - np.array([bot_pose[0], bot_pose[1]])
        c, s = np.cos(-bot_pose[2]), np.sin(-bot_pose[2])
        return np.array([c * new_pnt[0] - s * new_pnt[1], s * new_pnt[0] + c * new_pnt[1]])

    @staticmethod
    def bot_to_global_transform(pnt, bot_pose):
        c, s = np.cos(bot_pose[2]), np.sin(bot_pose[2])
        dx = pnt[0] * c - pnt[1] * s
        dy = pnt[0] * s + pnt[1] * c
        return np.array([dx + bot_pose[0], dy + bot_pose[1]])

    def terminate_following(self):
        rospy.loginfo("Setting robot speed to zero.")
        self.stop_robot()
        rospy.loginfo("Robot has stopped.")
        if self.mode == "move":
            rospy.loginfo("Starting correction by odometry movement.")
            self.move_odometry(self.cmd_id, *self.goal)
        else:
            self.pub_response.publish(self.cmd_id + " finished")

        self.cmd_id = None
        self.t_prev = None
        self.goal = None
        self.mode = None
        self.path_found = False
        self.path = []
        self.rangefinder_data = np.zeros(self.NUM_RANGEFINDERS)
        self.rangefinder_status = np.zeros(self.NUM_RANGEFINDERS)
        self.active_rangefinder_zones = np.ones(3, dtype="int")
        # p = Point()
        # p.x = -1
        # p.y = -1
        # p.z = -1
        # self.path_plan_pub.publish(p)

    def stop_robot(self):
        self.cmd_stop_robot_id = "stop_" + self.robot_name + str(self.stop_id)
        self.stop_id += 1
        self.robot_stopped = False
        cmd = self.cmd_stop_robot_id + " 8 0 0 0"
        rospy.loginfo("Sending cmd: " + cmd)
        self.pub_cmd.publish(cmd)
        for i in range(20):
            if self.robot_stopped:
                self.cmd_stop_robot_id = None
                rospy.loginfo("Robot stopped.")
                return
            rospy.sleep(1.0 / 40)
        rospy.loginfo("Have been waiting for response for .5 sec. Stopped waiting.")
        self.cmd_stop_robot_id = None

    def set_goal(self, goal, cmd_id, mode='move', active_rangefinder_zones=np.ones(3, dtype="int")):
        rospy.loginfo("Setting a new goal:\t" + str(goal))
        rospy.loginfo("Mode:\t" + str(mode))
        self.cmd_id = cmd_id
        self.mode = mode
        self.active_rangefinder_zones = active_rangefinder_zones
        rospy.loginfo("Active Rangefinder Zones: " + str(active_rangefinder_zones))
        self.t_prev = rospy.get_time()
        self.goal = goal

        goal_delta_angle = np.arctan2(self.goal[1] - self.coords[1], self.goal[0] - self.coords[0]) - self.goal[2]
        print goal_delta_angle, "gda"
        self.desired_direction = self.DESIRED_DIRECS[
            np.abs((self.DESIRED_DIRECS - goal_delta_angle + np.pi) % (2 * np.pi) - np.pi).argmin()]
        print self.desired_direction, "DESIRED DIRECTION OF TRAVEL"

        goal = Point()
        goal.x = self.goal[0]
        goal.y = self.goal[1]
        goal.z = self.goal[2]
        self.path_plan_pub.publish(goal)

    def set_speed(self, vel):
        vx, vy, wz = vel
        tw = Twist()
        tw.linear.x = vx
        tw.linear.y = vy
        tw.angular.z = wz
        self.pub_twist.publish(tw)
        self.vel = vel

    def cmd_callback(self, data):
        self.mutex.acquire()
        coords = Point()
        coords.x = self.coords[0]
        coords.y = self.coords[1]
        coords.z = self.coords[2]
        # self.pub_current_coords.publish(coords)
        rospy.loginfo("===============================================================================================")
        rospy.loginfo("NEW CMD:\t" + str(data.data))

        # parse name,type
        data_splitted = data.data.split()
        cmd_id = data_splitted[0]
        cmd_type = data_splitted[1]
        cmd_args = data_splitted[2:]

        if cmd_type == "move" or cmd_type == "move_fast":
            args = np.array(cmd_args).astype('float')
            goal = args[:3]
            goal[2] %= (2 * np.pi)
            if len(cmd_args) >= 6:
                active_rangefinder_zones = np.array(cmd_args[3:7]).astype('float')
                self.set_goal(goal, cmd_id, cmd_type, active_rangefinder_zones)
            else:
                self.set_goal(goal, cmd_id, cmd_type)

        elif cmd_type == "move_heap":
            self.mode = cmd_type
            n = int(cmd_args[0])
            if len(cmd_args) >= 4:
                active_rangefinder_zones = np.array(cmd_args[1:4]).astype('float')
                self.move_heap(cmd_id, n, active_rangefinder_zones)
            else:
                self.move_heap(cmd_id, n)

        elif cmd_type == "move_odometry":  # simple movement by odometry
            inp = np.array(cmd_args).astype('float')
            inp[2] %= 2 * np.pi
            self.move_odometry(cmd_id, *inp)

        elif cmd_type == "translate_odometry":  # simple liner movement by odometry
            inp = np.array(cmd_args).astype('float')
            self.translate_odometry(cmd_id, *inp)

        elif cmd_type == "rotate_odometry":  # simple rotation by odometry
            inp = np.array(cmd_args).astype('float')
            inp[0] %= 2 * np.pi
            self.rotate_odometry(cmd_id, *inp)

        elif cmd_type == "face_heap":  # rotation (odom) to face cubes
            n = int(cmd_args[0])
            if len(cmd_args) > 1:
                w = np.array(cmd_args[1]).astype('float')
                self.face_heap(cmd_id, n, w)
            else:
                self.face_heap(cmd_id, n)

        elif cmd_type == "stop":
            self.cmd_id = cmd_id
            self.mode = cmd_type
            self.terminate_following()

        self.mutex.release()

    def face_heap(self, cmd_id, n, w=1.0):
        rospy.loginfo("-------NEW ROTATION MOVEMENT TO FACE CUBES-------")
        rospy.loginfo("Heap number: " + str(n) + ". Heap coords: " + str(self.heap_coords[n]))
        while not self.update_coords():
            rospy.sleep(0.05)
        angle = (np.arctan2(self.heap_coords[n][1] - self.coords[1], self.heap_coords[n][0] - self.coords[0]) - np.pi / 2) % (2 * np.pi)
        rospy.loginfo("Goal angle: " + str(angle))
        self.rotate_odometry(cmd_id, angle, w)

    def move_heap(self, cmd_id, n, active_rangefinder_zones=np.ones(3, dtype="int")):
        rospy.loginfo("-------NEW HEAP MOVEMENT-------")
        rospy.loginfo("Heap number: " + str(n) + ". Heap coords: " + str(self.heap_coords[n]))
        while not self.update_coords():
            rospy.sleep(0.05)
        angle = (np.arctan2(self.heap_coords[n][1] - self.coords[1], self.heap_coords[n][0] - self.coords[0]) - np.pi / 2) % (2 * np.pi)
        goal = np.append(self.heap_coords[n], angle)
        goal[:2] -= self.rotation_transform(np.array([.0, .06]), angle)
        rospy.loginfo("Goal:\t" + str(goal))
        self.set_goal(goal, cmd_id, "move_heap", active_rangefinder_zones)

    def move_odometry(self, cmd_id, goal_x, goal_y, goal_a, vel=0.3, w=1.5):
        goal = np.array([goal_x, goal_y, goal_a])
        rospy.loginfo("-------NEW ODOMETRY MOVEMENT-------")
        rospy.loginfo("Goal:\t" + str(goal))
        while not self.update_coords():
            rospy.sleep(0.05)

        d_map_frame = self.distance(self.coords, goal)
        rospy.loginfo("Distance in map frame:\t" + str(d_map_frame))
        d_robot_frame = self.rotation_transform(d_map_frame, -self.coords[2])
        rospy.loginfo("Distance in robot frame:\t" + str(d_robot_frame))
        d = np.linalg.norm(d_robot_frame[:2])
        rospy.loginfo("Distance:\t" + str(d))

        beta = np.arctan2(d_robot_frame[1], d_robot_frame[0])
        rospy.loginfo("beta:\t" + str(beta))
        da = d_robot_frame[2]
        dx = d * np.cos(beta - da / 2)
        dy = d * np.sin(beta - da / 2)
        d_cmd = np.array([dx, dy, da])
        if da != 0:
            d_cmd[:2] *= da / (2 * np.sin(da / 2))
        rospy.loginfo("d_cmd:\t" + str(d_cmd))

        v_cmd = np.abs(d_cmd) / np.linalg.norm(d_cmd[:2]) * vel
        if abs(v_cmd[2]) > w:
            v_cmd *= w / abs(v_cmd[2])
        rospy.loginfo("v_cmd:\t" + str(v_cmd))
        cmd = cmd_id + " 162 " + str(d_cmd[0]) + " " + str(d_cmd[1]) + " " + str(d_cmd[2]) + " " + str(v_cmd[0]) + " " + str(v_cmd[1]) + " " + str(v_cmd[2])
        rospy.loginfo("Sending cmd: " + cmd)
        self.pub_cmd.publish(cmd)

    def translate_odometry(self, cmd_id, goal_x, goal_y, vel=0.2):
        goal = np.array([goal_x, goal_y])
        rospy.loginfo("-------NEW LINEAR ODOMETRY MOVEMENT-------")
        rospy.loginfo("Goal:\t" + str(goal))
        while not self.update_coords():
            rospy.sleep(0.05)

        d_map_frame = goal[:2] - self.coords[:2]
        rospy.loginfo("Distance in map frame:\t" + str(d_map_frame))
        d_robot_frame = self.rotation_transform(d_map_frame, -self.coords[2])
        v = np.abs(d_robot_frame) / np.linalg.norm(d_robot_frame) * vel
        cmd = cmd_id + " 162 " + str(d_robot_frame[0]) + ' ' + str(d_robot_frame[1]) + ' 0 ' + str(
            v[0]) + ' ' + str(v[1]) + ' 0'
        rospy.loginfo("Sending cmd: " + cmd)
        self.pub_cmd.publish(cmd)

    def rotate_odometry(self, cmd_id, goal_angle, w=1.0):
        rospy.loginfo("-------NEW ROTATIONAL ODOMETRY MOVEMENT-------")
        rospy.loginfo("Goal angle:\t" + str(goal_angle))
        while not self.update_coords():
            rospy.sleep(0.05)

        delta_angle = goal_angle - self.coords[2]
        rospy.loginfo("Delta angle:\t" + str(delta_angle))
        cmd = cmd_id + " 162 0 0 " + str(delta_angle) + ' 0 0 ' + str(w)
        rospy.loginfo("Sending cmd: " + cmd)
        self.pub_cmd.publish(cmd)

    def update_coords(self):
        try:
            # trans = self.tfBuffer.lookup_transform('map', self.other_robot_name, rospy.Time())
            # q = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
            # angle = euler_from_quaternion(q)[2] % (2 * np.pi)
            # self.other_robot_coords = np.array([trans.transform.translation.x, trans.transform.translation.y, angle])

            (trans, rot) = self.listener.lookupTransform('/map', '/'+self.other_robot_name, rospy.Time(0))
            yaw = tf.transformations.euler_from_quaternion(rot)[2]
            self.other_robot_coords = np.array([trans[0], trans[1], yaw])

        # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            # rospy.loginfo("MotionPlanner failed to find other robot's coordinates.")
            pass
        try:
            # trans = self.tfBuffer.lookup_transform('map', self.robot_name, rospy.Time())
            # q = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
            # angle = euler_from_quaternion(q)[2] % (2 * np.pi)
            # self.coords = np.array([trans.transform.translation.x, trans.transform.translation.y, angle])
            # # rospy.loginfo("Robot coords:\t" + str(self.coords))

            (trans, rot) = self.listener.lookupTransform('/map', '/' + self.robot_name, rospy.Time(0))
            yaw = tf.transformations.euler_from_quaternion(rot)[2]
            self.coords = np.array([trans[0], trans[1], yaw])

            return True
        # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo("MotionPlanner failed to lookup tf2.")
            return False

    # def detected_robots_callback(self, data):
    #     print "detected enemy robots"
    #     if len(data.points) == 0:
    #         self.opponent_robots = np.array([])
    #     else:
    #         self.opponent_robots = np.array([[robot.x, robot.y] for robot in data.points])
    #     # self.robots_upd_time = data.header.stamp

    def distance_to_closest_robot(self):
        ans = np.linalg.norm(self.other_robot_coords)
        # print ans, "OTHER ROBOT COORDS"
        if self.opponent_robots.shape[0] > 0:
            ans = min(ans, np.min(np.linalg.norm(self.opponent_robots)))
        return ans

    def response_callback(self, data):
        if self.cmd_stop_robot_id is None:
            return
        data_splitted = data.data.split()
        if data_splitted[0] == self.cmd_stop_robot_id and data_splitted[1] == "finished":
            self.robot_stopped = True
            rospy.loginfo(data.data)

    def rangefinder_data_callback(self, data):
        self.rangefinder_data = np.array(data.data[:self.NUM_RANGEFINDERS])
        self.rangefinder_status = np.array(data.data[-self.NUM_RANGEFINDERS:])

        line_strip = Marker()
        line_strip.type = line_strip.LINE_STRIP
        line_strip.action = line_strip.ADD
        line_strip.header.frame_id = "/map"

        line_strip.scale.x = 0.025

        line_strip.color.a = 1.0
        line_strip.color.r = 0.0
        line_strip.color.g = 1.0
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
        for rng in range(self.NUM_RANGEFINDERS):
            point = Point()
            point.x, point.y = self.dists(data.data[rng]/1000.0, rng)
            # point.y = rng[1]
            point.z = 0.0
            line_strip.points.append(point)
            if data.data[rng] < self.COLLISION_STOP_DISTANCE:
                # self.rangefinder_status[rng]
                line_strip.color.r = 1.0
            if data.data[rng] < self.COLLISION_STOP_NEIGHBOUR_DISTANCE:
                line_strip.color.g = 0.0

        point = Point()
        point.x, point.y = self.dists(data.data[0] / 1000.0, 0)
        line_strip.points.append(point)
        self.pub_rangefinders.publish(line_strip)

    def dists(self, dist, rng):
        x = self.coords[0] - (self.RF_DISTS[rng]+dist) * np.cos(self.coords[2]+self.RANGEFINDER_ANGLES[rng])
        y = self.coords[1] + (self.RF_DISTS[rng]+dist) * np.sin(self.coords[2]+self.RANGEFINDER_ANGLES[rng])
        return x, y

    def visualize_path(self):
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

        for node in self.path:
            new_point = Point()
            new_point.x = node[0]
            new_point.y = node[1]
            new_point.z = 0.0
            line_strip.points.append(new_point)

        # Publish the Marker
        self.path_viz_pub.publish(line_strip)


if __name__ == "__main__":
    planner = MotionPlanner()
rospy.spin()
