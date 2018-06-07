#!/usr/bin/env python
import rospy
import numpy as np
import tf2_ros
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import String
from threading import Lock
from std_msgs.msg import Int32MultiArray
from visualization_msgs.msg import MarkerArray, Marker

class Node(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0
        
    def distance(self, random_point):
        distance = np.sqrt((self.x-random_point[0])**2 + (self.y - random_point[1])**2)
        return distance

class MotionPlanner:
    def __init__(self):
        rospy.init_node("motion_planner", anonymous=True)

        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)

        self.robot_name = rospy.get_param("robot_name")
        self.team_color = rospy.get_param("/field/color")
        self.coords = np.array(rospy.get_param('start_' + self.team_color))
        self.coords[:2] /= 1000.0
        self.vel = np.zeros(3)

        self.RATE = rospy.get_param("motion_planner/RATE")
        self.XY_GOAL_TOLERANCE = rospy.get_param("motion_planner/XY_GOAL_TOLERANCE")
        self.YAW_GOAL_TOLERANCE = rospy.get_param("motion_planner/YAW_GOAL_TOLERANCE")
        self.V_MAX = rospy.get_param("motion_planner/V_MAX")
        self.W_MAX = rospy.get_param("motion_planner/W_MAX")
        self.ACCELERATION = rospy.get_param("motion_planner/ACCELERATION")
        self.D_DECELERATION = rospy.get_param("motion_planner/D_DECELERATION")
        self.GAMMA = rospy.get_param("motion_planner/GAMMA")
        # for movements to water towers or cube heaps
        self.XY_ACCURATE_GOAL_TOLERANCE = rospy.get_param("motion_planner/XY_ACCURATE_GOAL_TOLERANCE")
        self.YAW_ACCURATE_GOAL_TOLERANCE = rospy.get_param("motion_planner/YAW_ACCURATE_GOAL_TOLERANCE")
        self.D_ACCURATE_DECELERATION = rospy.get_param("motion_planner/D_ACCURATE_DECELERATION")
        self.NUM_RANGEFINDERS = rospy.get_param("motion_planner/NUM_RANGEFINDERS")
        self.COLLISION_STOP_DISTANCE = rospy.get_param("motion_planner/COLLISION_STOP_DISTANCE")
        self.COLLISION_STOP_NEIGHBOUR_DISTANCE = rospy.get_param("motion_planner/COLLISION_STOP_NEIGHBOUR_DISTANCE")
        self.COLLISION_GAMMA = rospy.get_param("motion_planner/COLLISION_GAMMA")
        
        self.MAX_RRT_ITERS = rospy.get_param("motion_planner/MAX_RRT_ITERS") #1000
        self.STEP = rospy.get_param("motion_planner/STEP") #0.1
        self.GAMMA_RRT = rospy.get_param("motion_planner/GAMMA_RRT") #3
        self.EPSILON_GOAL = rospy.get_param("motion_planner/EPSILON_GOAL") #0.2
        self.PATH_WIDTH = rospy.get_param("motion_planner/PATH_WIDTH") #0.1

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

        self.o_map = None
        self.secondary_o_map = None
        self.nodes = None
        self.nodes_secondary = None
        self.resolution = None
        
        self.cmd_id = None
        self.t_prev = None
        self.goal = None
        self.mode = None
        self.rangefinder_data = np.zeros(self.NUM_RANGEFINDERS)
        self.rangefinder_status = np.zeros(self.NUM_RANGEFINDERS)
        self.active_rangefinder_zones = np.ones(3, dtype="int")

        self.cmd_stop_robot_id = None
        self.stop_id = 0

        self.pub_twist = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.pub_response = rospy.Publisher("response", String, queue_size=10)
        self.pub_cmd = rospy.Publisher("stm_command", String, queue_size=1)
        rospy.Subscriber("move_command", String, self.cmd_callback, queue_size=1)
        rospy.Subscriber("response", String, self.response_callback, queue_size=1)
        rospy.Subscriber("barrier_rangefinders_data", Int32MultiArray, self.rangefinder_data_callback, queue_size=1)
        rospy.Subscriber("/main_robot/map", OccupancyGrid, self.update_main_map, queue_size=1)
        rospy.Subscriber("/secondary_robot/map", OccupancyGrid, self.update_secondary_map, queue_size=1)
        # start the main timer that will follow given goal points
        rospy.Timer(rospy.Duration(1.0 / self.RATE), self.plan)
        
    def update_main_map(self, msg):
        self.x0 = msg.info.origin.position.x
        self.y0 = msg.info.origin.position.y
        self.map_width = msg.info.width * msg.info.resolution
        self.map_height = msg.info.height * msg.info.resolution
        self.resolution = msg.info.resolution
        self.shape_map = (msg.info.width, msg.info.height)
        array255 = msg.data.reshape((msg.info.height, msg.info.width))
        self.permissible_region = np.ones_like(array255, dtype=bool)
        self.permissible_region[array255==100]=0 #setting occupied regions (100) to 0. Unoccupied regions are a 1 
        
    def update_secondary_map(self, msg):
        self.s_x0 = msg.info.origin.position.x
        self.s_y0 = msg.info.origin.position.y
        self.s_map_width = msg.info.width * msg.info.resolution #meters
        self.s_map_height = msg.info.height * msg.info.resolution #meters
        self.s_resolution = msg.info.resolution
        self.s_shape_map = (msg.info.width, msg.info.height) #pixels
        array255 = msg.data.reshape((msg.info.height, msg.info.width))
        self.s_permissible_region = np.ones_like(array255, dtype=bool)
        self.s_permissible_region[array255==100]=0 #setting occupied regions (100) to 0. Unoccupied regions are a 1 
    
    def plan(self):
        self.mutex.acquire()
        
        if self.cmd_id is None:
            self.mutex.release()
            return

        rospy.loginfo("-------NEW MOTION PLANNING ITERATION-------")

        if not self.update_coords():
            self.set_speed(np.zeros(3))
            rospy.loginfo('Stopping because of tf2 lookup failure.')
            self.mutex.release()
            return
            
        if not self.goal:
            self.set_speed(np.zeros(3))
            rospy.loginfo('Stopping because there is no goal yet')
            self.mutex.release()
            return

        # current linear and angular goal distance
        goal_distance = np.zeros(3)
        goal_distance = self.goal - self.coords
        rospy.loginfo('Goal distance:\t' + str(goal_distance))
        goal_d = np.linalg.norm(goal_distance[:2])
        rospy.loginfo('Goal d:\t' + str(goal_d))

        # stop and publish response if we have reached the goal with the given tolerance
        if ((self.mode == 'move' or self.mode == 'move_fast') and goal_d < self.XY_GOAL_TOLERANCE and goal_distance[2] < self.YAW_GOAL_TOLERANCE) or (self.mode == 'move_heap' and goal_d < self.XY_ACCURATE_GOAL_TOLERANCE and goal_distance[2] < self.YAW_ACCURATE_GOAL_TOLERANCE):
            rospy.loginfo(self.cmd_id + " finished, reached the goal")
            self.terminate_following()
            self.mutex.release()
            return
            
        active_rangefinders, stop_ranges = self.choose_active_rangefinders()
        rospy.loginfo("Active rangefinders: " + str(active_rangefinders) + "\t with ranges: " + str(stop_ranges))
        rospy.loginfo("Active rangefinders data: " + str(self.rangefinder_data[active_rangefinders]) + ". Status: " + str(self.rangefinder_status[active_rangefinders]))
        
        # CHOOSE VELOCITY COMMAND.

        if np.any(self.rangefinder_status[active_rangefinders]):
            # collision avoidance
            rospy.loginfo('EMERGENCY STOP: collision avoidance. Active rangefinder error.')
            vel_robot_frame = np.zeros(3)
        else:
            t = rospy.get_time()
            dt = t - self.t_prev
            self.t_prev = t

            # set speed limit
            speed_limit_acs = min(self.V_MAX, np.linalg.norm(self.vel[:2]) + self.ACCELERATION * dt)
            rospy.loginfo('Acceleration Speed Limit:\t' + str(speed_limit_acs))

            speed_limit_dec = (goal_d / (self.D_ACCURATE_DECELERATION if self.mode == "move_heap" else self.D_DECELERATION)) ** self.GAMMA * self.V_MAX
            rospy.loginfo('Deceleration Speed Limit:\t' + str(speed_limit_dec))

            if active_rangefinders.shape[0] != 0:
                speed_limit_collision = []
                for i in range(active_rangefinders.shape[0]):
                    if self.rangefinder_data[active_rangefinders[i]] < stop_ranges[i]:
                        speed_limit_collision.append(0)
                    else:
                        speed_limit_collision.append(((self.rangefinder_data[active_rangefinders[i]] - stop_ranges[i]) / (255.0 - stop_ranges[i])) ** self.COLLISION_GAMMA * self.V_MAX)
                rospy.loginfo('Collision Avoidance  Speed Limit:\t' + str(speed_limit_collision))
            else:
                speed_limit_collision = [self.V_MAX]
           
            speed_limit = min(speed_limit_dec, speed_limit_acs, *speed_limit_collision)
            rospy.loginfo('Final Speed Limit:\t' + str(speed_limit))
            
            #Find a path and then follow it
            path = self.find_path_rrtstar()
            vel = self.follow_path(path)

            ## maximum speed in goal distance proportion
            #vel = self.V_MAX * goal_distance / goal_d
            if abs(vel[2]) > self.W_MAX:
                vel *= self.W_MAX / abs(vel[2])
            rospy.loginfo('Vel before speed limit\t:' + str(vel))

            # apply speed limit
            v_abs = np.linalg.norm(vel[:2])
            rospy.loginfo('Vel abs before speed limit\t:' + str(vel))
            if v_abs > speed_limit:
                vel *= speed_limit / v_abs
            rospy.loginfo('Vel after speed limit\t:' + str(vel))
            
            # vel to robot frame
            vel_robot_frame = self.rotation_transform(vel, -self.coords[2])

        # send cmd: vel in robot frame
        self.set_speed(vel_robot_frame)
        rospy.loginfo('Vel cmd\t:' + str(vel_robot_frame))

        self.mutex.release()
        
    def find_path_astar(self):
        pass

    def find_path_rrtstar(self):
        x,y,th = self.coords
        start_node = Node(x,y)
        self.nodes.append(start_node)
        j = 0
        path_found = False
        while not(path_found) and (j < self.MAX_RRT_ITERS):

            # random sample
            if j%25 == 0:
                # sample goal location every 25 iterations
                rnd = [self.goal[0], self.goal[1]]
            else:
                rnd = [self.x0 + self.map_width*np.random.uniform(), self.y0 + self.map_height*np.random.uniform()]

            # find the nearest node
            nn_idx = self.nearest_node(rnd)
            x_near = self.nodes[nn_idx]

            # expand the tree
            x_new, theta = self.steer_to_node(x_near,rnd) #tree goes from x_near to x_new which points in the direction to rnd
            x_new.parent = nn_idx

            print "length nodes"
            print j
            print ""

            # check for obstacle collisions
            if self.obstacle_free(x_near, x_new, theta):
		
                # find close nodes
                near_idxs = self.find_near_nodes(x_new)


                # find the best parent for the node
                x_new = self.choose_parent(x_new, near_idxs)

                # add new node and rewire
                self.nodes.append(x_new)
                self.rewire(x_new, near_idxs)

                # check if sample point is in goal region
                dx = x_new.x - self.goal[0]
                dy = x_new.y - self.goal[1]
                d = np.sqrt(dx**2 + dy**2)
                if d <= self.EPSILON_GOAL:
                    # construct path
                    last_idx = self.get_last_idx()
                    if last_idx is None:
                        return None
                    path = self.get_path(last_idx)
                    self.path = path


                    #for point in self.path:
                    #	pt_obj = Point()
                    #	pt_obj.x = point[0]
                    #	pt_obj.y = point[1]

                    #	self.trajectory.addPoint(pt_obj)
                    #self.publish_trajectory()

                    #self.visualize()
                    return self.path
			
            j += 1

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
        new_x = x_near.x + self.STEP*math.cos(theta)
        new_y = x_near.y + self.STEP*math.sin(theta)

        x_new = Node(new_x, new_y)
        x_new.cost += self.STEP

        return x_new, theta

    def obstacle_free(self, nearest_node, new_node, theta):
        """ Checks if the path from x_near to x_new is obstacle free """

        dx = math.sin(theta)*self.PATH_WIDTH
        dy = math.cos(theta)*self.PATH_WIDTH

        # line = [[nearest_node.x, nearest_node.y], [new_node.x, new_node.y]]

        if (nearest_node.x < new_node.x):
            bound0 = ((nearest_node.x, nearest_node.y), (new_node.x, new_node.y))
            bound1 = ((nearest_node.x+dx, nearest_node.y-dy), (new_node.x+dx, new_node.y-dy))
            bound2 = ((nearest_node.x-dx, nearest_node.y+dy), (new_node.x-dx, new_node.y+dy))
        else:
            bound0 = ((new_node.x, new_node.y), (nearest_node.x, nearest_node.y))
            bound1 = ((new_node.x+dx, new_node.y-dy), (nearest_node.x+dx, nearest_node.y-dy))
            bound2 = ((new_node.x-dx, new_node.y+dy), (nearest_node.x-dx, nearest_node.y+dy))


        if (self.line_collision(bound0) or self.line_collision(bound1) or self.line_collision(bound2)):
            return False
        else:
            return True

    def line_collision(self, line):
        """ Checks if line collides with obstacles in the map"""

        # discretize values of x and y along line according to map using Bresemham's alg


        x_ind = np.arange(np.ceil((line[0][0])/self.resolution),np.ceil((line[1][0])/self.resolution + 1), dtype = int)
        y_ind = []

        dx = max(1,np.ceil(line[1][0]/self.resolution) - np.ceil(line[0][0]/self.resolution))
        dy = np.ceil(line[1][1]/self.resolution) - np.ceil(line[0][1]/self.resolution)
        deltaerr = abs(dy/dx)
        error = .0

        y = int(np.ceil((line[0][1])/self.resolution))
        for x in x_ind:
            y_ind.append(y)
            error = error + deltaerr
            while error >= 0.5:
                y += np.sign(dy)*1
                error += -1

        y_ind = np.array(y_ind)
        # check if any cell along the line contains an obstacle
        for i in range(len(x_ind)):

            row = min([int(-self.y0/self.resolution + y_ind[i]), self.shape_map[0] - 1])
            column = min([int(-self.x0/self.resolution + x_ind[i]), self.shape_map[1]-1])
            # print row, column, "rowcol"
            # print self.map_width, self.map_height
            if self.permissible_region[row,column] == 0:                
                return True
        return False

    def find_near_nodes(self, x_new):
        length_nodes = len(self.nodes)
        r = self.GAMMA_RRT*math.sqrt((math.log(length_nodes)/length_nodes))
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
            theta = math.atan2(dy,dx)

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
                theta = math.atan2(dy,dx)
                if self.obstacle_free(node,x_new,theta):
                    node.parent = n - 1
                    node.cost = scost

    def get_last_idx(self):
        dlist = []
        for node in self.nodes:
            d = node.distance((self.goal[0], self.goal[1]))
            dlist.append(d)
        goal_idxs = [dlist.index(i) for i in dlist if i <= self.step]

        if len(goal_idxs) == 0:
            return None

        mincost = min([self.nodes[i].cost for i in goal_idxs])

        for i in goal_idxs:
            if self.nodes[i].cost == mincost:
                return i

        return None

    def get_path(self, last_idx):
        path = [(self.goal[0], self.goal[1])]
        while self.nodes[last_idx].parent is not None:
            node = self.nodes[last_idx]
            path.append((node.x, node.y))
            last_idx = node.parent
        path.append((self.coords[0], self.coords[1]))
        path.reverse()
        return path

    def visualize(self):
        if self.nodes != []:
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

            plt.clf()

            for node in self.nodes:
                if node.parent is not None:
                    plt.plot([node.x, self.nodes[node.parent].x], [node.y, self.nodes[node.parent].y], "-g")

            x,y = self.coords[:2]
            plt.plot(x,y, "ob")
            x,y = self.goal[:2]
            plt.plot(x,y, "og")
            plt.axis([-26, self.max_x-26, -11, self.max_y-11])
            plt.pause(0.01)

            for i = np.shape(self.permissible_region)[0]:
                for j = np.shape(self.permissible_region)[1]:
                    if permissible_region[i,j] == 0:
                        plt.plot(i*self.resolution, j*self.resolution, "rx")

            # print("plotting")
            # for node in self.path:
            # 	print(node)
            # 	new_point = Point()
            # 	new_point.x = node[0]
            # 	new_point.y = node[1]
            # 	new_point.z = 0.0
            # 	line_strip.points.append(new_point)

            # # Publish the Marker
            # self.viz_pub.publish(line_strip)


    def follow_path(self,path):

        return vel
        
    def choose_active_rangefinders(self):
        if self.robot_name == "secondary_robot":
            d_map_frame = self.goal[:2] - self.coords[:2]
            d_robot_frame = self.rotation_transform(d_map_frame, -self.coords[2])
            rospy.loginfo("Choosing active rangefinders")
            goal_angle = (np.arctan2(d_robot_frame[1], d_robot_frame[0]) % (2 * np.pi))
            rospy.loginfo("Goal angle in robot frame:\t" + str(goal_angle))
            n = int(round((np.pi / 2 - goal_angle) / (np.pi / 4))) % 8
            rospy.loginfo("Closest rangefinder: " + str(n))
            return np.array([n - 1, n, n + 1])[np.where(self.active_rangefinder_zones)] % 8, np.array([self.COLLISION_STOP_NEIGHBOUR_DISTANCE, self.COLLISION_STOP_DISTANCE, self.COLLISION_STOP_NEIGHBOUR_DISTANCE])[np.where(self.active_rangefinder_zones)]
        else:
            d_map_frame = self.goal[:2] - self.coords[:2]
            d_robot_frame = self.rotation_transform(d_map_frame, -self.coords[2])
            rospy.loginfo("Choosing active rangefinders")
            goal_angle = (np.arctan2(d_robot_frame[1], d_robot_frame[0]) % (2 * np.pi))
            k = int(round((-np.pi / 2 + goal_angle) / (np.pi / 4))) % 8
            rospy.loginfo("Goal angle in robot frame:\t" + str(goal_angle) + "\t; k = " + str(k))
            if k == 0:
                rospy.loginfo("Closest rangefinders: 9, 0")
                rf = np.array([8, 9, 0, 1])            
            elif k < 4:
                n = k
                rospy.loginfo("Closest rangefinder: " + str(n))
                rf = np.array([n - 1, n, n + 1]) % 10
            elif k == 4:
                rospy.loginfo("Closest rangefinders: 4, 5")
                rf = np.array([3, 4, 5, 6])
            elif k > 4:
                n = k + 1
                rospy.loginfo("Closest rangefinder: " + str(n))
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
        self.rangefinder_data = np.zeros(self.NUM_RANGEFINDERS)
        self.rangefinder_status = np.zeros(self.NUM_RANGEFINDERS)
        self.active_rangefinder_zones = np.ones(3, dtype="int")

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

    def set_goal(self, goal, cmd_id, mode='move', active_rangefinder_zones = np.ones(3, dtype="int")):
        rospy.loginfo("Setting a new goal:\t" + str(goal))
        rospy.loginfo("Mode:\t" + str(mode))
        self.cmd_id = cmd_id
        self.mode = mode
        self.active_rangefinder_zones = active_rangefinder_zones
        rospy.loginfo("Active Rangefinder Zones: " + str(active_rangefinder_zones))
        self.t_prev = rospy.get_time()
        self.goal = goal

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
        rospy.loginfo("========================================================================================================")
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


    def move_heap(self, cmd_id, n, active_rangefinder_zones = np.ones(3, dtype="int")):
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
            trans = self.tfBuffer.lookup_transform('map', self.robot_name, rospy.Time())
            q = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
            angle = euler_from_quaternion(q)[2] % (2 * np.pi)
            self.coords = np.array([trans.transform.translation.x, trans.transform.translation.y, angle])
            rospy.loginfo("Robot coords:\t" + str(self.coords))
            return True
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.loginfo("MotionPlanner failed to lookup tf2.")
            return False

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


if __name__ == "__main__":
    planner = MotionPlanner()
rospy.spin()
