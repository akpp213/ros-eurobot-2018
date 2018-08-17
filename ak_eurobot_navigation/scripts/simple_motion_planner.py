#!/usr/bin/env python
import rospy
import numpy as np
import tf2_ros
import tf
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import String
from threading import Lock


class SimpleMotionPlanner:
    def __init__(self):
        rospy.init_node("simple_motion_planner", anonymous=True)

        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.listener = tf.TransformListener()

        self.robot_name = rospy.get_param("robot_name")
        self.team_color = rospy.get_param("/field/color")
        self.coords = np.array(rospy.get_param('start_' + self.team_color))
        self.coords[:2] /= 1000.0
        self.vel = np.zeros(3)
        self.last_valid_vel = np.zeros(3)

        self.RATE = rospy.get_param("simple_motion_planner/RATE")
        self.XY_GOAL_TOLERANCE = rospy.get_param("simple_motion_planner/XY_GOAL_TOLERANCE")
        self.YAW_GOAL_TOLERANCE = rospy.get_param("simple_motion_planner/YAW_GOAL_TOLERANCE")
        self.V_MAX = rospy.get_param("simple_motion_planner/V_MAX")
        self.W_MAX = rospy.get_param("simple_motion_planner/W_MAX")
        self.ACCELERATION = rospy.get_param("simple_motion_planner/ACCELERATION")
        self.D_DECELERATION = rospy.get_param("simple_motion_planner/D_DECELERATION")
        self.GAMMA = rospy.get_param("simple_motion_planner/GAMMA")
        # for movements to water towers or cube heaps
        self.XY_ACCURATE_GOAL_TOLERANCE = rospy.get_param("simple_motion_planner/XY_ACCURATE_GOAL_TOLERANCE")
        self.YAW_ACCURATE_GOAL_TOLERANCE = rospy.get_param("simple_motion_planner/YAW_ACCURATE_GOAL_TOLERANCE")
        self.D_ACCURATE_DECELERATION = rospy.get_param("simple_motion_planner/D_ACCURATE_DECELERATION")
        self.color = rospy.get_param("/field/color")
        self.robot_name = rospy.get_param("robot_name")
        self.lidar_point = np.array([rospy.get_param("lidar_x"), rospy.get_param("lidar_y"), rospy.get_param("lidar_a")])

        self.mutex = Lock()

        self.cmd_id = None
        self.t_prev = None
        self.goal = None
        self.mode = None

        self.cmd_stop_robot_id = None
        self.robot_stopped = None
        self.stop_id = 0

        self.pub_twist = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.pub_response = rospy.Publisher("response", String, queue_size=1)
        self.pub_cmd = rospy.Publisher("stm_command", String, queue_size=1)
        self.pub_rangefinders = rospy.Publisher("/rangefinder_data", Marker, queue_size=3)
        rospy.Subscriber("move_command", String, self.cmd_callback, queue_size=1)
        rospy.Subscriber("response", String, self.response_callback, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / self.RATE), self.plan)
        print "Finished Setup"

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
        t = rospy.get_time()
        dt = t - self.t_prev
        self.t_prev = t

        # set speed limit
        speed_limit_acs = min(self.V_MAX, np.linalg.norm(self.vel[:2]) + self.ACCELERATION * dt)
        # rospy.loginfo('Acceleration Speed Limit:\t' + str(speed_limit_acs))

        speed_limit_dec = (goal_d / (self.D_ACCURATE_DECELERATION if self.mode == "move_heap" else self.D_DECELERATION)) ** self.GAMMA * self.V_MAX
        # rospy.loginfo('Deceleration Speed Limit:\t' + str(speed_limit_dec))

        speed_limit_collision = [self.V_MAX]
        speed_limit = min(speed_limit_dec, speed_limit_acs, *speed_limit_collision)
        # rospy.loginfo('Final Speed Limit:\t' + str(speed_limit))

        # maximum speed in goal distance proportion
        vel = self.V_MAX * goal_distance / goal_d

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
        vel = self.rotation_transform(vel, -self.coords[2])

        # send cmd: vel in robot frame
        self.set_speed(vel)
        # rospy.loginfo('Vel cmd\t:' + str(vel))

        self.mutex.release()

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

    def set_goal(self, goal, cmd_id, mode='move'):
        rospy.loginfo("Setting a new goal:\t" + str(goal))
        rospy.loginfo("Mode:\t" + str(mode))
        self.cmd_id = cmd_id
        self.mode = mode
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

        elif cmd_type == "stop":
            self.cmd_id = cmd_id
            self.mode = cmd_type
            self.terminate_following()

        self.mutex.release()

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

    def update_coords(self):
        try:
            (trans, rot) = self.listener.lookupTransform('/map', '/' + self.robot_name, rospy.Time(0))
            yaw = tf.transformations.euler_from_quaternion(rot)[2]
            self.coords = np.array([trans[0], trans[1], yaw])
            return True
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo("MotionPlanner failed to lookup tf2.")
            return False

    def response_callback(self, data):
        if self.cmd_stop_robot_id is None:
            return
        data_splitted = data.data.split()
        if data_splitted[0] == self.cmd_stop_robot_id and data_splitted[1] == "finished":
            self.robot_stopped = True
            rospy.loginfo(data.data)


if __name__ == "__main__":
    planner = SimpleMotionPlanner()
rospy.spin()
