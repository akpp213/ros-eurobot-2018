#!/usr/bin/env python
import rospy
import numpy as np
import serial
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import String
import tf

class BarrierNavigator():
    def __init__(self, stm_command_publisher_name, response_pub_name):
        self.sensors = []
        self.phase = 0
        self.sensors_goals = []
        self.dx_quant = 0.002
        self.dy_quant = 0.002
        self.speed_xy = 0.1
        self.speed_z  = 0.1
        self.command_name = str(0xa2)
        self.ready = True
        self.command_pub_name = stm_command_publisher_name
        self.command_pub = rospy.Publisher(stm_command_publisher_name, String, queue_size=10)
        self.response_pub_name = response_pub_name
        self.response_pub = rospy.Publisher(response_pub_name, String, queue_size=2)
        self.sensors_queue = np.zeros((5,6),dtype=np.int)
        self.listener = tf.TransformListener()
        self.i = 0

    def barrier_sensors_callback(self):
        def cb(data):
            sd = np.array(data.data)
            self.sensors_queue = np.roll(self.sensors_queue,-1, axis=0)
            self.sensors_queue[0] = sd
            # print(self.sensors_queue)
            self.sensors = np.sum(self.sensors_queue, axis=0) >= 3
            # rospy.loginfo(self.sensors_queue)
            # rospy.loginfo(self.sensors)
        return cb

    def move_cycle(self, phase):
        while not rospy.is_shutdown():
            if np.all(np.array(self.sensors) - np.array(self.sensors_goals) == 0):
                rospy.loginfo("FINISHED by sensor_goals")
                break
            if np.any(self.sensors[:3]) and phase == 0:
                rospy.loginfo("FINISHED by y sensor PHASE 0")
                break
            if np.all(self.sensors[:3]) and phase == 1:
                rospy.loginfo("FINISHED by y sensor PHASE 1")
                break
            command, dx, dy = self.get_command(phase)
            if dx == 0 and dy == 0:
                rospy.loginfo("FINISED by dx dy")
                break
            rospy.loginfo(command)
            self.command_pub.publish(command)
            self.wait_for_movement("MOVEODOM"+str(self.i))
    
    def move_cycle_one(self, phases_x, phases_y, case):
        i_x = 0
        i_y = 0
        phase_y = phases_y[0]
        phase_x = phases_x[0]
        allowed_mask = self.get_allowed_mask(case)
        while not rospy.is_shutdown():
            phase_0_cond = np.any(self.sensors[:3]*allowed_mask) and phase_y == 0
            phase_1_cond = np.all(self.sensors[:3]*allowed_mask == 0) and phase_y == 1
            if  phase_0_cond or phase_1_cond: 
                if i_y == len(phases_y)-1:
                    phase_y = -1
                else:
                    i_y += 1
                    phase_y = phases_y[i_y]
            phase_0_cond = np.any(self.sensors[3:]*allowed_mask) and phase_x == 0
            phase_1_cond = np.all(self.sensors[3:]*allowed_mask == 0) and phase_x == 1
            if  phase_0_cond or phase_1_cond: 
                if i_x == len(phases_x)-1:
                    phase_x = -1
                else:
                    i_x += 1
                    phase_x = phases_x[i_x] if sum(np.array([1,1,0])*allowed_mask) == 1 else 1 - phases_x[i_x]

            command, dx, dy = self.get_command_one(phase_x,phase_y,case)
            rospy.loginfo(str(phase_x) + ' ' +  str(dx) + ' ' +  str(phase_y)+ ' ' + str(dy))
            if dx == 0 and dy == 0:
                rospy.loginfo("FINISED by dx dy")
                break
            rospy.loginfo(command)
            self.command_pub.publish(command)
            self.wait_for_movement("MOVEODOM"+str(self.i))
    


    def get_allowed_mask(self, case):
        if case == 0:
            return np.array([1,1,1])
        if case == 7:
            return np.array([1,0,0])
        if case == 8:
            return np.array([0,1,0])
        if case == 9:
            return np.array([0,0,1])


    def start_command_callback(self):
        def cb(data):
            data_splitted = data.data.split()
            action_type = data_splitted[1]
                        
            if action_type == "MOVETOHEAP":
                rospy.loginfo("Receive command " + data.data)

                rospy.sleep(1)
                
                (trans, rot) = self.listener.lookupTransform('/map', '/main_robot', rospy.Time(0))
                yaw = tf.transformations.euler_from_quaternion(rot)[2]
                yaw = yaw % (np.pi / 2)
                a = - (yaw if yaw < np.pi / 4 else yaw - np.pi / 2)

                rospy.loginfo("rotation on angle " + str(a))
                
                
                def tm(e):
                    self.command_pub.publish("move_heap_121 162 0 0 " + str(a) + ' 0 0 0.5')
                rospy.Timer(rospy.Duration(0.1),tm,oneshot=True)
                self.wait_for_movement("move_heap_121")
                rospy.loginfo("rotation finished")


                case = int(data_splitted[2])
                rospy.loginfo(case)
                yellow_fix = case // 20
                case = case % 20
                rospy.loginfo(str(yellow_fix) + ' ' +str(case))
                rospy.sleep(0.5)
                rospy.loginfo("Start move to heap by barrier sensors")
                
                mask = self.get_allowed_mask(case)

                if case == 0:
                    if not np.any(self.sensors[:3]):
                        self.set_sensors_goals(0)
                        self.move_cycle(0)
                        rospy.loginfo("PHASE 0 FINISHED")
                    
                    # rospy.sleep(5)
                    else:
                        self.set_sensors_goals(1)
                        self.move_cycle(1)
                        rospy.loginfo("PHASE 1 FINISHED")
                        
                        self.set_sensors_goals(0)
                        self.move_cycle(0)
                        rospy.loginfo("PHASE 0 FINISHED")

                if case in [7,8,9]:
                    mask = self.get_allowed_mask(case)
                    phases_y = [0] if sum(self.sensors[:3]*mask) == 0 else [1,0]
                    phases_x = [0] if sum(self.sensors[3:]*mask) == 0 else [1,0]
                    rospy.loginfo(phases_x)
                    rospy.loginfo(phases_y)
                    self.move_cycle_one(phases_x,phases_y,case)

                if case == 9 and yellow_fix == 1:
                    cmd, dx, dy = self.get_command_dx_dy(0.004, 0)
                    self.command_pub.publish(cmd)
                    self.wait_for_movement("MOVEODOM"+str(self.i))

                self.response_pub.publish(data_splitted[0] + ' finished')

        return cb

    def wait_for_movement(self,name="MOVEODOM"):
        self.has_moved = False
        def cb(msg):
            if msg.data == name + " finished":
                self.has_moved = True
        rospy.Subscriber(self.response_pub_name, String, cb)
        while not self.has_moved:
            rospy.sleep(0.1)
            # msg = rospy.wait_for_message(self.response_pub_name, String, timeout=3)
            # if msg.data == name + " finished":
            #    break

    def set_sensors_goals(self, phase):
        if phase == 0:
            self.sensors_goals = np.zeros((6), dtype=np.int)
            self.sensors_goals[1] = 1
        if phase == 1:
            self.sensors_goals = np.zeros((6), dtype=np.int)

    def get_command(self,phase):
        rospy.loginfo(self.sensors)
        ds = self.sensors - self.sensors_goals
        dx = -ds[5]*self.dx_quant + (ds[3] or ds[4])*self.dx_quant
        # rospy.loginfo(np.array(self.sensors[:3]))
        # rospy.loginfo(np.any(np.array(self.sensors[:3])))
        rospy.loginfo(int(np.any(ds[:3])))
        dy = 0
        if phase == 0:
            dy = +self.dy_quant
        if phase == 1 and np.any(self.sensors[:3]):
            dy = -self.dy_quant
        dz = 0
        self.i+=1
        command_string = "MOVEODOM"+str(self.i)+" " + self.command_name + ' '
        command_string += str(dx) + ' ' + str(dy) + ' ' + str(dz) + ' '
        command_string += str(self.speed_xy) + ' ' + str(self.speed_xy) + ' ' + str(self.speed_z)
        return command_string, dx ,dy

    def get_command_dx_dy(self, dx, dy):
        dz = 0
        self.i+=1
        command_string = "MOVEODOM"+str(self.i)+" " + self.command_name + ' '
        command_string += str(dx) + ' ' + str(dy) + ' ' + str(dz) + ' '
        command_string += str(self.speed_xy) + ' ' + str(self.speed_xy) + ' ' + str(self.speed_z)
        return command_string, dx ,dy


    def get_command_one(self, phase_x, phase_y, case):
        rospy.loginfo(self.sensors)

        if case in [7,8,9]:
            if phase_x == 0:
                dx = -self.dx_quant 
            elif phase_x == 1:
                dx = +self.dx_quant
            else:
                dx = 0
            
            if case == 9:
                dx = -dx
            if phase_y == 0:
                dy = +self.dy_quant
            elif phase_y == 1:
                dy = -self.dy_quant
            else:
                dy = 0
            return self.get_command_dx_dy(dx,dy)

        


def get_sensor_goal_values(sensor_data):
    # 0 - move to y+ (biggest x)
    # 1 - move to y+ (middle x)
    # 2 - move to y+ (smallest x)
    # 3 - move to x- (0)
    # 4 - move to x- (1)
    # 5 - move to x+ (2)
    pass

if __name__ == '__main__':
    try:
        rospy.init_node('barrier_move_node', anonymous=True)
        bn = BarrierNavigator("/main_robot/stm_command", "main_robot/response")
        # pub_command = rospy.Publisher("/main_robot/stm_command", String, queue_size=10)
        rospy.Subscriber("/main_robot/move_command", String, bn.start_command_callback())
        rospy.Subscriber("/main_robot/barrier_movement", Int32MultiArray, bn.barrier_sensors_callback())
        # pub_response = rospy.Publisher("/main_robot/response", String, queue_size=2)
         
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
