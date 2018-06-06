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

    # current linear and angular goal distance
    goal_distance = np.zeros(3)
    goal_distance = self.distance(self.coords, self.goal)
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
