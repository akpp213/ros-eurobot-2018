#!/usr/bin/env python
import rospy
#from sensor_msgs.msg import LaserScan, PointCloud
#from geometry_msgs.msg import Point
import numpy as np


if __name__ == "__main__":
    rospy.init_node("spy", anonymous=True)
    #R = rospy.get_param("R")
    #eps = rospy.get_param("clustering/eps")
    #min_samples = rospy.get_param("clustering/min_samples")
    #rospy.Subscriber("scan", LaserScan, scan_callback, queue_size = 1)
    #pub_center = rospy.Publisher("detected_robots", PointCloud, queue_size=1)
    #if visualization == True:
        #pub_landmarks = rospy.Publisher("landmarks", PointCloud, queue_size=1)

rospy.spin()
