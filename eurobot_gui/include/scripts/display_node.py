#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import sys
import time
import random
import numpy as np
from Tkinter import *
COLORS = np.array([[0, 123, 176], [208, 90, 40], [28, 28, 32], [96, 153, 59], [247, 181, 0]], dtype=np.uint8)
COLORS_NAME = ["blue", "orange", "black", "green", "yellow"]

class App:
    def __init__(self, master):

        frame = Frame(master, bg="white", colormap="new")
        frame.pack()
        frame1 = Frame(frame, bg="white", colormap="new")
        frame1.pack(side="left")
        frame2 = Frame(frame, bg="white", colormap="new")
        frame2.pack(side="right")

        Label(frame1, bg="white", height=1, width=8, font=("Helvetica", 32), text="POINTS").pack(side="top")
        self.points = StringVar()
        Label(frame1, bg="white", height=1, width=5, textvariable=self.points, font=("Helvetica", 128)).pack(side="top")
        self.points.set("0")

        Label(frame2, bg="white", height=1, width=13, font=("Helvetica", 32), text="SIDE").pack(side="top")
        Label(frame2, bg='#%02x%02x%02x' % tuple(COLORS[1]), height=1, width=13, font=("Helvetica", 32)).pack(side="top")
        Label(frame2, bg="white", height=1, width=13, font=("Helvetica", 32), text="COLORS PLAN").pack(side="top")
        self.w = Canvas(frame2, bg="white", width=300, height=100)
        self.rect1 = self.w.create_rectangle(0, 0, 100, 100, fill="red", outline='white')
        self.rect2 = self.w.create_rectangle(100, 0, 200, 100, fill="red", outline='white')
        self.rect3 = self.w.create_rectangle(200, 0, 300, 100, fill="red", outline='white')
        self.w.pack(side="top")
        self.set_plan('orange black green')

    def points_callback(self, data):
        self.points.set(data.data)

    def plan_callback(self, data):
        self.set_plan(data.data)

    def set_plan(self, plan):
        splitted_plan = plan.split()
        self.w.itemconfig(self.rect1, fill='#%02x%02x%02x' % tuple(COLORS[COLORS_NAME.index(splitted_plan[0])]))
        self.w.itemconfig(self.rect2, fill='#%02x%02x%02x' % tuple(COLORS[COLORS_NAME.index(splitted_plan[1])]))
        self.w.itemconfig(self.rect3, fill='#%02x%02x%02x' % tuple(COLORS[COLORS_NAME.index(splitted_plan[2])]))


if __name__ == '__main__':
    rospy.init_node("display_node")
    root = Tk()
    root.title("Eurobot Reset")
    # root.geometry("500x400")
    app = App(root)

    rospy.Subscriber("/server/point", String, app.points_callback)
    rospy.Subscriber("/server/plan", String, app.plan_callback)
    rate = rospy.Rate(100)
    rospy.loginfo("Start display")

    root.mainloop()
    root.destroy()




