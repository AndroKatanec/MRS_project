#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Controlling a Swarm of Sphero Robots in a Simulator Stage using Reynolds Rules
"""

import rospy
import tf2_ros
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import message_filters

num_of_robots = rospy.get_param("/num_of_robots")

def wall_check(pose):
    flag = False
    distance_to_the_wall = 4
    if (abs(pose.x) > distance_to_the_wall) or (abs(pose.y) > distance_to_the_wall):
        flag = True

    return flag

def callback(msg):
    robot_id = int(list(msg._connection_header['topic'])[7])  #get the id of the boid that this message came from
    last_positions[robot_id] = msg #update last known positions
    move = Twist()
    if wall_check(msg.pose.pose.position):
        move.angular.z = 10
        move.linear.x=0.5
    else:
        move.linear.x=1
        move.angular.z=0
    pub[robot_id].publish(move)
    return None

def main():
    rospy.init_node('control_node')
    global pub
    global last_positions #list of last known positions of each boid
    last_positions = [Odometry for i in range(num_of_robots)]
    pub = [rospy.Publisher(f'/robot_{i}/cmd_vel', Twist, queue_size=2) for i in range(num_of_robots)]
    [rospy.Subscriber("/robot_{}/odom".format(i), Odometry, callback) for i in range(num_of_robots)]
    rate = rospy.Rate(10) #10hz
    rospy.spin()


if __name__ == '__main__':
    main()