#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import random
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry, OccupancyGrid
import message_filters
from time import sleep
from numpy import linalg
import numpy as np
from typing import List
from math import atan


def quaternion_to_euler(w, x, y, z):
    ysqr = y * y
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2 > +1.0, +1.0, t2)
    #t2 = +1.0 if t2 > +1.0 else t2
    t2 = np.where(t2 < -1.0, -1.0, t2)
    #t2 = -1.0 if t2 < -1.0 else t2
    Y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.arctan2(t3, t4)
    return X, Y, Z


def limit_speed(speed, limit):
    if speed > limit:
        speed = limit
    if speed < -limit:
        speed = -limit

    return speed
    


class Simulation():
    def __init__(self) -> None:
        self.map = []
        self.map_resolution = 0
        self.map_origin = Pose()

        self.enable_setpoints = False
        self.current_goal=0
        self.tocke =[[-4,-4],[3,-3],[3, 3],[-4, 3],[-3,-2]] #set_points

        self.radius = 2  # radius of FoV of a boid
        self.avoiding_distance_to_the_wall = 3
        self.fov = 2.5 # +- field of view of a boid in radians
        self.num_of_robots = rospy.get_param("/num_of_robots")
        self.initialization = [False for i in range(self.num_of_robots)]
        # set initial velocity
        odom = Odometry()
        odom.twist.twist.linear.x = 1.0
        odom.twist.twist.linear.y = 1.0

        self.last_positions = [odom for i in range(self.num_of_robots)]
        self.pub = [rospy.Publisher(
            f'/robot_{i}/cmd_vel', Twist, queue_size=2) for i in range(self.num_of_robots)]
        sleep(3)
        self.sub = [rospy.Subscriber(
            "/robot_{}/odom".format(i), Odometry, self.callback) for i in range(self.num_of_robots)]
        self.map_sub = rospy.Subscriber(
            '/map', OccupancyGrid, self.map_callback, queue_size=1)

    def map_callback(self, msg) -> None:
        self.map = []
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin.position
        # int(self.radius/self.map_resolution)
        height = msg.info.height
        width = msg.info.width
        for i in range(height - 1, -1, -1):  # Backwards iteration
            self.map.append(msg.data[i * width: (i + 1) * width])

    def callback(self, msg) -> None:
        # get the id of the boid that this message came from
        robot_id = int(list(msg._connection_header['topic'])[7])
        self.last_positions[robot_id] = msg  # update last known positions
        neoighbors_id = self.get_neighbours_ids(robot_id, self.last_positions)
        odom_i = self.last_positions[robot_id]
        move = Twist()
        k_cohesion = 0.3 #0.4
        k_separation = 0.15 #0.3
        k_alignment = 1 #0.15

        if len(neoighbors_id) > 0:  # check if there are nearby boids to apply Reynolds rules
            cohesion_vec = self.cohesion(robot_id, neoighbors_id)
            separation_vec = self.separation(robot_id, neoighbors_id)
            alignment_vec = self.alignment(robot_id, neoighbors_id)
            if self.enable_setpoints:
                move.linear.x = k_alignment*alignment_vec[0] + k_cohesion*cohesion_vec[0] + k_separation*separation_vec[0]
                move.linear.y = k_alignment*alignment_vec[1] + k_cohesion*cohesion_vec[1] + k_separation*separation_vec[1]
            else:
                move.linear.x = self.last_positions[robot_id].twist.twist.linear.x + k_alignment*alignment_vec[0] + k_cohesion*cohesion_vec[0] + k_separation*separation_vec[0]
                move.linear.y = self.last_positions[robot_id].twist.twist.linear.y + k_alignment*alignment_vec[1] + k_cohesion*cohesion_vec[1] + k_separation*separation_vec[1]

            
        if self.enable_setpoints:
            leading_speed = 0.4
            robots_leaders = [0, 4]
            if robot_id in robots_leaders:
                if self.current_goal >= len(self.tocke):
                    smjer_x = 0
                    smjer_y = 0
                elif abs(odom_i.pose.pose.position.x - self.tocke[self.current_goal][0]) > 0.3 or abs(odom_i.pose.pose.position.y - self.tocke[self.current_goal][1]) > 0.3:
                    smjer_x=leading_speed*-(odom_i.pose.pose.position.x - self.tocke[self.current_goal][0])/(np.sqrt((odom_i.pose.pose.position.x - self.tocke[self.current_goal][0])**2 + (odom_i.pose.pose.position.y - self.tocke[self.current_goal][1])**2))
                    smjer_y=leading_speed*-(odom_i.pose.pose.position.y - self.tocke[self.current_goal][1])/(np.sqrt((odom_i.pose.pose.position.x - self.tocke[self.current_goal][0])**2 + (odom_i.pose.pose.position.y - self.tocke[self.current_goal][1])**2))
                else:
                    print('Dosao je')
                    self.current_goal+=1

                if len(neoighbors_id) > 0:
                    cohesion_vec = self.cohesion(robot_id, neoighbors_id)
                    separation_vec = self.separation(robot_id, neoighbors_id)
                    alignment_vec = self.alignment(robot_id, neoighbors_id)
                else: 
                    cohesion_vec = [0,0]
                    separation_vec = [0,0]
                    alignment_vec = [0,0]
                move.linear.x = smjer_x + 0.3*(k_alignment*alignment_vec[0] + k_cohesion*cohesion_vec[0] + k_separation*separation_vec[0])
                move.linear.y = smjer_y + 0.3*(k_alignment*alignment_vec[1] + k_cohesion*cohesion_vec[1] + k_separation*separation_vec[1])

        # check if the boid is near the wall
        if self.wall_check(odom_i.pose.pose.position, odom_i.pose.pose.orientation, 0):
            #print('zid')
            pozicija_zida = self.wall_check(msg.pose.pose.position, msg.pose.pose.orientation, 1)
            delta_x = (pozicija_zida[0] - msg.pose.pose.position.x)/(abs(pozicija_zida[0] - msg.pose.pose.position.x)**2)
            delta_y = (pozicija_zida[1] - msg.pose.pose.position.y)/(abs(pozicija_zida[1] - msg.pose.pose.position.y)**2)

            move.linear.y = max(min(-delta_y*2,0.5), -0.5) 
            move.linear.x = max(min(-delta_x*2, 0.5),-0.5)

            #move.linear.x = -delta_x*0.3
            #move.linear.y = -delta_y*0.3
            print('BRZINA X:', move.linear.x)
            print('BRZINA Y:', move.linear.y)


        #move.linear.x = limit_speed(move.linear.x, 10.0)
        #move.linear.y = limit_speed(move.linear.y, 10.0)
        self.pub[robot_id].publish(move)

    def wall_check(self, pose, orient, flag):
        "Returns True if the boid is close to the wall"
        _, _, z = quaternion_to_euler(orient.w, orient.x, orient.y, orient.z)

        agent_column = int((pose.x - self.map_origin.x) / self.map_resolution)
        agent_row = int((self.map_origin.y + len(self.map) *
                         self.map_resolution - pose.y) / self.map_resolution)

        rows = len(self.map)
        columns = len(self.map[0])

        column_neighbourhood = range(max(0, agent_column - self.avoiding_distance_to_the_wall), min(
            columns, agent_column + self.avoiding_distance_to_the_wall + 1))
        row_neighbourhood = range(max(0, agent_row - self.avoiding_distance_to_the_wall),
                                  min(rows, agent_row + self.avoiding_distance_to_the_wall + 1))

        # Checking if there is a wall in the neighbourhood
        # x and y are locations of the wall
        # can return if necessary
        for row in row_neighbourhood:
            for column in column_neighbourhood:
                distance = pow(row - agent_row, 2) + \
                    pow(column - agent_column, 2)
                # We found the wall

                if distance <= pow(self.avoiding_distance_to_the_wall, 2) and self.map[row][column] == 100:
                    x = self.map_origin.x + column * self.map_resolution
                    y = self.map_origin.y + (rows - row) * self.map_resolution
                    alpha = np.arctan(y/x)

                    if alpha < z + (self.fov/2) and alpha > z - (self.fov/2):
                        print(f"Wall is at: ({x}, {y}, {alpha})")
                        print(
                            f"While the robot is at ({pose.x}, {pose.y}, {z})")
                        if (flag):
                            return [x, y, alpha]
                        else:
                            return True
                        # Tu mozes vracati x, y, alpha, to su ti lokacije zida i kut izmedu
                        # robota i zida
        return False

    # def avoid_the_wall(self, odometry, robot_id) -> Twist():
        #"Calculates linear and angular velocities in order to avoid hitting the wall"

        #move = Twist()
        #pose = odometry.pose.pose.position
        # print('POZICIJA:',pose.x)
        #orient = odometry.pose.pose.orientation
        #turning_speed_kof = 0.8
        #_, _, z = quaternion_to_euler(orient.w, orient.x, orient.y, orient.z)

        #pozicija_zida=self.wall_check(pose.x, orient, 1)
        #delta_x=pozicija_zida[0] - pose.x
        #delta_y=pozicija_zida[1] - pose.y

        #move.linear.x = -delta_x*turning_speed_kof
        #move.linear.y = -delta_y*turning_speed_kof
        #print('BRZINA X:',move.linear.x)
        #print('BRZINA Y:',move.linear.y)
        # return move

    def get_neighbours_ids(self, current_robot_id, positions_of_all_boids) -> List[int]:
        "Returns list of ids of boids that are in neighbourhood of a current boid"

        neighbours = []
        current_boid_pose = positions_of_all_boids[current_robot_id].pose.pose.position
        radius_list = [(linalg.norm([boid.pose.pose.position.x - current_boid_pose.x,
                        boid.pose.pose.position.y - current_boid_pose.y]) < self.radius) for boid in positions_of_all_boids]
        # radius_list -> list of True or False values depending if boids are within a defined radius
        angle_list = [np.arctan2(boid.pose.pose.position.y - current_boid_pose.y,
                                 boid.pose.pose.position.x - current_boid_pose.x) for boid in positions_of_all_boids]
        # angle_list -> list of angles between all the boids
        for i, element in enumerate(radius_list):
            if element:  # if the boid is inside radius then it is considered for furthers inspection if he is in FoV
                current_robot_angle = positions_of_all_boids[current_robot_id].pose.pose.orientation
                _, _, z = quaternion_to_euler(
                    current_robot_angle.w, current_robot_angle.x, current_robot_angle.y, current_robot_angle.z)
                if (abs(angle_list[i]-z) < self.fov) and (i != current_robot_id):
                    neighbours.append(i)
        return neighbours

    def cohesion(self, robot_id, neighbours):
        curent_rob_position = self.last_positions[robot_id].pose.pose.position
        x_poz_poc = curent_rob_position.x
        y_poz_poc = curent_rob_position.y
        x_poz = 0
        y_poz = 0
        for i in neighbours:
            x_poz += self.last_positions[i].pose.pose.position.x - x_poz_poc
            y_poz += self.last_positions[i].pose.pose.position.y - y_poz_poc

        x_poz_k = x_poz/(len(neighbours)) #0.5
        y_poz_k = y_poz/(len(neighbours))
        poz = [x_poz_k, y_poz_k]

        return poz

    def separation(self, robot_id, neighbours):
        curent_rob_position = self.last_positions[robot_id].pose.pose.position
        x_poz_poc_s = curent_rob_position.x
        y_poz_poc_s = curent_rob_position.y
        x_poz_s = 0
        y_poz_s = 0
        for i in neighbours:
            if((self.last_positions[i].pose.pose.position.x - x_poz_poc_s) != 0):
                x_poz_s += (self.last_positions[i].pose.pose.position.x - x_poz_poc_s)/abs(
                    (self.last_positions[i].pose.pose.position.x - x_poz_poc_s)*(self.last_positions[i].pose.pose.position.x - x_poz_poc_s))
            if ((self.last_positions[i].pose.pose.position.y - y_poz_poc_s) != 0):
                y_poz_s += (self.last_positions[i].pose.pose.position.y - y_poz_poc_s)/abs(
                    (self.last_positions[i].pose.pose.position.y - y_poz_poc_s)*(self.last_positions[i].pose.pose.position.y - y_poz_poc_s))

        x_poz_s_k = -x_poz_s/(len(neighbours))
        y_poz_s_k = -y_poz_s/(len(neighbours)) #-0.08/(len(neighbours))*y_poz_s
        poz_s = [x_poz_s_k, y_poz_s_k]

        return poz_s

    def alignment(self, robot_id, neighbours):
        x_vel_poc = self.last_positions[robot_id].twist.twist.linear.x
        y_vel_poc = self.last_positions[robot_id].twist.twist.linear.y
        x_vel = 0
        y_vel = 0
        for i in neighbours:
            x_vel += self.last_positions[i].twist.twist.linear.x - x_vel_poc
            y_vel += self.last_positions[i].twist.twist.linear.y - y_vel_poc

        x_vel_k = x_vel/(len(neighbours)) #0.08/(len(neighbours))*x_vel
        y_vel_k = y_vel/(len(neighbours))

        vel = [x_vel_k, y_vel_k]

        return vel


if __name__ == '__main__':
    try:
        rospy.init_node('control_node')
        sim = Simulation()
        rate = rospy.Rate(10000*sim.num_of_robots)  # 10hz
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

