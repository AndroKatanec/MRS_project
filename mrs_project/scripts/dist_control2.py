#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import message_filters
from time import sleep
from numpy import linalg
import numpy as np
from typing import List



def quaternion_to_euler(w, x, y, z):
    ysqr = y * y
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0,+1.0,t2)
    #t2 = +1.0 if t2 > +1.0 else t2
    t2 = np.where(t2<-1.0, -1.0, t2)
    #t2 = -1.0 if t2 < -1.0 else t2
    Y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.arctan2(t3, t4)
    return X, Y, Z

class Simulation():
    def __init__(self) -> None:
        self.avoiding_distance_to_the_wall = 4
        self.radius = 1.5 #radius of DoV of a boid
        self.fov = 3 #+- field of view of a boid in radians
        self.num_of_robots = rospy.get_param("/num_of_robots")
        self.initialization = [False for i in range(self.num_of_robots)]
        #set initial velocity
        odom = Odometry()
        odom.twist.twist.linear.x=1

        self.last_positions = [odom for i in range(self.num_of_robots)]
        self.pub = [rospy.Publisher(f'/robot_{i}/cmd_vel', Twist, queue_size=2) for i in range(self.num_of_robots)]
        sleep(1)
        self.sub = [rospy.Subscriber("/robot_{}/odom".format(i), Odometry, self.callback) for i in range(self.num_of_robots)]


    def callback(self, msg) -> None:
        robot_id = int(list(msg._connection_header['topic'])[7])  #get the id of the boid that this message came from
        self.last_positions[robot_id] = msg #update last known positions
        neoighbors_id = self.get_neighbours_ids(robot_id, self.last_positions)
        odom_i = self.last_positions[robot_id]
        move = Twist()
        if self.wall_check(msg.pose.pose.position): #check if the boid is near the wall
            move = self.avoid_the_wall(odom_i, robot_id)

        elif len(neoighbors_id)>0: #check if there is nearby boids to apply Reynolds rules
            #move.linear.x = min(odom_i.twist.twist.linear.x+0.1, 1)
            cohesion_vec = self.cohesion(robot_id, neoighbors_id)
            separation_vec = self.separation(robot_id, neoighbors_id)
            #move.angular.z = min(self.last_positions[robot_id].twist.twist.angular.z + separation_vec*0.04, 5)
            brzina = self.alignment(robot_id, neoighbors_id)
            move.linear.x = brzina[0] + cohesion_vec[0] + separation_vec[0]
            move.linear.y = brzina[1] + cohesion_vec[1] + separation_vec[1]

            if brzina[0]<0.03:
                move.linear.x = move.linear.x + move.linear.x + move.linear.x + move.linear.x + move.linear.x
                move.linear.y = move.linear.y + move.linear.y + move.linear.y + move.linear.y + move.linear.y


        else: #if there is no nearby boids to be affected by just go straight
            move.linear.x = min(odom_i.twist.twist.linear.x+0.08, 0.3) 
            #move.angular.z = max(self.last_positions[robot_id].twist.twist.angular.z-0.2, 0)
            move.angular.z = 0
        self.pub[robot_id].publish(move)


    def wall_check(self, pose) -> bool:

        "Returns True if the boid is close to the wall"

        flag = False
        if (abs(pose.x) > self.avoiding_distance_to_the_wall) or (abs(pose.y) > self.avoiding_distance_to_the_wall):
            flag = True
        return flag

    def avoid_the_wall(self, odometry, robot_id) -> Twist():

        "Calculates linear and angular velocities in order to avoid hitting the wall"

        move = Twist()
        pose = odometry.pose.pose.position
        orient = odometry.pose.pose.orientation
        turning_speed = 10
        _ ,_ ,z = quaternion_to_euler(orient.w, orient.x, orient.y, orient.z)
        if pose.x > self.avoiding_distance_to_the_wall and abs(z)<3:  # -> || pozicija
            if z>0:
                move.angular.z = min(self.last_positions[robot_id].twist.twist.angular.z+0.1, turning_speed)
            else:
                move.angular.z = max(self.last_positions[robot_id].twist.twist.angular.z-0.1, -turning_speed)

        elif pose.x < -self.avoiding_distance_to_the_wall and abs(z)>0.2: # || <- pozicija
            if z<0:
                move.angular.z = min(self.last_positions[robot_id].twist.twist.angular.z+0.1, turning_speed)
            else:
                move.angular.z = max(self.last_positions[robot_id].twist.twist.angular.z-0.1, -turning_speed)

        elif pose.y > self.avoiding_distance_to_the_wall and (z<-1.8 or z > -1.3):
            if z>1.57:
                move.angular.z = min(self.last_positions[robot_id].twist.twist.angular.z+0.1, turning_speed)
            else:
                move.angular.z = max(self.last_positions[robot_id].twist.twist.angular.z-0.1, -turning_speed)
        elif pose.y < -self.avoiding_distance_to_the_wall and (z<1.3 or z > 1.7):
            if z>-1.57:
                move.angular.z = min(self.last_positions[robot_id].twist.twist.angular.z+0.1, turning_speed)
            else:
                move.angular.z = max(self.last_positions[robot_id].twist.twist.angular.z-0.1, -turning_speed)
        else:
            move.angular.z=0
        
        move.linear.x= max(odometry.twist.twist.linear.x-0.1, 0.2)     
        return move

    def get_neighbours_ids(self, current_robot_id, positions_of_all_boids) -> List[int]:

        "Returns list of ids of boids that are in neighbourhood of a current boid"

        neighbours = []
        current_boid_pose = positions_of_all_boids[current_robot_id].pose.pose.position
        radius_list = [(linalg.norm([boid.pose.pose.position.x - current_boid_pose.x, boid.pose.pose.position.y - current_boid_pose.y]) < self.radius) for boid in positions_of_all_boids]
        #radius_list -> list of True or False values depending if boids are within a defined radius
        angle_list = [np.arctan2(boid.pose.pose.position.y - current_boid_pose.y, boid.pose.pose.position.x - current_boid_pose.x) for boid in positions_of_all_boids]
        #angle_list -> list of angles between all the boids
        for i, element in enumerate(radius_list):
            if element: #if the boid is inside radius then it is considered for furthers inspection if he is in FoV
                current_robot_angle = positions_of_all_boids[current_robot_id].pose.pose.orientation
                _ ,_ ,z = quaternion_to_euler(current_robot_angle.w, current_robot_angle.x, current_robot_angle.y, current_robot_angle.z)
                if (abs(angle_list[i]-z) < self.fov) and (i!=current_robot_id):
                    neighbours.append(i)
        return neighbours

    def cohesion(self, robot_id, neighbours):
        curent_rob_position = self.last_positions[robot_id].pose.pose.position
        x_poz_poc = curent_rob_position.x
        y_poz_poc = curent_rob_position.y
        x_poz=0
        y_poz=0
        for i in neighbours:
            x_poz+=self.last_positions[i].pose.pose.position.x - x_poz_poc
            y_poz+=self.last_positions[i].pose.pose.position.y - y_poz_poc

        x_poz_k=0.1/(len(neighbours))*x_poz
        y_poz_k=0.1/(len(neighbours))*y_poz

        poz=[x_poz_k, y_poz_k]
        #x_center = x_sum/(1+len(neighbours))
        #y_center = y_sum/(1+len(neighbours))
        #angle = np.arctan2(y_center - curent_rob_position.y, x_center - curent_rob_position.x)

        return poz

    def separation(self, robot_id, neighbours):
        curent_rob_position = self.last_positions[robot_id].pose.pose.position
        x_poz_poc_s = curent_rob_position.x
        y_poz_poc_s = curent_rob_position.y
        x_poz_s=0
        y_poz_s=0
        for i in neighbours:
            if((self.last_positions[i].pose.pose.position.x - x_poz_poc_s)!=0):
                x_poz_s+=(self.last_positions[i].pose.pose.position.x - x_poz_poc_s)/abs((self.last_positions[i].pose.pose.position.x - x_poz_poc_s)*(self.last_positions[i].pose.pose.position.x - x_poz_poc_s))
            elif((self.last_positions[i].pose.pose.position.y - y_poz_poc_s)!=0):
                y_poz_s+=(self.last_positions[i].pose.pose.position.y - y_poz_poc_s)/abs((self.last_positions[i].pose.pose.position.y - y_poz_poc_s)*(self.last_positions[i].pose.pose.position.y - y_poz_poc_s))
        
        x_poz_s_k=-0.08/(len(neighbours))*x_poz_s
        y_poz_s_k=-0.08/(len(neighbours))*y_poz_s
        poz_s=[x_poz_s_k, y_poz_s_k]
        #distances = []
        #vectors = []
        #for i in neighbours:
            #vector_i = (self.last_positions[robot_id].pose.pose.position.x - self.last_positions[i].pose.pose.position.x, self.last_positions[robot_id].pose.pose.position.y - self.last_positions[i].pose.pose.position.y)
            #vectors.append(vector_i)
            #distance_i = linalg.norm([self.last_positions[i].pose.pose.position.x - self.last_positions[robot_id].pose.pose.position.x, self.last_positions[i].pose.pose.position.y - self.last_positions[robot_id].pose.pose.position.y])
            #distances.append(distance_i)
        #force_vector = [0,0]
        #for i in range(len(neighbours)):
            #force_vector[0]+=(np.sum(distances)/distances[i])*vectors[i][0]
            #force_vector[1]+=(np.sum(distances)/distances[i])*vectors[i][1]
        #angle = np.arctan2(force_vector[1], force_vector[0])
        return poz_s

    def alignment(self, robot_id, neighbours):
        x_vel_poc=self.last_positions[robot_id].twist.twist.linear.x
        y_vel_poc=self.last_positions[robot_id].twist.twist.linear.y
        x_vel=0
        y_vel=0
        for i in neighbours:
            x_vel+=self.last_positions[i].twist.twist.linear.x - x_vel_poc
            y_vel+=self.last_positions[i].twist.twist.linear.y - y_vel_poc

        x_vel_k=0.08/(len(neighbours))*x_vel
        y_vel_k=0.08/(len(neighbours))*y_vel

        vel=[x_vel_k, y_vel_k]

        return vel
        pass

if __name__ == '__main__':
    try:
        rospy.init_node('control_node')
        sim = Simulation()
        rate = rospy.Rate(10000*sim.num_of_robots) #10hz
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
