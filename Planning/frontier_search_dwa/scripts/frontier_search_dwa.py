#!/usr/bin/python3

import array
from ast import Return
from turtle import distance
import numpy as np
import math as m
import rospy
import tf
import functions as f
import matplotlib.pyplot as plt
import random


from frontier_expl import frontier
from DWA import DWA 
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Float64MultiArray
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger, TriggerResponse
from move_base_msgs.msg import MoveBaseActionResult

from online_planning import StateValidityChecker, Node, move_to_point, compute_path_global, dist_between_points

class OnlinePlanner:

    # OnlinePlanner Constructor
    def __init__(self, gridmap_topic, odom_topic, cmd_vel_topic, dominion, distance_threshold):

        # ATTRIBUTES 
        # List of points which define the plan. None if there is no plan
        self.path = []
        # State Validity Checker object (passes Map)                                                
        self.svc = StateValidityChecker(distance_threshold)
        self.dist_threshold = distance_threshold
        # Current robot SE2 pose [x, y, yaw], None if unknown            
        self.current_pose = None
        # Goal where the robot has to move, None if it is not set                                                                   
        self.goal = None
        # Last time a map was received (to avoid map update too often)                                                
        self.last_map_time = rospy.Time.now()
        # Dominion [min_x_y, max_x_y] in which the path planner will sample configurations                           
        self.dominion = dominion 
        # Flags to prevent errors
        self.map_loaded = False   
        self.planning = False  

        # FRONTIER SEARCH WEIGHTS      
        self.vpDist_w = 0.3     
        self.fSize_w = 0.9            

        # CONTROLLER PARAMETERS
        # Proportional linear velocity controller gain
        self.Kv = 0.5
        # Proportional angular velocity controller gain                   
        self.Kw = 0.5
        # Maximum linear velocity control action                   
        self.v_max = 0.5
        # Maximum angular velocity control action               
        self.w_max = 1.0  
        # linear and angular accel limits
        self.accel_limits  = [0.5, 0.5]    
        # Current Velocity
        self.best_u = [0.0, 0.0]       

        # PUBLISHERS
        # Publisher for sending velocity commands to the robot
        self.cmd_pub =  rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)# TODO: publisher to cmd_vel_topic
        # Publisher for visualizing the path to with rviz
        self.marker_pub = rospy.Publisher('/path_marker', Marker, queue_size=1)
        # Publisher for visualizing frontiers and viewpoints
        self.frontier_pub = rospy.Publisher('/frontier_markers', Marker, queue_size=1)
        self.dwa_pub = rospy.Publisher('/dwa_arcs', Marker, queue_size=1)
        
        # SUBSCRIBERS
        self.gridmap_sub = rospy.Subscriber("/projected_map", OccupancyGrid, self.get_gridmap) #subscriber to gridmap_topic from Octomap Server  
        self.odom_sub = rospy.Subscriber("/turtlebot/odom_ground_truth", Odometry, self.get_odom) #subscriber to odom_topic  
        self.gotolocation = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.get_selected_goal) #subscriber to /move_base_simple/goal published by rviz
        self.testing = rospy.Service('/explore', Trigger, self.debug_explore)
        self.exploring = rospy.Timer(rospy.Duration(1.0), self.explore, oneshot=True)

        
        # TIMERS
        # Timer for velocity controller
        rospy.Timer(rospy.Duration(0.1), self.controller)
    
    # Odometry callback: Gets current robot pose and stores it into self.current_pose
    def get_odom(self, odom):
        _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
                                                              odom.pose.pose.orientation.y,
                                                              odom.pose.pose.orientation.z,
                                                              odom.pose.pose.orientation.w])
        # Store current position (x, y, yaw) as a np.array in self.current_pose var.
        self.current_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw])

    # Map callback: Gets the latest occupancy map published by Octomap server and update 
    # the state validity checker
    def get_gridmap(self, gridmap):
        # To avoid map update too often (change value '1' if necessary)
        if (gridmap.header.stamp - self.last_map_time).to_sec() > 1:            
            self.last_map_time = gridmap.header.stamp
            # Update State Validity Checker
            env = np.array(gridmap.data).reshape(gridmap.info.height, gridmap.info.width).T
            # self.svc.map_viz_debug(env, 'env')
            origin = [gridmap.info.origin.position.x, gridmap.info.origin.position.y]
            self.svc.set(env, gridmap.info.resolution, origin)
            cell_pos = self.svc.position_to_map(self.current_pose[:2])
            # self.svc.map_viz_debug(self.svc.map, 'svc', cell_pos)
            self.map_loaded = True
            if len(self.path) > 0:
                path_good = self.svc.check_path(self.path)
                if not path_good:
                    self.plan()

            # If the robot is following a path, check if it is still valid
            if len(self.path) > 0:
                # create total_path adding the current position to the rest of waypoints in the path
                total_path = [self.current_pose[0:2]] + self.path
                # TODO: check total_path validity. If total_path is not valid replan
                
    

    # Goal callback: Get new goal from /move_base_simple/goal topic published by rviz 
    # and computes a plan to it using self.plan() method
    def get_selected_goal(self, goal):
        self.need_new_path = False
        _, _, yaw = tf.transformations.euler_from_quaternion([goal.pose.orientation.x, 
                                                                goal.pose.orientation.y,
                                                                goal.pose.orientation.z,
                                                                goal.pose.orientation.w])
        print('going to a selected point')
        # Store current position (x, y, yaw) as a np.array in self.current_pose var.
        self.goal = np.array([goal.pose.position.x, goal.pose.position.y, goal.pose.position.z, yaw])
        print("the goal has been set to: ", self.goal)
        self.plan()

    def closest_valid_point(self, cell_pos, desired_location, check_dist):
        if self.svc.is_valid_pixel(desired_location):
            return desired_location
        else:
            print("in loop")
            best = np.inf
            valid_vp = np.array([None])
            for i in range(-check_dist, check_dist+1):
                for j in range(-check_dist, check_dist+1):
                    cell_x = desired_location[0] + i
                    cell_y = desired_location[1] + j
                    cell_loc = np.array([cell_x, cell_y])
                    if self.svc.is_valid_pixel(cell_loc, added_pixel_assurance=3, free_space_goal=True):
                        score = np.linalg.norm(cell_loc-desired_location) + np.linalg.norm(cell_loc-cell_pos)
                        if score < best:
                            best = score
                            valid_vp = cell_loc
            return valid_vp

    def get_viewpoint(self):
        map = self.svc.map
        if np.count_nonzero(map == -1) < (map.shape[0]*map.shape[1])*0.03:
            print('Map has been 99 percent explored!')
            raise AssertionError("Map sufficiently explored! :)")
        
        cell_pos = self.svc.position_to_map(self.current_pose[:2])
        distance_pixel = int(self.svc.distance / self.svc.resolution)+1
        check_dist = int(distance_pixel*1.25)
        min_travel_dist = distance_pixel
        valid_vp = np.array([None])
        while valid_vp[0] == None:
            explore = frontier(map, cell_pos, self.vpDist_w, self.fSize_w)
            vp, best_group, frontiers = explore.choose_vp(travel_dist=min_travel_dist)
            valid_vp = self.closest_valid_point(cell_pos, vp, check_dist)  
            min_travel_dist += int(distance_pixel*0.25) 
            print('No valid vp at desired location... expanding possible vps')
        # self.svc.map_viz_debug(map,'get_viewpoint', cell_pos, valid_vp)
        self.goal = np.array(self.svc.map_to_position(valid_vp))

        # for plotting in rviz
        fromx, fromy = valid_vp[0]-distance_pixel, valid_vp[1]-distance_pixel
        tox, toy = valid_vp[0] + distance_pixel, valid_vp[1]+distance_pixel
        square = []
        for i in range(fromx,tox+1):
            square.append([i,fromy])
            square.append([i,toy])
        for i in range(fromy,toy+1):
            square.append([fromx,i])
            square.append([tox,i])
        square = np.array(square)
        
        self.publish_frontiers(frontiers, best_group, square)
        print(f"Current gridmap location: {cell_pos}")
        print(f"Next viewpoint selected at {self.goal}")

    def debug_explore(self, event):
        rospy.loginfo("Received explore request")
        while not self.map_loaded and not rospy.is_shutdown():
            rospy.loginfo("Waiting for map to become ready...")
            rospy.sleep(0.5)   
        self.get_viewpoint()
        self.plan()
        return TriggerResponse(success=True, message="Exploring...")
    
    def explore(self, event):
        while not self.map_loaded and not rospy.is_shutdown():
            rospy.loginfo("Waiting for map to become ready...")
            rospy.sleep(0.5) 
        self.get_viewpoint()
        self.plan()

    # Solve plan from current position to self.goal. 
    def plan(self, attempts=5):

        attempts_left = attempts
        self.planning = True
        while attempts_left > 0:
            # Invalidate previous plan if available

            print("Compute new path")
            if self.goal.size > 2:
                goal = self.goal[0:2]
            else:
                goal = self.goal
            try:
                if self.svc.is_valid_pixel(self.svc.position_to_map(goal)) == False:
                    print("Goal is in collision")
                    self.get_viewpoint()
                    attempts -= 1
                    continue
                self.path = compute_path_global(start_p=self.current_pose[0:2], goal_p=goal, state_validity_checker=self.svc, dominion=self.dominion)

                if not self.path:
                    raise AssertionError("Empty Path")
                break
            except AssertionError as e:
                self.__send_commnd__(0,0)
                self.vpDist_w = random.uniform(0,1)
                self.fSize_w = random.uniform(0,1)
                self.path = []
                rospy.logwarn(f"Error occurred: {e}")
                rospy.loginfo(f"New randomized frontier weights to avoid tough to reach spot: dist({self.vpDist_w}), size({self.fSize_w})")
                self.get_viewpoint()
                
                attempts_left -= 1

            finally:
                self.planning = False
                
        else:
            rospy.logwarn("Max attempts reached. No valid viewpoint :(")
            return

        if len(self.path) == 0:
            print("Path not found!")
            self.__send_commnd__(0,0)
            self.get_viewpoint()
        else:
            print("Path found")
            # Publish plan marker to visualize in rviz
            self.publish_path()
            # remove initial waypoint in the path (current pose is already reached)
            #del self.path[0]                 
        

    # This method is called every 0.1s. It computes the velocity comands in order to reach the 
    # next waypoint in the path. It also sends zero velocity commands if there is no active path.
    def controller(self, event):
        # v = 0
        # w = 0

        if self.planning:
            rospy.loginfo("Planning...")
            self.__send_commnd__(0, 0)
            return
        
        if len(self.path) > 0: 
            if dist_between_points(self.current_pose[0:2], self.path[0]) <= 0.2:# If current waypoint is reached with some tolerance move to the next waypoint. 
                del self.path[0]
                # If it was the last waypoint in the path show a message indicating it
                if len(self.path) == 0:
                    print("Goal reached!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    self.get_viewpoint()
                    self.plan()
            
            else: #TODO right now has no map to begin with, need to fix!!!!!!
                print(f'planner waypoint {self.path[0]}')
                print(f'Current position from planner {self.current_pose[:3]}')
                dwa = DWA(self.current_pose[:3], self.path[0], self.svc, self.goal, 
                          [self.v_max, self.w_max], self.accel_limits, self.svc.distance, 
                          np.array([0.4, 0.8, 1.0, 1.3]), '/odom')
                # Updating current pose and velocities
                self.best_u, best_course, arcs = dwa.create_DWA_arcs(self.best_u)
                # RVIZ plots for trajectories
                self.publish_dwa_arcs(best_course, arcs)
                
                # v,w = move_to_point(self.current_pose[0:2], self.current_pose[3],self.path[0], self.Kv, self.Kw)
        else: 
            rospy.loginfo("No path stopping movement")
            self.__send_commnd__(0,0)
            return
          
        
        # Publish velocity commands
        #print("v: ", v, "w: ", w)
        self.__send_commnd__(self.best_u[0], self.best_u[1])
    

    # PUBLISHER HELPERS
    # Transform linear and angular velocity (v, w) into a Twist message and publish it
    def __send_commnd__(self, v, w):
        cmd = Twist()
        cmd.linear.x = np.clip(v, -self.v_max, self.v_max)
        cmd.linear.y = 0
        cmd.linear.z = 0
        cmd.angular.x = 0
        cmd.angular.y = 0
        cmd.angular.z = np.clip(w, -self.w_max, self.w_max)
        self.cmd_pub.publish(cmd)


    # Publish a path as a series of line markers
    def publish_path(self):
        self.path.insert(0, self.current_pose[0:2])
        if len(self.path) > 1:
            m = Marker()
            m.header.frame_id = 'world_ned'
            m.header.stamp = rospy.Time.now()
            m.id = 0
            m.type = Marker.LINE_STRIP
            m.ns = 'path'
            m.action = Marker.DELETE
            m.lifetime = rospy.Duration(0)
            self.marker_pub.publish(m)

            m.action = Marker.ADD
            m.scale.x = 0.1
            m.scale.y = 0.0
            m.scale.z = 0.0
            
            m.pose.orientation.x = 0
            m.pose.orientation.y = 0
            m.pose.orientation.z = 0
            m.pose.orientation.w = 1
            
            color_red = ColorRGBA()
            color_red.r = 1
            color_red.g = 0
            color_red.b = 0
            color_red.a = 1
            color_blue = ColorRGBA()
            color_blue.r = 0
            color_blue.g = 0
            color_blue.b = 1
            color_blue.a = 1

            p = Point()
            p.x = self.current_pose[0]
            p.y = self.current_pose[1]
            p.z = 0.0
            m.points.append(p)
            m.colors.append(color_blue)
            
            for n in self.path:
                p = Point()
                p.x = n[0]
                p.y = n[1]
                p.z = 0.0
                m.points.append(p)
                m.colors.append(color_red)
            
            self.marker_pub.publish(m)
            print ("path published")
    
    def publish_frontiers(self, frontiers, selected_group, vp):
        delete_marker = Marker()
        delete_marker.header.frame_id = 'world_ned'
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.ns = 'frontier'
        delete_marker.id = 1
        delete_marker.action = Marker.DELETE
        self.frontier_pub.publish(delete_marker)

        m = Marker()
        m.header.frame_id = 'world_ned'
        m.header.stamp = rospy.Time.now()
        m.id = 1
        m.type = Marker.CUBE_LIST
        m.ns = 'frontier'
        m.action = Marker.ADD
        m.scale.x = self.svc.resolution
        m.scale.y = self.svc.resolution
        m.scale.z = 0.03
        m.pose.orientation.w = 1.0
        color_green = ColorRGBA()
        color_green.r = 0
        color_green.g = 1
        color_green.b = 0
        color_green.a = 1
        color_blue = ColorRGBA()
        color_blue.r = 0
        color_blue.g = 0
        color_blue.b = 1
        color_blue.a = 1
        color_purple = ColorRGBA()
        color_purple.r = 1
        color_purple.g = 0
        color_purple.b = 1
        color_purple.a = 1

        used = []
        for cell in vp:
            pos = self.svc.map_to_position(cell)
            if pos is not None:
                    used.append(tuple(pos))
                    p = Point()
                    p.x = pos[0]
                    p.y = pos[1]
                    p.z = -0.03
                    m.points.append(p)
                    m.colors.append(color_green)
        for cell in selected_group:
            pos = self.svc.map_to_position(cell)
            if pos is not None and tuple(pos) not in used:
                    used.append(tuple(pos))
                    p = Point()
                    p.x = pos[0]
                    p.y = pos[1]
                    p.z = -0.03
                    m.points.append(p)
                    m.colors.append(color_purple)
        for cell in frontiers:
            pos = self.svc.map_to_position(cell)
            if pos is not None and tuple(pos) not in used:
                p = Point()
                p.x = pos[0]
                p.y = pos[1]
                p.z = -0.03
                m.points.append(p)
                m.colors.append(color_blue)
        self.frontier_pub.publish(m)

    def publish_dwa_arcs(self, best, possible_arcs):
        # DEBUG 
        rospy.loginfo(f"Publishing {len(possible_arcs)} arcs")
        rospy.loginfo(f"Best arc points: {best[:3] if best is not None else 'None'}")
        delete_marker = Marker()
        delete_marker.header.frame_id = 'world_ned'
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.ns = 'dwa'
        delete_marker.id = 3
        delete_marker.action = Marker.DELETE
        self.dwa_pub.publish(delete_marker)

        m = Marker()
        m.header.frame_id = 'world_ned'
        m.header.stamp = rospy.Time.now()
        m.id = 3
        m.type = Marker.CUBE_LIST
        m.ns = 'dwa'
        m.action = Marker.ADD
        m.scale.x = self.svc.resolution
        m.scale.y = self.svc.resolution
        m.scale.z = 0.03
        m.pose.orientation.w = 1.0
        color_orange=ColorRGBA(1,0.647,0,1)
        color_yellow=ColorRGBA(1,1,0,1)
        used = []
        if best is not None:
            for point in best:
                pos = point[:2]
                used.append(tuple(pos))
                p = Point()
                p.x = pos[0]
                p.y = pos[1]
                p.z = -0.35
                m.points.append(p)
                m.colors.append(color_orange)
        for arc in possible_arcs:
            for point in arc:
                pos = point[:2]
                if pos is not None and tuple(pos) not in used:
                    p = Point()
                    p.x = pos[0]
                    p.y = pos[1]
                    p.z = -0.3
                    m.points.append(p)
                    m.colors.append(color_yellow)
        self.dwa_pub.publish(m)

# MAIN FUNCTION
if __name__ == '__main__':
    rospy.init_node('turtlebot_online_path_planning_node')   

    node = OnlinePlanner('/projected_map', '/odom', '/turtlebot/kobuki/commands/velocity', np.array([-10.0, 10.0, -10.0, 10.0]), 0.2)
    
    # Run forever
    rospy.spin()