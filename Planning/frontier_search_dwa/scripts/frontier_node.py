#!/usr/bin/python3
from turtle import distance
import numpy as np
import rospy
import tf
import matplotlib.pyplot as plt
import random

from frontier_expl import frontier
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Float64MultiArray
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger, TriggerResponse
from move_base_msgs.msg import MoveBaseActionResult
from frontier_search_dwa.msg import dwa

from online_planning import StateValidityChecker, Node, move_to_point, compute_path_global, dist_between_points

class OnlinePlanner:

    # OnlinePlanner Constructor
    def __init__(self, gridmap_topic, odom_topic, dominion, distance_threshold):

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
        # area to search for frontiers
        self.search_center = None #[0,0] #pixel location
        self.search_dist = None #20 #in pixels

        # Flags to prevent errors
        self.map_loaded = False   
        self.planning = False  
        self.replan = False

        # FRONTIER SEARCH WEIGHTS      
        self.vpDist_w = 0.8     
        self.fSize_w = 0.3           

        # Publisher for visualizing the path to with rviz
        self.marker_pub = rospy.Publisher('/path_marker', Marker, queue_size=1)
        # Publisher for visualizing frontiers and viewpoints
        self.frontier_pub = rospy.Publisher('/frontier_markers', Marker, queue_size=1)
        # Publisher for sending path to DWA controller
        self.path_pub = rospy.Publisher('/path_for_dwa', dwa, queue_size = 1)
        
        # SUBSCRIBERS
        self.gridmap_sub = rospy.Subscriber(gridmap_topic, OccupancyGrid, self.get_gridmap) #subscriber to gridmap_topic from Octomap Server  
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.get_odom, queue_size=5) #subscriber to odom_topic  
        self.gotolocation = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.get_selected_goal) #subscriber to /move_base_simple/goal published by rviz
        self.testing = rospy.Service('/explore', Trigger, self.debug_explore)
        self.exploring = rospy.Timer(rospy.Duration(10.0), self.explore, oneshot=False)
        self.replan_sub = rospy.Subscriber('/replan', dwa, self.replan_cb)
    
    # Receives boolean to check if replan is needed
    def replan_cb(self, msg):
        if msg.replan_bool:
            self.get_viewpoint()
            self.plan()

    # Odometry callback: Gets current robot pose and stores it into self.current_pose
    def get_odom(self, odom):
            _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
                                                                odom.pose.pose.orientation.y,
                                                                odom.pose.pose.orientation.z,
                                                                odom.pose.pose.orientation.w])
            # Store current position (x, y, yaw) as a np.array in self.current_pose var.
            self.current_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw])
            self.last_pose = self.current_pose
            self.last_odom_time = rospy.Time.now().to_sec()

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
            # self.svc.map_viz_debug(self.svc.map, 'svc', cell_pos)
            self.map_loaded = True
            if len(self.path) > 0:
                path_good = self.svc.check_path(self.path)
                if not path_good:
                    self.get_viewpoint()
                    self.plan()

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
        self.goal = np.array([goal.pose.position.x, goal.pose.position.y])
        print("[planner] the goal has been set to: ", self.goal)
        self.plan()

    def closest_valid_point(self, cell_pos, desired_location, check_dist):
        if self.svc.is_valid_pixel(desired_location):
            return desired_location
        else:
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
        if np.count_nonzero(map == -1) < (map.shape[0]*map.shape[1])*0.02:
            print(' has been 98 percent explored!')
            raise AssertionError("Map sufficiently explored! :)")
        
        cell_pos = self.svc.position_to_map(self.current_pose[:2])
        distance_pixel = int(self.svc.distance / self.svc.resolution)+1
        check_dist = int(distance_pixel*1.25)
        min_travel_dist = distance_pixel
        valid_vp = np.array([None])
        while valid_vp[0] == None:
            explore = frontier(map, cell_pos, self.vpDist_w, self.fSize_w, search_center=self.search_center, search_distance=self.search_dist)
            vp, best_group, frontiers = explore.choose_vp(travel_dist=min_travel_dist)
            if len(best_group) < 1:
                rospy.logerr('Nothing left to explore, frontiers are all too small')
                self.goal = None
                return
            valid_vp = self.closest_valid_point(cell_pos, vp, check_dist)  
            min_travel_dist += int(distance_pixel*0.25) 
        # self.svc.map_viz_debug(map,'get_viewpoint', cell_pos, valid_vp)
        self.goal = np.array(self.svc.map_to_position(valid_vp))

        # for plotting in rviz
        rob_fromx, rob_fromy = valid_vp[0]-distance_pixel, valid_vp[1]-distance_pixel
        rob_tox, rob_toy = valid_vp[0] + distance_pixel, valid_vp[1]+distance_pixel
        rob_square = []
        for i in range(rob_fromx,rob_tox+1):
            rob_square.append([i,rob_fromy])
            rob_square.append([i,rob_toy])
        for i in range(rob_fromy,rob_toy+1):
            rob_square.append([rob_fromx,i])
            rob_square.append([rob_tox,i])
        rob_square = np.array(rob_square)

        fr_fromx, fr_fromy = valid_vp[0]-distance_pixel, valid_vp[1]-distance_pixel
        fr_tox, fr_toy = valid_vp[0] + distance_pixel, valid_vp[1]+distance_pixel
        fr_square = []
        for i in range(fr_fromx,fr_tox+1):
            fr_square.append([i,fr_fromy])
            fr_square.append([i,fr_toy])
        for i in range(fr_fromy,fr_toy+1):
            fr_square.append([fr_fromx,i])
            fr_square.append([fr_tox,i])
        fr_square = np.array(fr_square)
        
        self.publish_frontiers(frontiers, best_group, rob_square, fr_square)
        print(f"Current gridmap location: {cell_pos}")
        print(f"Next viewpoint selected at {self.goal}")

    def debug_explore(self, event):
        rospy.loginfo("Received explore request")
        while not self.map_loaded and not rospy.is_shutdown():
            rospy.loginfo_throttle(1.0, "Waiting for map to become ready...")
        self.get_viewpoint()
        self.plan()
        return TriggerResponse(success=True, message="Exploring...")
    
    def explore(self, event):
        while not self.map_loaded and not rospy.is_shutdown():
            rospy.loginfo_throttle(1.0,"[planner] Waiting for map to become ready...")
        self.get_viewpoint()
        self.plan()

    # Solve plan from current position to self.goal. 
    def plan(self, attempts=50):

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
                self.publish_path_dwa(np.array([self.goal]))
                self.path = compute_path_global(start_p=self.current_pose[0:2], goal_p=goal, state_validity_checker=self.svc, dominion=self.dominion, max_iterations=10000)

                if not self.path:
                    raise AssertionError("Empty Path")
                break
            except AssertionError as e:
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
            self.get_viewpoint()
        
        else:
            print("Path found")
            # Publish plan marker to visualize in rviz
            self.publish_path_rviz()
        
            self.publish_path_dwa(self.path)
            # remove initial waypoint in the path (current pose is already reached)
            #del self.path[0]                 
            #    
        
        if self.replan:
            self.get_viewpoint()
            self.replan = False 

    # PUBLISHER HELPERS

    # Publish a path for dwa
    def publish_path_dwa(self, path):
        path_msg = dwa()
        path_msg.planning = self.planning
        path_msg.replan_bool = False
        
        if len(path) > 0 and self.path is not None:
            for point in path:
                pose = PoseStamped()
                pose.header.frame_id = "world_ned"
                pose.header.stamp = rospy.Time.now()
                pose.pose.position.x = point[0]
                pose.pose.position.y = point[1]
                path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    # Publish a path as a series of line markers
    def publish_path_rviz(self):
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
    
    def publish_frontiers(self, frontiers, selected_group, vp, frontier_area):
        delete_marker = Marker()
        delete_marker.header.frame_id = 'world_ned'
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.ns = 'frontier'
        delete_marker.id = 3542
        delete_marker.action = Marker.DELETE
        self.frontier_pub.publish(delete_marker)

        search_area = Marker()
        search_area.header.frame_id = 'world_ned'
        search_area.header.stamp = rospy.Time.now()
        search_area.id = 9000
        search_area.type = Marker.LINE_STRIP
        search_area.ns = 'search_square'
        search_area.action = Marker.ADD
        search_area.scale.x = 0.12
        search_area.scale.z = 1.0
        search_area.pose.orientation.w = 1.0
        search_area.color = ColorRGBA(1,0,0,1)
        for cell in frontier_area:
            pos = self.svc.map_to_position(cell)
            if pos is not None:
                    print('here')
                    p = Point()
                    p.x = pos[0]
                    p.y = pos[1]
                    p.z = -0.06
                    search_area.points.append(p)
            else:
                print('WHYYYYYYYYYYYYYYYYYYYYYYYYY')
        search_area.points.append(search_area.points[0])
        self.frontier_pub.publish(search_area)
                    

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

# MAIN FUNCTION
if __name__ == '__main__':
    rospy.init_node('frontier_dwa_node')   

    node = OnlinePlanner('/projected_map', "/turtlebot/odom_ground_truth", np.array([-10.0, 10.0, -10.0, 10.0]), 0.2)
    
    # Run forever
    rospy.spin()