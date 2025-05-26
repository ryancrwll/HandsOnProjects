#!/usr/bin/python3

import rospy
import numpy as np
import traceback
import threading
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Twist, PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from frontier_search_dwa.msg import dwa
import tf
from online_planning import StateValidityChecker, dist_between_points

def wrap_angle(angle): 
    return (angle + np.pi) % (2 * np.pi) - np.pi

class DWA:
    def __init__(self, radius, odom_topic, cmd_vel_topic, path_topic, replan_topic, lidar_topic):
        # From frontier_node.py
        self.svc = StateValidityChecker(radius)
        self.current_pose = None # (x,y,theta) #TODO put back if makes sense
        self.last_odom_time = rospy.Time.now().to_sec() # last logged time from receiving pose (seconds)
        self.good_pose = False # bool to tell if pose is recent enough to use
        self.path = []
        self.waypoint = None
        self.map_resolution = 0.05
        self.map_loaded = False
        self.obstacles = [] # first two args are center coords and third is radius
        self.goal = [] # need goal to know stopping distance (x,y,theta)
        self.planning = True
        
        #### DWA CONTROLLER PARAMETERS ###
        # Maximum linear velocity control action, # Maximum angular velocity control action                 
        self.v_lim = np.array([0.15,1.4])
        self.backwards_vel = -self.v_lim[0]/8
        # linear and angular accel limits
        self.a_lim  = np.array([0.5, 1.0])    
        # Current Velocity np.array([0.7, 0.1, 0.4, 2.3])
        self.current_velocity = [0.0, 0.0]  
        # DWA weights for tuning dyanmic window (heading, clearance, velocity, distance to goal) 
        self.weights = np.array([0.5, 0.5, 0.6, 1.3]) # overriden in control loop
        self.score_with_angular = True
        self.old_weights = self.weights
        # First velocity always opposite #TODO figure out!
        self.control_iteration = 0
        # Last time control function was called to know how ofen its gets called 
        self.last_control_time = rospy.Time.now().to_sec() # last logged time from setting velocities (seconds)
        self.sim_time = 6.0 # how far ahead to project velocities (seconds)
        self.max_dist = self.v_lim[0]*self.sim_time
        self.dt = 0.2 # time step for simulating trajectories (seconds)
        self.radius = radius # radius of robot (meters)
        self.num_vel = 5 # sqrt of number of simulated trajectories to create
        self.heading_diff = 0

        # PUBLISHERS
        # Publisher for sending velocity commands to the robot
        self.cmd_pub =  rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)
        # Publisher for possible dwa_arcs
        self.dwa_pub = rospy.Publisher('/dwa_arcs', Marker, queue_size=25)
        # Publisher to tell planner to replan
        self.replan_pub = rospy.Publisher(replan_topic, dwa, queue_size=1)
        # Debug
        self.ob_pub = rospy.Publisher('/dwa_obstacles', MarkerArray, queue_size=1)

        
        # SUBSCRIBERS
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.get_odom, queue_size=5) #subscriber to odom_topic  
        self.path_sub = rospy.Subscriber(path_topic, dwa, self.get_path)
        self.gotolocation = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.get_selected_goal) #subscriber to /move_base_simple/goal published by rviz
        self.lidar = rospy.Subscriber(lidar_topic, LaserScan, self.create_obstacles)
        
        # TIMERS
        # Timer for velocity controller
        rospy.Timer(rospy.Duration(0.2), self.controller)

    def heading_diff_calc(self, position, desired):
        vector = desired - position[:2]
        angle_needed = np.arctan2(vector[1], vector[0])
        return wrap_angle(angle_needed - position[2]), np.linalg.norm(vector)
        
    
    # Odometry callback: Gets current robot pose and stores it into self.current_pose
    def get_odom(self, odom):
        _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
                                                            odom.pose.pose.orientation.y,
                                                            odom.pose.pose.orientation.z,
                                                            odom.pose.pose.orientation.w])
        # Store current position (x, y, yaw) as a np.array in self.current_pose var.
        yaw = wrap_angle(yaw)
        self.current_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw])
        self.good_pose = rospy.Time.now().to_sec() - self.last_odom_time < 0.2
        self.last_odom_time = rospy.Time.now().to_sec()
                
    def get_path(self, msg):
        # Pass numpy array of path from frontier_dwa.py
        self.path = []
        self.planning = msg.planning
        self.path = [[p.pose.position.x, p.pose.position.y] for p in msg.poses]
        if len(msg.poses) > 0:
            rospy.loginfo(f'Length of received path: {len(self.path)}')
            self.goal = [msg.poses[-1].pose.position.x, msg.poses[-1].pose.position.y]
    
    def get_selected_goal(self, goal):
        #for debugging dwa
        print('goint to selected goal')
        self.need_new_path = False
        _, _, yaw = tf.transformations.euler_from_quaternion([goal.pose.orientation.x, 
                                                                goal.pose.orientation.y,
                                                                goal.pose.orientation.z,
                                                                goal.pose.orientation.w])
        # Store current position (x, y, yaw) as a np.array in self.current_pose var.
        self.path = np.array([self.goal[:2]])
    
    def create_obstacles(self, scan):
        if not hasattr(self.current_pose, '__len__'):
            return
        
        clearance = 0 # use covariance max axis here
        self.obstacles = []
        robx, roby, heading = self.current_pose

        try:
            angle = scan.angle_min

            for r in scan.ranges:
                if r < self.max_dist:
                    x = (r * np.cos(angle+heading))+robx
                    y = (r * np.sin(angle+heading))+roby
                    self.obstacles.append((x, y, clearance))
                angle += scan.angle_increment
            self.map_loaded = True
            self.publish_obstacles(self.obstacles)
        except Exception as e:
            rospy.logerr(f"Obstacle creation failed: {str(e)}\n{traceback.format_exc()}")
            self.obstacles = []

    def motion_model(self, pose, u, dt=None):#TODO maybe switching x and y poses helps
        # standard motion model
        if dt == None:
            dt = self.dt
        x,y,theta = pose.astype(float)
        v, w = u
        
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
    
        theta += w * dt
        return np.array([x,y,theta], dtype=float)
    
    def generate_DWA(self, vel):
        staticConstraints = [self.backwards_vel, self.v_lim[0], -self.v_lim[1], self.v_lim[1]]
        if len(self.path) == 1:
            stopping_dist = np.linalg.norm(self.goal[:2] - self.current_pose[:2])
        else:
            stopping_dist = np.inf
        stop_assurance = self.radius
        if stopping_dist > stop_assurance:
            stopping_dist -= stop_assurance
        # from physics equation vf^2 - vi^2 = 2 * a *d
        max_vel = np.sqrt(2 * self.a_lim[0] * stopping_dist)
        max_vel = min(max_vel, self.v_lim[0])
        dynamicConstraints = [vel[0] - self.a_lim[0]*self.dt,
                                vel[0] + self.a_lim[0]*self.dt,
                                vel[1] - self.a_lim[1]*self.dt,
                                vel[1] + self.a_lim[1]*self.dt]
        # bounds for possible velocities
        dwa = np.array([max(staticConstraints[0], dynamicConstraints[0]),
                        min(max_vel, dynamicConstraints[1]),
                        max(staticConstraints[2], dynamicConstraints[2]),
                        min(staticConstraints[3], dynamicConstraints[3])])
        return dwa
    
    def predict_course(self, u):
        pose = self.current_pose.copy()
        possible_path = [pose.copy()]
        sim_t = 0
        while sim_t <= self.sim_time:
            pose = self.motion_model(pose, u)
            possible_path.append(pose.copy())
            sim_t += self.dt
        path = np.array(possible_path)
        return path

    def calc_scoring_vals(self, path):
        end_pose = path[-1, :]
        heading_diff, dist2goal = self.heading_diff_calc(end_pose,self.waypoint)

        d_score = -dist2goal
        
        closest = np.inf

        #freeze values so that its not outdated during scoring
        obstacles = self.obstacles

        for i in range(len(obstacles)):
            for j in range(len(path)):
                dist_obsticle = np.linalg.norm([obstacles[i][0] - path[j][0], obstacles[i][1] - path[j][1]]) - obstacles[i][2] - self.radius
                if dist_obsticle < closest:
                    closest = dist_obsticle

        return heading_diff, closest, d_score
    
    def create_DWA_arcs(self, current_vel):
        "creates all possible trajectories within a ceratin velocity limits and at a number of intervals of num_vel and selects the one with best score"

        if not self.good_pose:
            rospy.logwarn('Pose update out of date, vel -> 0')
            return np.array([0,0]), None, []
        
        dw = self.generate_DWA(current_vel)
        dyn_windowL = np.linspace(dw[0], dw[1], self.num_vel)
        dyn_windowA = np.linspace(dw[2], dw[3], self.num_vel)
        arcs = []
        best_score = -np.inf
        best_u = np.array([0,0])
        best_course = None

        for i in range(len(dyn_windowL)):
            for j in range(len(dyn_windowA)):
                arc = self.predict_course(np.array([dyn_windowL[i], dyn_windowA[j]]))
                arcs.append(arc) #for plotting
                h_score, c_score, d_score = self.calc_scoring_vals(arc)
                
                # norm values to be range (0,1)
                h_score = 1 - (abs(h_score)/np.pi)
                # use sigmoid to minimize importance of clearances that are very far away
                if c_score <= 0:
                    c_score = -np.inf # values between one and zero
                else:
                    # c_score = (2/(1+np.exp(-50*c_score)))-1
                    c_score = 1
                # velocity score uses initial velocity and not final bc assuming constant velocity over the window bc it is our control input
                # normalize it so that 1 is highest possible value
                v_score = (dyn_windowL[i]/self.v_lim[0])
                if self.score_with_angular:
                    v_score += 0.1*(dyn_windowA[i]/self.v_lim[1])
                score = np.array([h_score,c_score,v_score,d_score]) @ self.weights
                if score > best_score:
                    best_score = score
                    best_u = np.array([dyn_windowL[i], dyn_windowA[j]])
                    self.heading_diff = h_score
                    best_course = arc
        # return to original values of weights so dynamic weights can be used
        self.weights = self.old_weights

        return best_u, best_course, arcs
    
    # This method is called every 0.1s. It computes the velocity comands in order to reach the 
    # next waypoint in the path. It also sends zero velocity commands if there is no active path.
    def controller(self, event):
        # if not self.svc.check_path(self.path):
        #     self.path = []
        #     msg = dwa()
        #     msg.replan_bool = True
        #     rospy.loginfo("Replan requested")
        #     self.replan_pub.publish(msg)
        #     return
        
        rospy.loginfo_throttle(3.0,f'len of path in controller(): {len(self.path)}')

        while not self.map_loaded and not rospy.is_shutdown():
            rospy.loginfo_throttle(1.0, "[dwa] Waiting for map to become ready...")

        if len(self.path) == 0:
            if len(self.goal) != 0:
                rospy.loginfo_throttle(1.0, "Moving without path! Testing?")

            else:
                rospy.logwarn_throttle(1.0, "No path or goal, stopping movement")
                self.__send_commnd__(0, 0)
                return  
        
        if not self.good_pose or not hasattr(self.current_pose, '__len__'):
            print(self.current_pose, self.good_pose)
            rospy.logwarn("[Planner] Outdated control values, stopping movement")
            self.__send_commnd__(0,0)
            return 

        try:
            self.waypoint = self.path[0]
            self.control_loop()
        
        except Exception as e:
            rospy.logerr(f'DWA failure: {str(e)}\n{traceback.format_exc()}')
            self.__send_commnd__(0,0)
    
    def control_loop(self):
        '''executes main loop for following path'''
        if self.planning:
            rospy.logwarn_throttle(2.0, 'Still planning...')
            # self.weights = np.array([0.7, 1.0, 1.4, 0.0])
            self.__send_commnd__(0,0)
            return
        else:
            self.weights = np.array([0.1, 1.0, 0.6, 1.3])
            self.heading_diff, _ = self.heading_diff_calc(self.current_pose[:3], self.waypoint)
            # if self.heading_diff < -np.pi/2:
            #     rospy.logwarn_throttle(2.0, 'Adjusting heading to align robot with goal.')
            #     rospy.loginfo(self.current_pose[2])
            #     rospy.loginfo(self.heading_diff)
            #     self.__send_commnd__(0,self.v_lim[1]*0.3)
            # elif self.heading_diff > np.pi/2:
            #     self.__send_commnd__(0,-self.v_lim[1]*0.3)
            #     rospy.loginfo(self.current_pose[2])
            #     rospy.loginfo(self.heading_diff)
            #     rospy.logwarn_throttle(2.0, 'Adjusting heading to align robot with goal.')
        goal_location = self.goal[0]==self.path[0][0] and self.goal[1]==self.path[0][1]
        if goal_location:
            rospy.loginfo_throttle(3.0,'going to goal')
            # Goal is Reached
            if dist_between_points(self.current_pose[:2], self.goal[:2]) < 0.1:
                self.__send_commnd__(0,0)
                self.path = []
                self.goal = []
                rospy.loginfo("Viewpoint reached!!!")
                msg = dwa()
                msg.replan_bool = True
                rospy.loginfo("Replan requested")
                self.replan_pub.publish(msg)
                
        # Moves onto next waypoint in path
        elif self.path[0][0] != self.goal[0]:
            rospy.loginfo_throttle(3.0,'going to waypoint')
            rospy.loginfo_throttle(3.0, f'current pose: {self.current_pose[:2]}')
            rospy.loginfo_throttle(3.0, f'waypoint: {self.path[0]}')
            if np.linalg.norm(self.current_pose[:2] - self.path[0]) <= 0.25:
                del self.path[0]
                return
        self.current_velocity, best_course, arcs = self.create_DWA_arcs([self.current_velocity[0],self.current_velocity[1]])
        
        # if stuck set angular vel to turn away from obstacles
        if not hasattr(best_course, '__len__'):
            rospy.logerr('no good trajectories turning without moving to see if it helps')
            self.__send_commnd__(0,0.3)
        
        # Publish velocity commands
        
        # RVIZ plots for trajectories
        self.publish_dwa_arcs(best_course, arcs)
        self.__send_commnd__(self.current_velocity[0], -self.current_velocity[1])
        current_time = rospy.Time.now().to_sec()
        loop_time = current_time - self.last_control_time
        rospy.logwarn_throttle(5.0, f'Time laken for one loop: {loop_time}')
        self.last_control_time = current_time
        self.control_iteration += 1
    
    # PUBLISHER HELPERS

    # Transform linear and angular velocity (v, w) into a Twist message and publish it
    def __send_commnd__(self, v, w):
        cmd = Twist()
        cmd.linear.x = v
        cmd.linear.y = 0
        cmd.linear.z = 0
        cmd.angular.x = 0
        cmd.angular.y = 0
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

    def publish_obstacles(self, obstacles):
        marker_array = MarkerArray()
        
        delete_marker = Marker()
        delete_marker.header.frame_id = 'world_ned'
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.ns = 'obstacles'
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        if obstacles:
            green = ColorRGBA(0, 1, 0, 0.7)
            
            m = Marker()
            m.header.frame_id = 'world_ned'
            m.header.stamp = rospy.Time.now()
            m.ns = 'obstacles'
            m.id = 1
            m.type = Marker.SPHERE_LIST
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            
            if len(obstacles) > 0:
                m.scale.x = obstacles[0][2] * 2  
                m.scale.y = obstacles[0][2] * 2
                m.scale.z = obstacles[0][2] * 2
            
            for i, (x, y, radius) in enumerate(obstacles):
                p = Point()
                p.x = x
                p.y = y
                p.z = -0.3  
                m.points.append(p)
                m.colors.append(green)
            
            marker_array.markers.append(m)
        
        self.ob_pub.publish(marker_array)



    def publish_dwa_arcs(self, best, possible_arcs):
        delete_marker = Marker()
        delete_marker.header.frame_id = 'world_ned'
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.ns = 'dwa'
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        self.dwa_pub.publish(delete_marker)

        color_orange=ColorRGBA(1,0.647,0,1)
        color_yellow=ColorRGBA(1,1,0,1)
        marker_scale = Point(0.03, 0.06, 0.1)

        for i, arc in enumerate(possible_arcs):
            m = Marker()
            m.header.frame_id = 'world_ned'
            m.header.stamp = rospy.Time.now()
            m.id = 2*i+2
            m.type = Marker.LINE_STRIP
            m.ns = 'dwa'
            m.action = Marker.ADD
            m.scale.x = 0.01
            m.pose.orientation.w = 1.0
            for point in arc:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = -0.3
                m.points.append(p)
                m.colors.append(color_yellow)
                m.lifetime = rospy.Duration(self.dt)
            self.dwa_pub.publish(m)

            arrow_marker = Marker()
            arrow_marker.header.frame_id = 'world_ned'
            arrow_marker.ns = 'dwa'
            arrow_marker.id = 2*i+3
            arrow_marker.type = Marker.ARROW
            arrow_marker.scale = marker_scale
            for j in [-2,-1]:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = -0.3
                arrow_marker.points.append(p)
                arrow_marker.colors.append(color_yellow)
                arrow_marker.lifetime = rospy.Duration(self.dt)
            self.dwa_pub.publish(arrow_marker)



        if best is not None:
            ideal = Marker()
            ideal.header.frame_id = 'world_ned'
            ideal.header.stamp = rospy.Time.now()
            ideal.ns = 'dwa'
            ideal.id = 1001
            ideal.type = Marker.LINE_STRIP
            ideal.action = Marker.ADD
            ideal.scale.x = 0.03
            ideal.pose.orientation.w = 1.0
            for point in best:
                if point is not None:
                    p = Point()
                    p.x = point[0]
                    p.y = point[1]
                    p.z = -0.35
                    ideal.points.append(p)
                    ideal.colors.append(color_orange)
                    ideal.lifetime = rospy.Duration(self.dt)
                else:
                    print('WHYYYYYYYYYYYYYYYYY')
            self.dwa_pub.publish(ideal)

            ideal_arrow = Marker()
            ideal_arrow.header.frame_id = 'world_ned'
            ideal_arrow.ns = 'dwa'
            ideal_arrow.id = 1000
            ideal_arrow.type = Marker.ARROW
            ideal_arrow.scale = marker_scale
            for j in [-2,-1]:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = -0.3
                ideal_arrow.points.append(p)
                ideal_arrow.colors.append(color_yellow)
                ideal_arrow.lifetime = rospy.Duration(self.dt)
            self.dwa_pub.publish(ideal_arrow)
        else:
            rospy.logerr('no best?')
    

# MAIN FUNCTION
if __name__ == '__main__':
    rospy.init_node('frontier_dwa_node', log_level=rospy.DEBUG)   

    node = DWA(0.2, '/turtlebot/odom_ground_truth', '/turtlebot/kobuki/commands/velocity', '/path_for_dwa', '/replan', '/turtlebot/kobuki/sensors/rplidar')
    
    # Run forever
    rospy.spin()