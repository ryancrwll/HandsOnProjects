#!/usr/bin/python3

import rospy
import numpy as np
import traceback
from geometry_msgs.msg import PoseStamped, Twist, PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from frontier_search_dwa.msg import dwa
import tf
from online_planning import StateValidityChecker, dist_between_points

def wrap_angle(angle):
    '''returns positive values of difference of angles reguardless if it wraps around 360 or not'''
    return abs((angle + np.pi) % (2*np.pi) - np.pi)

class DWA:
    def __init__(self, radius, gridmap_topic, odom_topic, cmd_vel_topic, path_topic, replan_topic):
        # From frontier_search.py
        self.current_pose = None # (x,y,theta) #TODO put back if makes sense
        self.last_odom_time = rospy.Time.now().to_sec() # last logged time from receiving pose (seconds)
        self.good_pose = False # bool to tell if pose is recent enough to use
        self.path = []
        self.waypoint = None
        self.svc = StateValidityChecker(radius) 
        self.map_loaded = False # makes sure a map is ready
        # Last time a map was received (to avoid map update too often)                                                
        self.last_map_time = rospy.Time.now()
        self.obstacles = [] # first two args are center coords and third is radius
        if self.svc.map is not None:
            self.create_obstacles() 
        self.goal = None # need goal to know stopping distance (x,y,theta)
        
        #### DWA CONTROLLER PARAMETERS ###
        # Maximum linear velocity control action, # Maximum angular velocity control action                 
        self.v_lim = np.array([0.2,0.2])
        # linear and angular accel limits
        self.a_lim  = np.array([0.5, 1.0])    
        # Current Velocity
        self.current_velocity = [0.0, 0.0]  
        # DWA weights for tuning dyanmic window (heading, clearance, velocity, distance to goal) 
        self.weights = np.array([0.7, 0.8, 1.0, 1.3]) # np.array([0.4, 0.8, 1.0, 1.3]) weights for tuning dyanmic window (heading, clearance, velocity, distance to goal)
        # First velocity always opposite #TODO figure out!
        self.control_iteration = 0
        # Last time control function was called to know how ofen its gets called 
        self.last_control_time = None
        self.sim_time = 1.0 # how far ahead to project velocities (seconds)
        self.dt = 0.1 # time step for simulating trajectories (seconds)
        self.radius = self.svc.distance # radius of robot (meters)
        self.num_vel = 5 # sqrt of number of simulated trajectories to create

        # PUBLISHERS
        # Publisher for sending velocity commands to the robot
        self.cmd_pub =  rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)
        # Publisher for possible dwa_arcs
        self.dwa_pub = rospy.Publisher('/dwa_arcs', Marker, queue_size=1)
        # Publisher to tell planner to replan
        self.replan_pub = rospy.Publisher(replan_topic, dwa, queue_size=1)
        
        # SUBSCRIBERS
        self.gridmap_sub = rospy.Subscriber(gridmap_topic, OccupancyGrid, self.get_gridmap) #subscriber to gridmap_topic from Octomap Server  
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.get_odom, queue_size=5) #subscriber to odom_topic  
        self.path_sub = rospy.Subscriber(path_topic, dwa, self.get_path)
        self.gotolocation = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.get_selected_goal) #subscriber to /move_base_simple/goal published by rviz
        # TIMERS
        # Timer for velocity controller
        rospy.Timer(rospy.Duration(0.05), self.controller)

        # Should be same in both nodes
        print(f"SVC params - res: {self.svc.resolution}, origin: {self.svc.origin}")
    
    # Odometry callback: Gets current robot pose and stores it into self.current_pose
    def get_odom(self, odom):
            _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
                                                                odom.pose.pose.orientation.y,
                                                                odom.pose.pose.orientation.z,
                                                                odom.pose.pose.orientation.w])
            # Store current position (x, y, yaw) as a np.array in self.current_pose var.
            self.current_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw])
            self.good_pose = rospy.Time.now().to_sec() - self.last_odom_time < 0.2
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
            self.svc.set(env, gridmap.info.resolution, origin, flip=True)
            # self.svc.map_viz_debug(self.svc.map, 'svc', cell_pos)
            self.map_loaded = True
            self.create_obstacles()
                
    def get_path(self, msg):
        # Pass numpy array of path from frontier_dwa.py
        self.path = [[p.pose.position.x, p.pose.position.y] for p in msg.poses]
        if len(msg.poses) > 0:
            self.goal = [msg.poses[-1].pose.position.x, msg.poses[-1].pose.position.y]
    
    def get_selected_goal(self, goal):
        #for debugging dwa
        self.need_new_path = False
        _, _, yaw = tf.transformations.euler_from_quaternion([goal.pose.orientation.x, 
                                                                goal.pose.orientation.y,
                                                                goal.pose.orientation.z,
                                                                goal.pose.orientation.w])
        # Store current position (x, y, yaw) as a np.array in self.current_pose var.
        self.goal = np.array([goal.pose.position.x, goal.pose.position.y])
        rospy.loginfo(f'Goal selected at {self.goal}')
        self.path = np.array([self.goal[:2]])
    
    def create_obstacles(self):
        if self.svc.map is None:
            return
        clearance = np.hypot(self.svc.resolution,self.svc.resolution)
        self.obstacles = []

        height, width = self.svc.map.shape
        center_y = self.svc.origin[1] + (height * self.svc.resolution) / 2
        center_x = self.svc.origin[0] + (width * self.svc.resolution) / 2
        try:
            obstacle_indices = np.argwhere(self.svc.map == 1)
            for i,j in obstacle_indices:
                map_x = j
                map_y = i
                
                # Convert to world coordinates by mirroring about center axes
                world_x = center_x*2 - (self.svc.origin[0] + (map_x + 0.5) * self.svc.resolution)
                world_y = center_y*2 - (self.svc.origin[1] + (map_y + 0.5) * self.svc.resolution)
                
                self.obstacles.append([world_x, world_y, clearance])
        except Exception as e:
            rospy.logerr(f"Obstacle creation failed: {str(e)}\n{traceback.format_exc()}")
            self.obstacles = []

    def motion_model(self, pose, u, dt=None):#TODO maybe switching x and y poses helps
        # standard motion model
        if dt == None:
            dt = self.dt
        x,y,theta = pose.astype(float)
        v, w = u
        # assuming straight line trajectory
        if w < .001:
            x += v * np.cos(theta)*dt
            y += v * np.sin(theta)*dt
            theta += w*dt
        # curved trajectory
        else:
            R = v/w
            icr = np.array([x + R*np.cos(theta + np.pi/2),
                            y + R*np.sin(theta + np.pi/2)])
            beta = -(np.pi/2 - theta) + w*dt
            x = icr[0] + R*np.cos(beta)
            y = icr[1] + R*np.sin(beta)
            theta += w*dt
        return np.array([x,y,theta], dtype=float)
    
    def generate_DWA(self, vel):
        staticConstraints = [0.0, self.v_lim[0], -self.v_lim[1], self.v_lim[1]]
        stopping_dist = np.linalg.norm(self.goal[:2] - self.current_pose[:2])
        stop_assurance = self.svc.map.shape[0]*self.svc.resolution/50
        if stopping_dist > stop_assurance:
            stopping_dist -= stop_assurance
        # from physics equation vf^2 - vi^2 = 2 * a *d
        max_vel = np.sqrt(2 * self.a_lim[0] * stopping_dist)
        max_vel = min(max_vel, self.v_lim[0])
        
        # if np.abs(vel[0]<0.01) and stopping_dist>1.0:
        #     dwa = np.array(staticConstraints)
        #     dwa[1] = max_vel
        # else:
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
        vector = self.waypoint - end_pose[:2]
        dist2goal = np.linalg.norm(vector)
        angle_needed = np.arctan2(vector[1], vector[0])
        heading_diff = wrap_angle(angle_needed - end_pose[2]) 

        if dist2goal < 0.25:
            d_score = -dist2goal*10
            heading_diff = heading_diff*5
        else:
            d_score = -dist2goal
        
        closest = np.inf

        #freeze values so that its not obdated during scoring
        obstacles = self.obstacles

        for i in range(len(obstacles)):
            for j in range(len(path)):
                dist_obsticle = np.linalg.norm([obstacles[i][0] - path[j][0], obstacles[i][1] - path[j][1]]) - obstacles[i][2] - self.radius
                if dist_obsticle < closest:
                    closest = dist_obsticle
        rospy.loginfo_throttle(1.0, f'nearest obstacle: {closest}')

        return heading_diff, closest, d_score
    
    def create_DWA_arcs(self, current_vel):
        "creates all possible trajectories within a ceratin velocity limits and at a number of intervals of num_vel and selects the one with best score"
        end_pose = self.current_pose
        vector = self.waypoint - end_pose[:2]
        dist2goal = np.linalg.norm(vector)

        rospy.loginfo_throttle(1.0, f'how close? {dist2goal}')
        angle_needed = np.arctan2(vector[1], vector[0])

        heading_diff = wrap_angle(angle_needed - end_pose[2]) 
        rospy.loginfo_throttle(1.0, f'heading_diff: {np.degrees(heading_diff)}')

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
                h_score = 1 - (h_score/np.pi) 
                # use log to minimize importance of clearances that are very far away
                c_score = np.log(c_score+0.01) # avoids log zero
                # velocity score uses initial velocity and not final bc assuming constant velocity over the window bc it is our control input
                # normalize it so that 1 is highest possible value
                v_score = (dyn_windowL[i]/self.v_lim[0]);''' + 0.3*(dyn_windowA[i]/self.v_lim[1])'''
                score = np.array([h_score,c_score,v_score,d_score]) @ self.weights
                if score > best_score:
                    best_score = score
                    best_u = np.array([dyn_windowL[i], dyn_windowA[j]])
                    best_course = arc
        return best_u, best_course, arcs
    
    # This method is called every 0.1s. It computes the velocity comands in order to reach the 
    # next waypoint in the path. It also sends zero velocity commands if there is no active path.
    def controller(self, event):

        while not self.map_loaded and not rospy.is_shutdown():
            rospy.loginfo_throttle(1.0, "[dwa] Waiting for map to become ready...")

        if len(self.path) == 0:
            if self.goal is not None:
                rospy.loginfo_throttle("Moving without path! Testing?")

            else:
                rospy.loginfo_throttle(1.0, "No path or goal, stopping movement")
                self.__send_commnd__(0, 0)
                return  
        
        if not self.good_pose or not hasattr(self.current_pose, '__len__'):
            print(self.current_pose, self.good_pose)
            rospy.logwarn("[Planner] Outdated control values, stopping movement")
            self.__send_commnd__(0,0)
            return 

        try: #TODO angular velocities are switched
            self.waypoint = self.path[0]
            self.control_loop()
        
        except Exception as e:
            rospy.logerr(f'DWA failure: {str(e)}\n{traceback.format_exc()}')
            self.__send_commnd__(0,0)
    
    def control_loop(self):
        '''executes main loop for following path'''
        goal_location = self.goal[0]==self.path[0][0] and self.goal[1]==self.path[0][1]
        if goal_location:
            # Goal is Reached
            if dist_between_points(self.current_pose[:2], self.goal[:2]) < 0.02:
                self.__send_commnd__(0,0)
                del self.path[0]
                rospy.loginfo("Viewpoint reached!!!")
                msg = dwa()
                msg.replan_bool = True
                self.replan_pub.publish(msg)
                
        # Moves onto next waypoint in path
        elif dist_between_points(self.current_pose[:2], self.path[0]) <= 0.2 and not goal_location:
            del self.path[0]
            return
        # Updating current pose and velocities
        self.current_velocity, best_course, arcs = self.create_DWA_arcs([self.current_velocity[0],self.current_velocity[1]])
        # self.current_pose = dwa.motion_model(self.current_pose, self.current_velocity) #TODO maybe this would help get more recent pose updates
        # RVIZ plots for trajectories
        self.publish_dwa_arcs(best_course, arcs)
        # Publish velocity commands
        # To avoid sign confusion from first publish      
        if self.control_iteration >= 0:
            self.current_velocity[1] *= -1
        #print("v: ", v, "w: ", w)
        self.__send_commnd__(self.current_velocity[0], self.current_velocity[1])
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

    def publish_dwa_arcs(self, best, possible_arcs):
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
                    m.lifetime = rospy.Duration(self.dt)
        self.dwa_pub.publish(m)

        goal_marker = Marker()
        goal_marker.header.frame_id = "world_ned"
        goal_marker.type = Marker.SPHERE
        goal_marker.scale.x = goal_marker.scale.y = goal_marker.scale.z = 0.3
        goal_marker.color.r = 1.0; goal_marker.color.a = 1.0
        goal_marker.pose.position.x = self.goal[0]
        goal_marker.pose.position.y = self.goal[1]
        self.dwa_pub.publish(goal_marker)

# MAIN FUNCTION
if __name__ == '__main__':
    rospy.init_node('frontier_dwa_node')   

    node = DWA(0.2, '/projected_map', '/turtlebot/odom_ground_truth', '/turtlebot/kobuki/commands/velocity', '/path_for_dwa', '/replan')
    
    # Run forever
    rospy.spin()