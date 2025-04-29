import numpy as np
import rospy
from nav_msgs.msg import Odometry
import tf
from online_planning import StateValidityChecker

class DWA:
    def __init__(self, current_pose, next_waypoint, svc: StateValidityChecker, goal, vel_limits, accel_limits, radius, weights, odom_topic):
        # self.current_pose = None # (x,y,theta)
        self.current_pose = current_pose # (x,y,theta) #TODO put back if makes sense
        self.map = svc.map.T # map values to know obstacles to avoid 
        self.goal = goal # need goal to know stopping distance (x,y,theta)
        self.weights = weights # weights for tuning dyanmic window (heading, clearance, velocity, distance to goal)
        self.v_lim = vel_limits # (linear velocity, angular velocity) maximum acheivable or wanted
        self.a_lim = accel_limits # (linear acceleration, angular acceleration) maximum acheivable or wanted
        self.sim_time = 10.0 # how far ahead to project velocities (seconds)
        self.dt = 0.1 # time step for simulating trajectories (seconds)
        self.radius = radius # radius of robot (meters)
        self.num_vel = 5 # sqrt of number of simulated trajectories to create
        self.svc = svc
        self.obstacles = [] # first two args are center coords and third is radius
        if self.map is not None:
            self.create_obstacles()
        self.waypoint = next_waypoint # next point along the path to steer to   

        # Subscriber for current pose
        self.odom_sub = rospy.Subscriber("/turtlebot/odom_ground_truth", Odometry, self.get_odom) #subscriber to odom_topic  

    # Odometry callback: Gets current robot pose and stores it into self.current_pose
    def get_odom(self, odom):
        _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
                                                              odom.pose.pose.orientation.y,
                                                              odom.pose.pose.orientation.z,
                                                              odom.pose.pose.orientation.w])
        # Store current position (x, y, yaw) as a np.array in self.current_pose var.
        self.current_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw])  
        # print(f'debug{self.current_pose}') 

    def create_obstacles(self):
        clearance = np.hypot(self.svc.resolution,self.svc.resolution)
        self.obstacles = []
        # for i in range(self.map.shape[0]):
        #     for j in range(self.map.shape[1]):
        #         if self.map[i,j] == 1:
        #             loc = self.svc.map_to_position(np.array([i,j]))
        #             self.obstacles.append([np.array([loc[0], loc[1], clearance])])
        try:
            obstacle_indices = np.argwhere(self.map == 1)
            for i,j in obstacle_indices:
                loc = self.svc.map_to_position(np.array([i,j]))
                if loc is not None:
                    self.obstacles.append([loc[0],loc[1],clearance]) #TODO maybe switch loc[0] and loc[1]
        except Exception as e:
            rospy.logerr(f"Obstacle creation failed: {str(e)}")
            self.obstacles = []

    def motion_model(self, pose, u):#TODO maybe switching x and y poses helps
        # standard motion model
        pose[1] += u[0] * np.cos(pose[2])*self.dt
        pose[0] += u[0] * np.sin(pose[2])*self.dt
        pose[2] += u[1]*self.dt
        return pose
    
    def generate_DWA(self, vel):
        staticConstraints = [0.0, self.v_lim[0], -self.v_lim[1], self.v_lim[1]]
        print(f'static constraints {staticConstraints}')

        print(f'debug1{self.current_pose}') 
        stopping_dist = np.linalg.norm(self.goal[:2] - self.current_pose[:2])
        stop_assurance = self.map.shape[0]*self.svc.resolution/50
        if stopping_dist > stop_assurance:
            stopping_dist -= stop_assurance
        # from physics equation vf^2 - vi^2 = 2 * a *d
        max_vel = np.sqrt(2 * self.a_lim[0] * stopping_dist)
        # max_vel = min(max_vel, self.v_lim[0])
        
        # if np.abs(vel[0]<0.01) and stopping_dist>1.0:
        #     dwa = np.array(staticConstraints)
        #     dwa[1] = max_vel
        # else:
        dynamicConstraints = [vel[0] - self.a_lim[0]*self.dt,
                                vel[0] + self.a_lim[0]*self.dt,
                                vel[1] - self.a_lim[1]*self.dt,
                                vel[1] + self.a_lim[1]*self.dt]
        print(f'dynamicConstraints {dynamicConstraints}')
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
        angle_needed = np.arctan2(self.waypoint[1] - end_pose[1], self.waypoint[0] - end_pose[0])
        dist2goal = np.linalg.norm(self.waypoint - end_pose[:2])
        # returns positive values of difference of angles reguardless if it wraps around 360 or not
        heading_diff = abs(np.arctan2(np.sin(angle_needed - end_pose[2]), np.cos(angle_needed - end_pose[2])))

        closest = np.inf
        for i in range(len(self.obstacles)):
            for j in range(len(path)):
                dist_obsticle = np.linalg.norm([self.obstacles[i][0] - path[j][0], self.obstacles[i][1] - path[j][1]]) - self.obstacles[i][2] - self.radius
                if dist_obsticle < closest:
                    closest = dist_obsticle
        return heading_diff, closest, dist2goal
    
    def create_DWA_arcs(self, current_vel):
        "creates all possible trajectories within a ceratin velocity limits and at a number of intervals of num_vel and selects the one with best score"
        dw = self.generate_DWA(current_vel)
        print(f"Velocity window: {dw}")
        dyn_windowL = np.linspace(dw[0], dw[1], self.num_vel)
        dyn_windowA = np.linspace(dw[2], dw[3], self.num_vel)
        arcs = []
        best_score = -np.inf
        for i in range(len(dyn_windowL)):
            for j in range(len(dyn_windowA)):
                arc = self.predict_course(np.array([dyn_windowL[i], dyn_windowA[j]]))
                arcs.append(arc) #for plotting
                h_score, c_score, d_score = self.calc_scoring_vals(arc)
                # norm values to be range (0,1)
                h_score /= np.pi 
                # use log to minimize importance of clearances that are very far away
                c_score = np.log(c_score+0.001) # avoids log zero
                # velocity score uses initial velocity and not final bc assuming constant velocity over the window bc it is our control input
                # normalize it so that 1 is highest possible value
                v_score = (dyn_windowL[i]/self.v_lim[0]);''' + 0.3*(dyn_windowA[i]/self.v_lim[1])'''
                # makes values negative where smaller is better such that larger valuses are more desired, better for finding larger score
                score = np.array([-h_score,c_score,v_score,-d_score]) @ self.weights
                if score > best_score:
                    best_score = score
                    best_u = np.array([dyn_windowL[i], dyn_windowA[j]])
                    best_course = arc
        return best_u, best_course, arcs