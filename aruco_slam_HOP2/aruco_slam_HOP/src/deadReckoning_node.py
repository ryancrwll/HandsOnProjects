#!/usr/bin/env python3

import math
from math import *
import rospy
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import JointState, Imu, LaserScan
from visualization_msgs.msg import Marker, MarkerArray
import threading
from geometry_msgs.msg import Quaternion, PoseStamped, Point
from std_msgs.msg import Header, ColorRGBA


# Define the Dead Reckoning Node class
class DeadReckoningNode:
    def __init__(self):
        # Intialize ROS node
        rospy.init_node("dead_reckoning_node", anonymous=True)

        # Robot spcific parameters
        self.wheel_radius = 0.035           # Radius of wheels
        self.wheel_base_distance = 0.235    # Distance between left and right wheels

        self.xb_dim = 3         # Base state dimension: [x, y, theta]
        self.xF_dim = 2         # Feature dimension

        # Initial robot pose [x, y, theta] and uncertainity
        self.xk = np.zeros((self.xb_dim, 1))        # Robot state vector
        self.Pk = np.diag([0.1, 0.2, 0.1])          # Covariance matrix for initial uncertainty
        
        # Motion model noise (process noise) covariance
        self.Qk = np.diag([0.2**2, 0.2**2, 0.01**2])

        # Wheel velocities and flags to check if both are recieved
        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0
        self.left_wheel_velocity_received = False
        self.right_wheel_velocity_received = False
        self.mutex = threading.Lock()                    # For thread-safe access

        # Time tracking
        self.last_time = rospy.Time.now()

        # Publish the predicted odometry of the robot (i.e., estimated position, orientation, and velocity).
        self.odom_pub = rospy.Publisher("turtlebot/kobuki/odom_predict", Odometry, queue_size=10)
        # Publish the entire robot path
        self.path_pub = rospy.Publisher("turtlebot/kobuki/path_update", Path, queue_size=10)
        # Publish a marker that visually represents the robot's path in RVIZ
        self.path_marker_pub = rospy.Publisher("turtlebot/kobuki/path_marker_update", Marker, queue_size=10)
        # Publish a marker representing the uncertainty ellipse of the robot's position
        self.marker_pub_odom = rospy.Publisher("visualization_marker/odom", Marker, queue_size=10)

        # Path initialization
        self.path = Path()
        self.path.header.frame_id = "odom"

        # Subscriber to joint states to receive wheel encoder data (joint velocities)
        self.js_sub = rospy.Subscriber("turtlebot/joint_states", JointState, self.joint_states_callback, queue_size=10)
        # Subscriber to IMU
        self.imu_sub = rospy.Subscriber("turtlebot/kobuki/sensors/imu", Imu, self.imu_callback, queue_size=10)

        # TF broadcaster for transformation between frames
        self.odom_broadcaster = tf.TransformBroadcaster()

        rospy.loginfo("Dead Reckoning node started (wheel data + IMU)")


    def wrap_angle(self, angle):
        '''
            Normalize angle to [-pi, pi]
        '''
        return (angle + (2.0 * np.pi * np.floor((np.pi - angle) / (2.0 * np.pi))))


    def joint_states_callback(self, msg):
        '''
            Callback function that processes messages from the joint states topic.

            param msg: The message recieved from the joint states topic, 
                       containing the names of joints and their velocities (wheel encoder data).
        '''
        self.left_wheel_name = 'turtlebot/kobuki/wheel_left_joint'      # Name of left joint
        self.right_wheel_name = 'turtlebot/kobuki/wheel_right_joint'    # Name of the right joint

        # Check the first joint name in the message and assign the corresponding velocity
        if msg.name[0] == self.left_wheel_name:
            self.left_wheel_velocity = msg.velocity[0]      # w_L = angular velocity of the left wheel
            self.left_wheel_velocity_received = True

        elif msg.name[0] == self.right_wheel_name:
            self.right_wheel_velocity = msg.velocity[0]     # w_R = angular velocity of the right wheel
            self.right_wheel_velocity_received = True

        if self.left_wheel_velocity_received and self.right_wheel_velocity_received:
            # Compute the linear velocity for each wheel by multiplying the angular velocity by the wheel radius
            self.left_linear_velocity = self.left_wheel_velocity * self.wheel_radius
            self.right_linear_velocity = self.right_wheel_velocity * self.wheel_radius

            # Calculate the overall linear and angular velocities
            '''
                linear velocity, v = (w_L + w_R) * wheel_radius / 2
                angular velocity, w = (w_L - w_R) * wheel_radius / wheel_base
            '''
            self.linear_velocity = (self.left_linear_velocity + self.right_linear_velocity) / 2
            self.angular_velocity = (self.left_linear_velocity - self.right_linear_velocity) / self.wheel_base_distance

            # Compute the current time from the message stamp and calculate the time elapsed since the last update
            self.current_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
            self.delta_time = (self.current_time - self.last_time).to_sec()
            self.last_time = self.current_time

            # Update the robot's pose using the motion model
            '''
                Motion Update Equations:
                                            x_k+1 = x_k + v * cos(theta) * dt
                                            y_k+1 = y_k + v * sin(theta) * dt
                                            theta_k+1 = theta_k + w * dt
            '''
            self.xk[0,0] = self.xk[0,0] + np.cos(self.xk[2,0]) * self.linear_velocity * self.delta_time
            self.xk[1,0] = self.xk[1,0] + np.sin(self.xk[2,0]) * self.linear_velocity * self.delta_time
            self.xk[2,0] = self.xk[2,0] + self.angular_velocity * self.delta_time

            # Normalize the robot's orientation angle
            self.xk[2,0] = self.wrap_angle(self.xk[2,0])

            # EKF prediction step for motion update
            self.xk, self.Pk = self.prediction(self.xk, self.Pk, self.linear_velocity, self.angular_velocity, self.delta_time)

            # Publishing the predicted odometry
            self.publish_odometry(self.xk, self.Pk)

            # Update the path
            self.update_path()

            # Reset the velocity reception flags for the next iterations
            self.left_wheel_velocity_received = False
            self.right_wheel_velocity_received = False


    def prediction(self, xk, Pk, v, w, dt):
        '''
            EKF Predicts the next state of the robot using the motion model.

            :param xk: Current state vector of the robot.
            :param Pk: Current state covariance matrix.
            :param v: Linear velocity of the robot.
            :param w: Angular velocity of the robot.
            :param t: Time interval since the last update.
            
            :return: Updated state vector and covariance matrix after prediction.
        '''
        # Acquire the lock to ensure thread safety during prediction and update
        self.mutex.acquire()

        # Extract the base state from the state vector
        xk_robot = self.xk[:self.xb_dim]    # [x, y, theta]

        '''
            Calculating Jacobians to linearize the motion model
        '''
        # Calculate Ak = Jacobian of the motion model with respect to the robot's state   
        # i.e. partial differentiation of robot states w.r.t. robot states (itself)  
        '''
            Motion Update Equations:
                                        x_k+1 = x_k + v * cos(theta) * dt
                                        y_k+1 = y_k + v * sin(theta) * dt
                                        theta_k+1 = theta_k + w * dt
        '''
        Ak = np.array([
                        [1.0, 0.0, -np.sin(self.xk[2,0]) * v * dt],
                        [0.0, 1.0,  np.cos(self.xk[2,0]) * v * dt],
                        [0.0, 0.0,          1.0                 ]
                    ])

        # Calculate Wk = Jacobian of the motion model with respect to the process noise (wheel velocities)  
        # i.e. partial differentiation of robot states w.r.t. w_L & w_R 
        '''
            w_L, w_R = angular velocities of the left and right wheels

            Influence of wheel velocities on state:
                                                    x = (cos(theta) * r * dt) / 2  (same influence by both wheels)
                                                    y = (sin(theta) * r * dt) / 2  (same influence by both wheels)
                                                    theta = (r * t) / wheel_base   (left wheel influence)
                                                    theta = -(r * t) / wheel_base  (right wheel influence)

            linear velocity, v = (w_L + w_R) * wheel_radius / 2
            angular velocity, w = (w_L - w_R) * wheel_radius / wheel_base

            Motion Update Equations:
                                        x_k+1 = x_k + v * cos(theta) * dt
                                        y_k+1 = y_k + v * sin(theta) * dt
                                        theta_k+1 = theta_k + w * dt

            In the motion update equations, we will equate the values of v & w in terms of
            w_L and w_R and then do the partial differentiation of robot states w.r.t w_L & w_R 
        '''
        Wk = np.array([
                        [np.cos(self.xk[2,0]) * self.wheel_radius * dt * 0.5,  np.cos(self.xk[2,0]) * self.wheel_radius * dt * 0.5, 0.0],    
                        [np.sin(self.xk[2,0]) * self.wheel_radius * dt * 0.5,  np.sin(self.xk[2,0]) * self.wheel_radius * dt * 0.5, 0.0],           
                        [(dt * self.wheel_radius) / self.wheel_base_distance, -(dt * self.wheel_radius) / self.wheel_base_distance, 1.0]])

        # Process noise covariance matrix accounting for the uncertainties in the motion model
        Qk = self.Qk

        # Update the state covariance matrix
        self.Pk = Ak @ Pk @ Ak.T + Wk @ Qk @ Wk.T

        self.mutex.release()

        return self.xk, self.Pk
    

    def publish_odometry(self, xk, Pk):
        """
        Publishes odometry data to ROS.

        :param xk: Current state vector [x, y, theta].
        :param Pk: Covariance matrix of the state.
        """
        # Convert yaw angle to a quaternion for representing 3D orientation
        self.q = quaternion_from_euler(0, 0, xk[2,0])

        # Initialize an odometry message
        odom = Odometry()
        odom.header.stamp = self.current_time
        odom.header.frame_id = "odom"
        odom.child_frame_id = "odom_deadreckoning"      

        # Set position in the odometry message
        odom.pose.pose.position.x = xk[0,0]
        odom.pose.pose.position.y = xk[1,0]

        # Set the orientation in the odometry message.
        odom.pose.pose.orientation.x = self.q[0]
        odom.pose.pose.orientation.y = self.q[1]
        odom.pose.pose.orientation.z = self.q[2]
        odom.pose.pose.orientation.w = self.q[3]

        # Set the velocities in the odometry message
        odom.twist.twist.linear.x = self.linear_velocity
        odom.twist.twist.angular.z = self.angular_velocity

        # Convert the covariance matrix from np.array to list
        P_list = Pk.tolist()

        '''
            The covariance matrix is a 6x6 matrix for [x, y, z, roll, pitch, yaw]
        '''

        # Update the diagonal elements directly for variance
        odom.pose.covariance[0] = P_list[0][0]  # Variance in x
        odom.pose.covariance[7] = P_list[1][1]  # Variance in y
        odom.pose.covariance[35] = P_list[2][2]  # Variance in yaw

        # Update the off-diagonal elements for covariance between variables
        odom.pose.covariance[1] = P_list[0][1]  # Covariance between x and y
        odom.pose.covariance[6] = P_list[1][0]  # Covariance between y and x 

        odom.pose.covariance[5] = P_list[0][2]  # Covariance between x and yaw
        odom.pose.covariance[30] = P_list[2][0]  # Covariance between yaw and x

        odom.pose.covariance[11] = P_list[1][2]  # Covariance between y and yaw
        odom.pose.covariance[31] = P_list[2][1]  # Covariance between yaw and y

        # Marker visualization for Covariance of the robot state 
        '''
            Extract the top-left 2x2 block from the 3x3 covariance matrix Pk.
            i.e. the covariance matrix for just the x and y position of the robot.
        '''
        position_covariance_2d = self.Pk[0:2, 0:2]
        '''
            Computing the eigenvalues and eigenvectors of the 2x2 uncertainity matrix.
            The eigenvectors gives us the principal axes (orientation) of the uncertainity ellipse.
            The eigenvalues gives us the spread (length of axes)
        '''
        eigenvalues, eigenvectors = np.linalg.eigh(position_covariance_2d)

        # Compute the orientation angle (yaw) of the uncertainity ellipse
        ellipse_yaw = np.arctan2(eigenvectors[1,0], eigenvectors[0,0])
        # Convert the ellipse yaw into a quaternion for RViz marker orientation
        '''
            A quaternion is a 4-element vector used to represent rotations in 3D space:
                                        q = [x, y, z, w]
            
            x,y,z: imaginary (vector) components — they define the axis of rotation
            w: real (scalar) component — it defines the amount of rotation
        '''
        ellipse_orientation_quat = quaternion_from_euler(0, 0, ellipse_yaw)     

        # ROS visualization marker
        odom_uncertainity = Marker()                            # Creates a new Marker message object
        odom_uncertainity.header.frame_id = "odom"         # Sets the coordinate frame in which this marker is defined
        odom_uncertainity.header.stamp = self.current_time      # Sets the timestamp for when this marker was generated
        odom_uncertainity.ns = "odom_uncertainity"              # Namespace of the marker
        odom_uncertainity.type = Marker.CYLINDER                # Type of shape this marker should display
        odom_uncertainity.action = Marker.ADD                   # "create or update" action this marker in RViz
        odom_uncertainity.pose.position.x = xk[0,0]
        odom_uncertainity.pose.position.y = xk[1,0]
        odom_uncertainity.pose.orientation.x = ellipse_orientation_quat[0]
        odom_uncertainity.pose.orientation.y = ellipse_orientation_quat[1]
        odom_uncertainity.pose.orientation.z = ellipse_orientation_quat[2]
        odom_uncertainity.pose.orientation.w = ellipse_orientation_quat[3]

        # Set the ellipse size (diameter = 2 × standard deviation along each axis)
        odom_uncertainity.scale.x = 2 * math.sqrt(eigenvalues[0])   # Width of the ellipse (minor/major)
        odom_uncertainity.scale.y = 2 * math.sqrt(eigenvalues[1])   # Height of the ellipse (major/minor)
        odom_uncertainity.scale.z = 0.02                            # Thin cylinder (flat ellipse)

        # Set color and transparency of the marker
        odom_uncertainity.color = ColorRGBA(0.0, 1.0, 0.7, 1.0)

        # Set a short lifetime so that the marker updates smoothly each frame
        odom_uncertainity.lifetime = rospy.Duration(0.1)

        # Publish the marker to RVIZ
        self.marker_pub_odom.publish(odom_uncertainity)
        # Publish the odometry data
        self.odom_pub.publish(odom)

        # Publish the transform over tf (transformation frames in ROS)
        '''
            Hey TF, at time t, the robot's base (base_footprint) is at position 
            (x, y, 0.0) with orientation q in the odom frame.
        '''
        self.odom_broadcaster.sendTransform(translation = (xk[0,0], xk[1,0], 0.0),
                                            rotation = self.q,
                                            time = rospy.Time.now(),
                                            child = "odom_deadreckoning",
                                            parent = odom.header.frame_id)
    

    def imu_callback(self, msg):
        self.mutex.acquire()
        # Extract the quaternion from the IMU message
        quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)

        # convert the orientation message recieved from quaternion to euler
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        self.update(yaw)

        self.mutex.release()


    def update(self, yaw_measurement):
        '''
            Kalman filter update
        '''
        # Actual measurement
        self.zk = np.array([yaw_measurement])

        # Jacobian of the observation model w.r.t. state vector
        Hk = np.zeros((1, len(self.xk))).reshape(1, -1)
        Hk[0, -1] = 1
        
        # Jacobian of the observation model with respect to the noise vector
        Vk = np.eye(1)

        # Covariance matrix of the noise vector (IMU noise)
        Rk = np.array([[0.00001]])

        # Expected observation
        self.h = np.array([self.xk[2,0]])

        # Innovation -> Difference between actual and predicted measurements
        innovation = self.wrap_angle(self.zk - self.h)

        # Compute the Kalman gain
        Kk = self.Pk @ Hk.T @ np.linalg.inv(Hk @ self.Pk @ Hk.T + Vk @ Rk @ Vk.T)

        # updating the mean value of the state vector
        self.xk = self.xk + Kk @ (innovation)

        # updating the mean value of the state vector
        I = np.eye((len(self.xk)))
        self.Pk = (I - Kk @ Hk) @ self.Pk @ (I - Kk @ Hk).T

        return self.xk, self.Pk


    def update_path(self):
        '''
            Update the path with the current position of the robot
        '''
        quat = quaternion_from_euler(0, 0, self.xk[2, 0])
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "odom"
        pose.pose.position.x = self.xk[0, 0]
        pose.pose.position.y = self.xk[1, 0]
        pose.pose.orientation = Quaternion(*quat)

        self.path.poses.append(pose)
        self.path.header.stamp = rospy.Time.now()
        self.path_pub.publish(self.path)

        point = Point(pose.pose.position.x, pose.pose.position.y,0.0)
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)
        marker.points.append(point)
        self.path_marker_pub.publish(marker)


if __name__ == "__main__":
    try:
        DeadReckoningNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
