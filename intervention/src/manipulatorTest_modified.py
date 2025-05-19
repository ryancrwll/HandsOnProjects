#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import math
from scipy.spatial.transform import Rotation as R

class Manipulator():

    def __init__(self, theta1, theta2, theta3, theta4):
        # Initialize the class with joint angles
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.theta4 = theta4

        # Joint configuration vector (used for kinematics and control)
        self.q = [[theta1], [theta2], [theta3], [theta4]]

        # Define geometric constants (in meters)
        self.a1 = 0.0132   # base offset
        self.a2 = 0.1588   # link 2 length
        self.a3 = 0.056    # wrist length

        self.d1 = 0.108   
        self.d2 = 0.142    # vertical offset to joint 2
        self.d3 = 0.0722   

        # Total horizontal offset from base to wrist (a1 + a3)
        self.a13 = self.a1 + self.a3  

        # Default goal position
        self.goal_position = np.array([[0.11], [0.11], [-0.03]])

        # Damping coefficient for DLS method
        self.damping = 0.1

        # Subscribe to joint state updates from the robot
        self.joints_states_sub = rospy.Subscriber("/swiftpro/joint_states", JointState, self.joint_state_callback)

        # Subcribe to the input live goal
        self.goal_sub = rospy.Subscriber("/desired_goal", PoseStamped, self.update_goal)

        # Publisher to send joint velocity commands to the robot

        self.goal_position = np.array([[0.11], [0.11], [-0.03]])
        self.goal_sub = rospy.Subscriber("/desired_goal", PoseStamped, self.update_goal)
        self.goal_pub = rospy.Publisher("/goal", PoseStamped, queue_size=10)

        self.joint_velocity_pub = rospy.Publisher("/swiftpro/joint_velocity_controller/command", Float64MultiArray, queue_size=10)
        
        # Publisher to broadcast the current pose of the end-effector (EE)
        self.pose_EE_pub = rospy.Publisher("/pose_EE", PoseStamped, queue_size=10)

        # Publisher to broadcast the target/goal pose
        self.goal_pub = rospy.Publisher("/goal", PoseStamped, queue_size=10)

        # Publisher to broadcast the end-effector trajectory
        self.ee_trajectory_pub = rospy.Publisher("/ee_trajectory_marker", Marker, queue_size=10)

        # Initialize a list to store EE trajectory points
        self.trajectory_points = []


    def update_goal(self, msg):
        self.goal_position = np.array([[msg.pose.position.x],
                                       [msg.pose.position.y],
                                       [msg.pose.position.z]])
        
        rospy.loginfo(f"[Goal updated] New Goal: {self.goal_position.flatten()}")
    

    
    def update_goal(self, msg):
        self.goal_position = np.array([[msg.pose.position.x],
                                       [msg.pose.position.y],
                                       [msg.pose.position.z]])
        rospy.loginfo(f"[Goal updated] New Goal: {self.goal_position.flatten()}")
        
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "swiftpro/manipulator_base_link"
        pose_goal.header.stamp = rospy.Time.now()
        pose_goal.pose.position.x = self.goal_position[0, 0]
        pose_goal.pose.position.y = self.goal_position[1, 0]
        pose_goal.pose.position.z = self.goal_position[2, 0]
        pose_goal.pose.orientation.w = 1.0
        self.goal_pub.publish(pose_goal)


    def joint_state_callback(self, data):
        '''
            Callback function that updates internal joint state when a new message is received
        '''
        if data.name == ['swiftpro/joint1', 'swiftpro/joint2', 'swiftpro/joint3', 'swiftpro/joint4']:
            # Update internal joint angle variables from the received joint state message
            self.theta1, self.theta2, self.theta3, self.theta4 = data.position
            
            # Update the joint configuration vector
            self.q = [[self.theta1], [self.theta2], [self.theta3], [self.theta4]]

            # Recalculate the end-effector pose and run the control loop
            self.pose_EE()
            self.run()


    def pose_EE(self):
        '''
            Compute and publish the end-effector pose using transformation matrix and quaternion conversion.
        '''
        rot = R.from_matrix(self.kinematics()[:3, :3])  # Extract 3x3 rotation matrix
        quat = R.as_quat(rot)                           # Convert to quaternion [x, y, z, w]

        # Create a PoseStamped message to publish the end-effector pose
        p = PoseStamped()
        p.header.frame_id = "swiftpro/manipulator_base_link"
        p.header.stamp = rospy.Time.now()

        # Set the position component from the transformation matrix
        p.pose.position.x = self.kinematics()[0, 3]
        p.pose.position.y = self.kinematics()[1, 3]
        p.pose.position.z = self.kinematics()[2, 3]

        # Set the orientation component using quaternion
        p.pose.orientation.x = quat[0]
        p.pose.orientation.y = quat[1]
        p.pose.orientation.z = quat[2]
        p.pose.orientation.w = quat[3]

        # Publish the pose message
        self.pose_EE_pub.publish(p)


    def kinematics(self):
        '''
            Compute forward kinematics: homogeneous transformation matrix from base to end-effector.
            (This matrix includes rotation and translation components and is derived manually)
        '''
        a1 = self.a1    # Offset from the base to the second joint along x-axis
        a2 = self.a2    # Length of the second link
        a3 = self.a3    # Length of the wrist link (from third joint to end-effector)

        d2 = self.d2    # Offset along the z-axis from the base to the second joint

        # Total horizontal offset from base to wrist (a1 + a3)
        a13 = self.a13

        # Total orientation from final joint
        theta = self.theta1 + self.theta4

        # Compute the homogeneous transformation matrix Hm_ee from base to end-effector
        x = (a13 - d2 * np.sin(self.theta2) + a2 * np.cos(self.theta3)) * np.cos(self.theta1)
        y = (a13 - d2 * np.sin(self.theta2) + a2 * np.cos(self.theta3)) * np.sin(self.theta1)
        z = -0.0358 - d2 * np.cos(self.theta2) - a2 * np.sin(self.theta3)

        self.Hm_ee = np.array([
                                [np.cos(theta), -np.sin(theta), 0, x],
                                [np.sin(theta),  np.cos(theta), 0, y],
                                [   0         ,         0     , 1, z],
                                [   0         ,         0     , 1, 1]
                            ])
        
        # Return the transformation matrix from base to end-effector
        return self.Hm_ee
    

    def Jacobian(self):
        '''
            Compute the 6x4 Jacobian matrix (3 linear rows + 3 angular rows) for the current joint configuration.
            (calculated manually using the above homogeneous transformation matrix)
        '''
        a1 = self.a1        # base offset (horizontal)
        a2 = self.a2        # length of second link
        a3 = self.a3        # wrist link
        d1 = self.d1        # (not used directly here)
        d2 = self.d2        # vertical offset link
        d3 = self.d3        # (not used directly here)
        a13 = self.a13      # total wrist offset

        self.J = np.array([ 
                            [-np.sin(self.theta1) * (a13 - d2 * np.sin(self.theta2) + a2 * np.cos(self.theta3)), -d2 * np.cos(self.theta1) * np.cos(self.theta2),  a2 * np.sin(self.theta3) * np.cos(self.theta1), 0],
                            [ np.cos(self.theta1) * (a13 - d2 * np.sin(self.theta2) + a2 * np.cos(self.theta3)), -d2 * np.sin(self.theta1) * np.cos(self.theta2), -a2 * np.sin(self.theta3) * np.sin(self.theta1), 0],
                            [               0                                                                  ,  d2 * np.sin(self.theta2)                      , -a2 * np.cos(self.theta3)                      , 0],
                            [               0                                                                  ,                 0                              ,                 0                              , 0 ],
                            [               0                                                                  ,                 0                              ,                 0                              , 0 ],
                            [               1                                                                  ,                 0                              ,                 0                              , 1 ]
                          ])
        
        return self.J
    

    def DLS(self, A, damping):
        '''
            Damped Least Squares method for computing the pseudo-inverse of a matrix A.
            Used to solve ill-conditioned inverse kinematics near singularities.

            Args:
                A (np.ndarray): The Jacobian or any matrix to pseudo-invert.
                damping (float): Damping factor to improve numerical stability.

            Returns:
                np.ndarray: The Damped Least Squares pseudo-inverse of A.
        '''
        x = A @ np.transpose(A)
        I = np.identity(x.shape[0])     # Identity matrix
        
        # Compute the damped pseudo-inverse
        DLS = np.transpose(A) @ np.linalg.inv(x + damping**2 * I)
        return DLS
    

    def send_velocity(self, velocities):
        '''
            Publishes joint velocity commands to the robot controller.
        '''
        # Publish joint velocities
        vel = Float64MultiArray()

        # Convert each joint velocity to float and store in the message
        vel.data = [
                    float(velocities[0]), 
                    float(velocities[1]), 
                    float(velocities[2]), 
                    float(velocities[3])
                   ]
        
        # Publish the velocity command to the joint velocity controller
        self.joint_velocity_pub.publish(vel)


    def run(self):
        '''
            Executes one step of resolved-rate motion control using the Damped Least Squares (DLS) method.
            Moves the end-effector towards a fixed goal position in Cartesian space.
        '''
        # Step 1: Get current transformation and Jacobian
        T = self.kinematics()       # 4x4 Homogeneous transform from base to end-effector
        J_full = self.Jacobian()    # 6x4 Jacobian (linear + angular parts)

        # Step 2: Extract only the linear (position) part of the Jacobian (top 3 rows)
        J_pos = J_full[:3, :]

        # Step 3: Extract the current end-effector position (translation part of T)
        pos_current = T[:3, 3].reshape(3, 1)

        # Step 4: Compute position error between goal and current EE position
        error = self.goal_position - pos_current 

        # Log error between the goal the current position of the end-effector only when the 
        # end effector is trying to reach the goal
        error_threshold = 0.005     # Threshold to consider how close the end-effector is to goal (in meters)
        error_norm = np.linalg.norm(error)
        if error_norm > error_threshold:
            # Log just once per second
            rospy.loginfo_throttle(1, f"Position Error: {error.flatten()}, Norm: {error_norm:.4f}")

        # Step 5: Compute joint velocities using DLS inverse
        dq = self.DLS(J_pos, self.damping) @ error

        
        # Clip joint velocities (limit: ±0.1 rad/s)
        dq = np.clip(dq, -0.1, 0.1)

# Step 6: Enforce joint velocity limit (normalize if any velocity exceeds max)
        max_joint_velocity = 0.1  # Max joint velocity in rad/s
        scaling_factor = np.max(np.abs(dq) / max_joint_velocity)
        if scaling_factor > 1:
            dq = dq / scaling_factor

        # Step 7: Integrate velocities to update joint angles
        time_step = 1.0 / 60.0
        self.q += time_step * dq

        # Send joint velocity command to robot
        self.send_velocity(dq)

        # Publish the goal position
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "swiftpro/manipulator_base_link"
        pose_goal.header.stamp = rospy.Time.now()
        pose_goal.pose.position.x = self.goal_position[0, 0]
        pose_goal.pose.position.y = self.goal_position[1, 0]
        pose_goal.pose.position.z = self.goal_position[2, 0]
        self.goal_pub.publish(pose_goal)

        # Store trajectory points for Rviz visulaization
        pt = Point()
        pt.x = pos_current[0, 0]
        pt.y = pos_current[1, 0]
        pt.z = pos_current[2, 0]

        self.trajectory_points.append(pt)
        self.publish_trajectory_marker()


    def publish_trajectory_marker(self):
        marker = Marker()
        marker.header.frame_id = "swiftpro/manipulator_base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "ee_trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale = Vector3()
        marker.scale.x = 0.005
        marker.scale.y = 0.0
        marker.scale.z = 0.0

        marker.pose.orientation.w = 1.0  # Identity quaternion for orientation

        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.points = self.trajectory_points
        
        self.ee_trajectory_pub.publish(marker)


if __name__ == "__main__":
    try:
        # Initialize ROS node
        rospy.init_node("kinematics_node", anonymous=True)

        # Create an instance of the Manipulator class with initial joint angles
        manipulator = Manipulator(0, 0, 0, 0)

        rospy.spin()

    except rospy.ROSInterruptException:
        pass

