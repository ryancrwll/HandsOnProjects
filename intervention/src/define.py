#!/usr/bin/env python3

import numpy as np
import math
import rospy
from geometry_msgs.msg import PoseStamped
import tf.transformations as tf
from tf.transformations import quaternion_from_euler
import transforms3d.axangles as t_ax

def scale(dq, x, j): # Keep one velocity value fixed and adjust the others to keep the same proportions.

    """
    Scale the elements of dq except the j-th element to a new value x.

    Args:
    dq (numpy.ndarray): The original array of values.
    x (float): The new value to scale to.
    j (int): The index of the element to be scaled.

    Returns:
    numpy.ndarray: The scaled array.
    """
    #  You are in fact fixing it to x, and adjusting the others around that.
    
    mask = np.ones_like(dq, dtype=bool)  # Create a mask of True values with the same shape as dq
    mask[j] = False  # Set the j-th element to False
 
    dq_rest = dq[mask]  # Extract all elements except the j-th element
    for i in range(dq_rest.shape[0]):  # Scale each element
        dq_rest[i] = (x * dq_rest[i]) / dq[j]

    scaled_dq = np.insert(dq_rest, j, x)  # Insert the scaled value x at the j-th position
    return scaled_dq

def W_DLS(A, damping, weight): # This function safely solves the inverse of a robot's Jacobian using damping and joint weights, 
                               # so the robot moves smoothly and follows the task while prioritizing certain joints over others.
                               # Here, # Invert the Jacobian to compute joint velocities needed to follow a desired end-effector motion.


    """
    Compute the weighted damped least-squares (DLS) solution to the matrix inverse problem.

    Args:
    A (numpy.ndarray): Matrix to be inverted.
    damping (float): Damping factor.
    weight (numpy.ndarray): Weights for each element.

    Returns:
    numpy.ndarray: Inversion of the input matrix using DLS.
    """

    '''
    A       → Jacobian matrix (usually: task-to-joint relationship)
    damping → A small number to stabilize the math (e.g., 0.1)
    weight  → A list that says how important each movement is (e.g., [1, 1, 1000, ...])
    '''

    '''
    We use this when:

    * A matrix can’t be inverted normally (e.g. it's too small or unstable).
    * We want to control how much each part of the robot is allowed to move (that's what the weights are for).
    * We want to prevent jerky, unstable movements (that’s what damping is for).
    '''
    w = np.diag(weight)  # Create a diagonal matrix from the weight vector | “Joints 1 and 2 can move freely, but joints 3–6 should move very little.”
    w_i = np.linalg.inv(w)  # Invert the weight matrix | Now the numbers are flipped: * Large weights become small influence. * This lets the robot "prefer" low-weighted joints.

    A_damped = A @ w_i @ A.T + damping**2 * np.eye(A.shape[0])  # Compute the damped matrix | np.eye(A.shape[0]) is an identity matrix (like multiplying by 1). So this whole thing becomes: A smooth, stabilized version of the matrix we're going to invert next.
    A_damped_inv = np.linalg.inv(A_damped)  # Invert the damped matrix
    A_DLS = w_i @ A.T @ A_damped_inv  # Compute the DLS solution | This is your final result: a matrix that can now be used to compute velocity commands that: * Prefer low-weight joint * Avoid instability * Let the robot follow the task accurately
    return A_DLS

def rotation_matrix(axis, angle): # # Return a 4x4 matrix that rotates around the chosen axis (x or z) by the given angle.
    """
    Provide a rotation matrix for a given axis and angle.

    Args:
    axis (str): The axis of rotation ('x' or 'z').
    angle (float): The angle of rotation in radians.

    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the rotation.
    """
    matrix = np.eye(4)  # Initialize a 4x4 identity matrix
    if axis == 'x':
        matrix[0:3, 0:3] = t_ax.axangle2mat([1, 0, 0], angle)  # Apply rotation about the x-axis
    elif axis == 'z':
        matrix[0:3, 0:3] = t_ax.axangle2mat([0, 0, 1], angle)  # Apply rotation about the z-axis
    else:
        raise ValueError("Invalid axis. Must be 'x' or 'z'.")  # Raise an error for invalid axis
    return matrix

def translation_matrix(translation): # Returns a 4x4 matrix that moves a point by [x, y, z] in space.

    """
    Provide a translation matrix for a given translation vector.

    Args:
    translation (list or numpy.ndarray): The translation vector as [tx, ty, tz].

    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the translation.
    """
    if len(translation) != 3:
        raise ValueError("Invalid translation vector. Must have three elements.")  # Check for valid translation vector
    matrix = np.eye(4)  # Initialize a 4x4 identity matrix
    matrix[:3, 3] = translation  # Set the translation vector
    return matrix


def fixed_base_to_link_transform():
    """
    Provide the fixed transformation matrix from 'base_footprint' to 'swiftpro_base_link'.

    Returns:
    numpy.ndarray: The transformation matrix.
    """
    angle = -math.pi / 2  # Define the rotation angle
    return np.array([
        [np.cos(angle), -np.sin(angle), 0, 0.051],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, -0.198], # The arm is on top of the TurtleBot base and in NED coordinates, moving up = negative Z. So we used -ve value meaning it's going up.
        [0, 0, 0, 1]
    ])

def compute_kinematics(joint_angles, base_transform): # It calculates where every part of your robot arm is.
    # Compute and return the full transformation chain from base to end-effector using all joint angles.

    """
    Calculate the kinematics of the robot from base to end-effector.

    Args:
    joint_angles (numpy.ndarray): The joint angles.
    base_transform (numpy.ndarray): The transformation matrix from the world frame to the base_footprint.

    Returns:
    list: A list of transformation matrices for each link with the base link.
    """
    transforms = [base_transform]  # Initialize the transformation list with the base transformation, initially it's the robot's base transformation

    # Transformation from the base_footprint to the swiftpro_baselink
    base_to_link_transform = fixed_base_to_link_transform()  # Define the fixed transformation

    # Transformation of each link with the previous link
    transform_1 = rotation_matrix('z', joint_angles[0, 0]) @ translation_matrix(np.array([0.0132, 0, 0])) @ rotation_matrix('x', -np.pi/2) @ translation_matrix(np.array([0, 0.108, 0]))
    transform_2 = translation_matrix(np.array([-0.142 * np.sin(joint_angles[1, 0]), 0.142 * np.cos(joint_angles[1, 0]), 0]))
    transform_3 = translation_matrix(np.array([0.1588 * np.cos(joint_angles[2, 0]), 0.1588 * np.sin(joint_angles[2, 0]), 0])) @ rotation_matrix('x', np.pi/2) @ translation_matrix(np.array([0.056, 0, 0]))
    transform_4 = rotation_matrix('z', joint_angles[3, 0]) @ translation_matrix(np.array([0, 0, 0.0722]))

    # Append transformations to the list
    link_transforms = [base_to_link_transform, transform_1, transform_2, transform_3, transform_4]
    for transform in link_transforms:
        new_transform = np.dot(transforms[-1], transform)
        transforms.append(new_transform)

    return transforms  # Return the list of transformations

'''
def compute_jacobian(joint_angles, yaw_angle, distance, link_number): # If I move each joint a tiny bit, how will the end-effector (hand) move in space?
    #This function calculates the robot’s Jacobian, showing how each joint and base motion affects the end-effector. 
    # It includes contributions from base translation, base rotation, and all joints up to the desired link.
    """
    Calculate the Jacobian matrix for the robot.

    Args:
    joint_angles (numpy.ndarray): The joint angles.
    yaw_angle (float): The yaw angle.
    distance (float): The distance.
    link_number (int): The link number.

    Returns:
    numpy.ndarray: The Jacobian matrix.
    """

    # The Jacobian has 6 columns, each representing one motion input: [base_yaw, base_translation, joint1, joint2, joint3, joint4]


    J = np.eye(6)  # Initialize a 6x6 identity matrix

    # Define Jacobian components
    component_1 = np.array([[-distance * np.sin(yaw_angle) + (-np.sin(joint_angles[0, 0]) * np.sin(yaw_angle) + np.cos(joint_angles[0, 0]) * np.cos(yaw_angle)) * (0.1588 * np.cos(joint_angles[2, 0]) + 0.056) - 0.142 * (-np.sin(joint_angles[0, 0]) * np.sin(yaw_angle) + np.cos(joint_angles[0, 0]) * np.cos(yaw_angle)) * np.sin(joint_angles[1, 0]) + 0.0132 * np.sin(joint_angles[0, 0]) * np.sin(yaw_angle) - 0.051 * np.sin(yaw_angle) + 0.0132 * np.cos(joint_angles[0, 0]) * np.cos(yaw_angle)],
                        [distance * np.cos(yaw_angle) + (np.sin(joint_angles[0, 0]) * np.cos(yaw_angle) + np.sin(yaw_angle) * np.cos(joint_angles[0, 0])) * (0.1588 * np.cos(joint_angles[2, 0]) + 0.056) - 0.142 * (np.sin(joint_angles[0, 0]) * np.cos(yaw_angle) + np.sin(yaw_angle) * np.cos(joint_angles[0, 0])) * np.sin(joint_angles[1, 0]) + 0.0132 * np.sin(joint_angles[0, 0]) * np.cos(yaw_angle) + 0.0132 * np.sin(yaw_angle) * np.cos(joint_angles[0, 0]) + 0.051 * np.cos(yaw_angle)],
                        [0],
                        [0],
                        [0],
                        [1]])
    component_2 = np.array([[math.cos(yaw_angle)],
                       [math.sin(yaw_angle)],
                       [0],
                       [0],
                       [0],
                       [0]])
    component_3 = np.array([[(-np.sin(joint_angles[0, 0]) * np.sin(yaw_angle) + np.cos(joint_angles[0, 0]) * np.cos(yaw_angle)) * (0.1588 * np.cos(joint_angles[2, 0]) + 0.056) - 0.142 * (-np.sin(joint_angles[0, 0]) * np.sin(yaw_angle) + np.cos(joint_angles[0, 0]) * np.cos(yaw_angle)) * np.sin(joint_angles[1, 0]) - 0.0132 * np.sin(joint_angles[0, 0]) * np.sin(yaw_angle) + 0.0132 * np.cos(joint_angles[0, 0]) * np.cos(yaw_angle)],
                        [(np.sin(joint_angles[0, 0]) * np.cos(yaw_angle) + np.sin(yaw_angle) * np.cos(joint_angles[0, 0])) * (0.1588 * np.cos(joint_angles[2, 0]) + 0.056) - 0.142 * (np.sin(joint_angles[0, 0]) * np.cos(yaw_angle) + np.sin(yaw_angle) * np.cos(joint_angles[0, 0])) * np.sin(joint_angles[1, 0]) + 0.0132 * np.sin(joint_angles[0, 0]) * np.cos(yaw_angle) + 0.0132 * np.sin(yaw_angle) * np.cos(joint_angles[0, 0])],
                        [0],
                        [0],
                        [0],
                        [1]])
    component_4 = np.array([[-0.142 * (np.sin(joint_angles[0, 0]) * np.cos(yaw_angle) + np.sin(yaw_angle) * np.cos(joint_angles[0, 0])) * np.cos(joint_angles[1, 0])],
                       [-0.142 * (np.sin(joint_angles[0, 0]) * np.sin(yaw_angle) - np.cos(joint_angles[0, 0]) * np.cos(yaw_angle)) * np.cos(joint_angles[1, 0])],
                       [0.142 * math.sin(joint_angles[1, 0])],
                       [0],
                       [0],
                       [0]])
    component_5 = np.array([[-0.1588 * (np.sin(joint_angles[0, 0]) * np.cos(yaw_angle) + np.sin(yaw_angle) * np.cos(joint_angles[0, 0])) * np.sin(joint_angles[2, 0])],
                       [-0.1588 * (np.sin(joint_angles[0, 0]) * np.sin(yaw_angle) - np.cos(joint_angles[0, 0]) * np.cos(yaw_angle)) * np.sin(joint_angles[2, 0])],
                       [-0.1588 * math.cos(joint_angles[2, 0])],
                       [0],
                       [0],
                       [0]])
    component_6 = np.array([[0],
                       [0],
                       [0],
                       [0],
                       [0],
                       [1]])

    # Reshape Jacobian components
    J1 = component_1.reshape(6, 1)
    J2 = component_2.reshape(6, 1)
    J3 = component_3.reshape(6, 1)
    J4 = component_4.reshape(6, 1)
    J5 = component_5.reshape(6, 1)
    J6 = component_6.reshape(6, 1)

    # Stack Jacobian components horizontally
    J = np.hstack((J1, J2, J3, J4, J5, J6))

    # Zero out the Jacobian columns after the specified link
    J[:, link_number:] = 0
    return J
    '''

### Faran's Jacobbian

def compute_jacobian(joint_angles, yaw_angle, distance, link_number):
    """
    Computes the 6x6 Jacobian matrix for a mobile manipulator.
    Inputs:
        joint_angles: 4x1 numpy array [theta1, theta2, theta3, theta4]
        yaw_angle: base yaw (psi)
        distance: base translation offset (from origin to base)
        link_number: how many columns of the Jacobian to keep active
    Output:
        6x6 Jacobian matrix
    """
    theta1 = joint_angles[0, 0]
    theta2 = joint_angles[1, 0]
    theta3 = joint_angles[2, 0]
    theta4 = joint_angles[3, 0]

    # Constants (robot-specific link parameters)
    a2 = 0.142     # link 2 length
    d2 = 0.056     # link 2 offset
    a3 = 0.1588    # link 3 length
    d4 = 0.0132    # wrist offset
    ee_offset = 0.051  # end-effector (gripper) offset

    cos1 = np.cos(theta1 + yaw_angle)
    sin1 = np.sin(theta1 + yaw_angle)
    cos2 = np.cos(theta2)
    sin2 = np.sin(theta2)
    cos3 = np.cos(theta3)
    sin3 = np.sin(theta3)
    psi = yaw_angle

    # Compute forward offset to EE
    d = a3 * cos3 + d2 * sin2 + ee_offset + d4

    # Base yaw contribution (J1)
    J1 = np.array([
        [ d * cos1 - distance * np.sin(psi)],
        [ d * sin1 + distance * np.cos(psi)],
        [0],
        [0],
        [0],
        [1]
    ])

    # Base forward contribution (J2)
    J2 = np.array([
        [np.cos(psi)],
        [np.sin(psi)],
        [0],
        [0],
        [0],
        [0]
    ])

    # Joint linear velocity part (Jv)
    Jv = np.array([
        [ cos1 * d, -d2 * sin1 * cos2, -a3 * sin3 * sin1, 0],
        [ sin1 * d,  d2 * cos1 * cos2,  a3 * sin3 * cos1, 0],
        [       0 ,      d2 * sin2,     -a3 * cos3,      0]
    ])

    # Joint angular velocity part (Jw)
    Jw = np.array([
        [1, 0, 0, 0],   # theta1 contributes to z-axis rotation
        [0, 0, 0, 0],   # theta2 (pitch) has no z-axis angular velocity here
        [0, 0, 0, 1]    # theta4 contributes to wrist rotation
    ])

    # Combine full Jacobian
    J_top = np.hstack([J1[:3], J2[:3], Jv])      # 3x6
    J_bottom = np.hstack([J1[3:], J2[3:], Jw])   # 3x6
    J = np.vstack([J_top, J_bottom])             # 6x6

    # Zero out columns beyond the active link
    J[:, link_number:] = 0

    return J

def pose_EE(transform_matrix): # Convert a 4x4 transformation matrix into a ROS PoseStamped message (for publishing position + orientation).

    """
    Convert a transformation matrix to a PoseStamped message.

    Args:
    transform_matrix (numpy.ndarray): The transformation matrix.

    Returns:
    PoseStamped: The pose in ROS message format.
    """
    translation = transform_matrix[:3, -1]  # Extract translation
    quaternion = tf.quaternion_from_matrix(transform_matrix)  # Extract rotation as quaternion

    pose_msg = PoseStamped()
    pose_msg.header.frame_id = "world_ned"  # Set the frame ID
    pose_msg.header.stamp = rospy.Time.now()  # Set the timestamp

    # Set the position
    pose_msg.pose.position.x = translation[0]
    pose_msg.pose.position.y = translation[1]
    pose_msg.pose.position.z = translation[2]

    # Set the orientation
    pose_msg.pose.orientation.x = quaternion[0]
    pose_msg.pose.orientation.y = quaternion[1]
    pose_msg.pose.orientation.z = quaternion[2]
    pose_msg.pose.orientation.w = quaternion[3]

    return pose_msg

def goal_pose(translation, rotation_angle):

    # Create a PoseStamped message for a goal position and facing direction (yaw), using quaternion orientation.

    # pose = goal_pose([0.6, -0.2, -0.3], math.pi/2), You’re telling the robot: “Go to position (0.6, -0.2, -0.3) and face 90° to the left.”
    """
    Publish the goal pose given a translation and a rotation about the z-axis.

    Args:
    translation (list): The translation vector [tx, ty, tz].
    rotation_angle (float): The rotation angle about the z-axis in radians.

    Returns:
    PoseStamped: The goal pose in ROS message format.
    """
    quaternion = quaternion_from_euler(0, 0, rotation_angle)  # Convert euler angles to quaternion, on the rotation for z axis
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = "odom"  # Set the frame ID
    goal_pose.header.stamp = rospy.Time.now()
    goal_pose.pose.position.x, goal_pose.pose.position.y, goal_pose.pose.position.z = translation  # Set the position
    goal_pose.pose.orientation.x, goal_pose.pose.orientation.y, goal_pose.pose.orientation.z, goal_pose.pose.orientation.w = quaternion  # Set the orientation
    return goal_pose


class Manipulator: # The Manipulator class gives your robot the ability to understand and control 
            #its own body in the world — it's the foundation for all smart movement in the project.
    def __init__(self, joint_angles):
        """
        Initialize the manipulator with given joint angles.

        Args:
        joint_angles (numpy.ndarray): Initial joint angles.
        """
        self.joint_angles = joint_angles
        self.revolute = [True, False, True, True, True, True]
        self.dof = len(self.revolute)
        self.base_pose = np.zeros((3, 1))
        self.transformations = np.zeros((4, 4))

    def update(self, dq, dt, base_state):
        """
        Update the state of the robot.

        Args:
        dq (numpy.ndarray): A column vector of joint velocities.
        dt (float): Sampling time.
        base_state (numpy.ndarray): Current state of the base [x, y, yaw].
        """
        self.joint_angles += (dq[2:, 0]).reshape(-1, 1) * dt

        self.base_pose[2, 0] = base_state[2]
        self.base_pose[0, 0] = base_state[0]
        self.base_pose[1, 0] = base_state[1]

        Tb = np.array([[math.cos(self.base_pose[2, 0]), -math.sin(self.base_pose[2, 0]), 0, self.base_pose[0, 0]],
                       [math.sin(self.base_pose[2, 0]), math.cos(self.base_pose[2, 0]), 0, self.base_pose[1, 0]],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        self.transformations = compute_kinematics(self.joint_angles, Tb)

    def getEEJacobian(self, link):
        """
        Get the Jacobian matrix for the end-effector.

        Args:
        link (int): The link number.

        Returns:
        numpy.ndarray: The Jacobian matrix.
        """
        return compute_jacobian(self.joint_angles, self.base_pose[2, 0], self.base_pose[0, 0], link)

    def getEETransform(self):
        """
        Get the transformation matrix for the end-effector.

        Returns:
        numpy.ndarray: The transformation matrix.
        """
        return self.transformations[-1]

    def getJointPos(self, joint):
        """
        Get the position of a selected joint.

        Args:
        joint (int): Index of the joint.

        Returns:
        float: Position of the joint.
        """
        return self.joint_angles[joint - 3]

    def getBasePose(self):
        """
        Get the pose of the base.

        Returns:
        numpy.ndarray: The pose of the base.
        """
        return self.base_pose

    def getDOF(self):
        """
        Get the degrees of freedom of the manipulator.

        Returns:
        int: Degrees of freedom.
        """
        return self.dof

    def get_Se_LTransform(self, link):
        """
        Get the transformation of a selected link.

        Args:
        link (int): The link number.

        Returns:
        numpy.ndarray: The transformation matrix of the link.
        """
        return self.transformations[link - 1]

class Task:
    '''The Task class is like a blueprint for any goal you want the robot to achieve. 
    It defines the structure of a task — like what the goal is, whether the task is active, 
    and how to calculate how far the robot is from completing it. Other specific tasks like 
    moving the hand or avoiding limits build on top of this base class.
    '''
    def __init__(self, name, desired):
        """
        Constructor.

        Args:
        name (str): Title of the task.
        desired (numpy.ndarray): Desired sigma (goal).
        """
        self.name = name
        self.sigma_d = desired
        self.mobi_base = None
        self.active = False
        self.a = 0

    def bool_is_Active(self):
        """
        Check if the task is active.

        Returns:
        bool: True if active, False otherwise.
        """
        return self.active

    def update(self, robot):
        """
        Update the task variables.

        Args:
        robot (Manipulator): Reference to the manipulator.
        """
        pass

    def setDesired(self, value):
        """
        Set the desired sigma.

        Args:
        value (numpy.ndarray): Value of the desired sigma (goal).
        """
        self.sigma_d = value

    def getDesired(self):
        """
        Get the desired sigma.

        Returns:
        numpy.ndarray: The desired sigma.
        """
        return self.sigma_d

    def getJacobian(self):
        """
        Get the task Jacobian.

        Returns:
        numpy.ndarray: The Jacobian matrix.
        """
        return self.J

    def getError(self):
        """
        Get the task error (tilde sigma).

        Returns:
        numpy.ndarray: The error.
        """
        return self.err

    def get_mobi_base(self):
        """
        Get the mobile base position.

        Returns:
        numpy.ndarray: The mobile base position.
        """
        return self.mobi_base

    def get_eep(self):
        """
        Get the end-effector position.

        Returns:
        numpy.ndarray: The end-effector position.
        """
        return self.eep

class Position3D(Task):

    # Task to move the robot’s end-effector to a 3D position by computing the position error and related Jacobian.

    # This is a specific task that tells the robot: “I want the gripper to be at this 3D position: (x, y, z). 
    # Tell me how far away I am, and how I can move the joints or base to reach it.
    def __init__(self, name, desired, link):
        """
        Initialize the Position3D task.

        Args:
        name (str): Title of the task.
        desired (numpy.ndarray): Desired sigma (goal).
        link (int): The link number.
        """
        super().__init__(name, desired)
        self.link = link
        self.J = np.zeros((3, self.link))
        self.err = np.zeros((3, 1))
        self.active = True

    def update(self, robot):
        """
        Update the task variables.

        Args:
        robot (Manipulator): Reference to the manipulator.
        """
        self.J = robot.getEEJacobian(self.link)[0:3]  # Update task Jacobian
        k = robot.getEETransform()[0:3, 3].reshape(3, 1)  # Get end-effector position
        self.err = self.getDesired().reshape(3, 1) - k  # Update task error

class Jointlimits3D(Task):
    #This task helps the robot avoid going past joint limits by activating a constraint 
    # whenever a joint gets too close to its limit.

    # Task to prevent joints from exceeding their safe angular limits.
    # Activates only when a joint nears a limit, forcing the controller to push it back.

    def __init__(self, name, desired, activation, link):
        """
        Initialize the Jointlimits3D task.

        Args:
        name (str): Title of the task.
        desired (numpy.ndarray): Desired sigma (goal).
        activation (numpy.ndarray): Activation limits.
        link (int): The link number.
        """
        super().__init__(name, desired)
        self.activation = activation
        self.link = link
        self.J = np.zeros((1, 6))
        self.a = 0
        self.err = np.zeros((1))

    def wrap_angle(self, angle):
        """
        Wrap the angle between -pi and +pi.

        Args:
        angle (float): The angle to be wrapped.

        Returns:
        float: The wrapped angle.
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def update(self, robot):
        """
        Update the task variables.

        Args:
        robot (Manipulator): Reference to the manipulator.
        """
        self.J = robot.getEEJacobian(self.link)[5, :].reshape(1, 6)  # Update task Jacobian

        link_transform = robot.get_Se_LTransform(self.link)  # Get link transformation

        orien = np.arctan2(link_transform[1, 2], link_transform[0, 2])  # Calculate orientation

        if self.a == 1 and orien > self.activation[2]:
            self.a = 0
            self.active = False
            self.err = 0.0

        if self.a == -1 and orien < self.activation[0]:
            print('activated')
            self.a = 0
            self.active = False
            self.err = 0.0

        if self.a == 0 and orien > self.activation[1]:
            print('un-activated')
            self.a = -1
            self.active = True
            self.err = -1.0

        if self.a == 0 and orien < self.activation[3]:
            self.a = 1
            self.active = True
            self.err = 1.0
