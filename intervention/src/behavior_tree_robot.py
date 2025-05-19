#!/usr/bin/env python3

from py_trees.behaviour import Behaviour
from py_trees.composites import Sequence
import traceback
import py_trees  # Import the py_trees library for behavior trees
import rospy  # Import the rospy library for ROS (Robot Operating System) interactions
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseStamped, Point, Pose, Vector3  # Import the PoseStamped message type from geometry_msgs
import py_trees.decorators  # Import decorators from py_trees
import py_trees.display  # Import display functions from py_trees
import time  # Import the time library for time-related functions
from nav_msgs.msg import Odometry  # Import the Odometry message type from nav_msgs
from std_msgs.msg import Float64MultiArray  # Import the Float64MultiArray message type from std_msgs
import math  # Import the math library for mathematical functions
from hands_on_intervention.srv import intervention  # Import the intervention service from hands_on_intervention

from tf.transformations import euler_from_quaternion, quaternion_from_euler  # Import the euler_from_quaternion function from tf.transformations
import numpy as np  # Import the numpy library for numerical operations
from std_srvs.srv import SetBool  # Import the SetBool service from std_srvs
import signal  # Import the signal library for signal handling


# Class to move the robot to a pick location
class MoveRobotToPick(py_trees.behaviour.Behaviour):

    def __init__(self, name):
        super(MoveRobotToPick, self).__init__(name)
        # Subscribe to the pose of the end-effector
        self.sub_pose_ee = rospy.Subscriber('pose_EE', PoseStamped, self.ee_pose_callback)
        self.pose_flag = False
        # Subscribe to the position of Aruco markers
        self.image_sub = rospy.Subscriber("/aruco_position", Float64MultiArray, self.aruco_position_callback)
        # Subscribe to the odometry information
        self.sub_odom = rospy.Subscriber("/turtlebot/kobuki/odom_ground_truth", Odometry, self.odom_callback)
        # Dimensions of box to pick (width, depth, height)
        self.dims = np.array([.07, .07, .15])
        time.sleep(2)  # Sleep to ensure the subscriptions are set up

    def setup(self):
        self.logger.debug("  %s [MoveRobotToPick::setup()]" % self.name)
        try:
            # Wait for the services to be available, with timeout
            rospy.wait_for_service('goal_server')
            rospy.wait_for_service('weight_server')
            rospy.wait_for_service('aruco_server')
            rospy.wait_for_service('task_server')

            # Create service proxies
            self.set_aruco = rospy.ServiceProxy('aruco_server', intervention)
            self.set_goal = rospy.ServiceProxy('goal_server', intervention)
            self.set_weight = rospy.ServiceProxy('weight_server', intervention)
            self.set_task = rospy.ServiceProxy('task_server', intervention)

            self.logger.debug("  %s [MoveRobotToPick::setup() Server connected!]" % self.name)
            return True
        except rospy.ROSException as e:
            rospy.logerr("  %s [MoveRobotToPick::setup() failed: %s]" % (self.name, e))
            return False  # Return Status.FAILURE if timeout/error

    def initialise(self):
        self.ee_pose = None
        self.isPose = False
        self.goal_xyz = 0
        self.aruco_pose = 0
        # Define weights for the robot's movements
        self.weight = [1.0, 1.0, 1000.0, 1000.0, 1000.0, 1000.0]
        self.weight_arm_pose = [100000.0, 100000.0, 1.0, 1.0, 1.0, 1.0]
        self.logger.debug("  %s [MoveRobotToPick::initialise()]" % self.name)

    def terminate(self, new_status):
        self.logger.debug("  %s [MoveRobotToPick::terminate().terminate()][%s->%s]" %
                          (self.name, self.status, new_status))

    def ee_pose_callback(self, data):
        self.ee_pose = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.isPose = True

    def aruco_position_callback(self, aruco_msg):
        self.aruco_pose = aruco_msg.data
        blackboard.aruco = self.aruco_pose
        print('Aruco_position', self.aruco_pose)
        

    def odom_callback(self, odom_data):
        self.dt = odom_data.twist.twist.linear.x
        quaternion = (odom_data.pose.pose.orientation.x, odom_data.pose.pose.orientation.y, odom_data.pose.pose.orientation.z, odom_data.pose.pose.orientation.w)
        euler = euler_from_quaternion(quaternion)
        self.robot_state = np.array([odom_data.pose.pose.position.x, odom_data.pose.pose.position.y])
        self.front_point = [self.robot_state[0] + 0.3, self.robot_state[1] - 0.01, -0.3, 0.0]
        blackboard.front_point = self.front_point

    def signal_handler(self, signal, frame):
        print("Loop stopped by user")
        exit(0)

    def update(self):
        try:
            # Set up signal handler for interrupting the process
            # signal.signal(signal.SIGINT, self.signal_handler)
            for i in range(3):
                self.logger.debug("  {}: Publishing goal position".format(self.name))
                if i == 0:
                    rospy.logerr('1st iteration: turning so arm faces box')
                    # Call the Aruco service to get the position
                    response = self.set_aruco()
                    time.sleep(0.5)
                    # Set the weight for the arm
                    response = self.set_weight(self.weight_arm_pose)
                    #response = self.set_task([0])    #### 
                    time.sleep(0.2)
                    threshold = 0.08
                    # Update the goal position from the blackboard
                    goal_position = blackboard.front_point
                    response = self.set_goal(goal_position)
                    #response = self.set_task([0]) ###
                    self.goal_xyz = goal_position[0:3]
                    # Calculate the distance between the goal and the current end-effector position
                    while not self.isPose:
                        time.sleep(.1)
                    distance = np.linalg.norm(self.goal_xyz - self.ee_pose)
                    while distance > threshold:
                        try:
                            distance = np.linalg.norm(self.goal_xyz - self.ee_pose)
                            time.sleep(0.5)
                        except KeyboardInterrupt:
                            print('Loop stopped by user')
                            break
                elif i==1:
                    rospy.logerr('2nd iteration: sets weights to only move with base')
                    # Set the weight for the robot
                    response = self.set_weight(self.weight)
                    time.sleep(0.2)
                    
                    # Update the goal position from the Aruco marker
                    goal_position = blackboard.aruco
                    #response = self.set_task([0]) ###
                    time.sleep(1)
                    self.goal_xyz = np.array([goal_position[0], goal_position[1], goal_position[2]-0.3, 0.0])

                    response = self.set_goal(self.goal_xyz)
                    threshold = 0.32
                    distance = np.linalg.norm(self.goal_xyz[:2] - self.robot_state[:2])
                    while distance > threshold:
                        try:
                            distance = np.linalg.norm(self.goal_xyz[:2] - self.robot_state[:2])
                          
                        except KeyboardInterrupt:
                            print('Loop stopped by user')
                            break
                else:
                    rospy.logerr('3rd iteration: Once in distance only uses arm again')
                    response = self.set_weight(self.weight_arm_pose)
                    time.sleep(0.2)

                    threshold = 0.002
                    approach_point = np.array([self.goal_xyz[0], self.goal_xyz[1], self.goal_xyz[2]])
                    distance = np.linalg.norm(approach_point - self.ee_pose)
                    while distance > threshold:
                        try:
                            distance = np.linalg.norm(approach_point - self.ee_pose)
                        except KeyboardInterrupt:
                            print('Loop stopped by user')
                            break

            rospy.logwarn('Goal reached')
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            self.logger.debug(f"Error in MoveRobotToPick block: {str(e)}\n{traceback.format_exc()}")
           
            return py_trees.common.Status.FAILURE





# Class to move the robot to a place location
class MoveRobotToPlace(py_trees.behaviour.Behaviour):

    def __init__(self, name):
        super(MoveRobotToPlace, self).__init__(name)
        # Subscribe to the pose of the end-effector
        self.sub_pose_ee = rospy.Subscriber('pose_EE', PoseStamped, self.ee_pose_callback)
        # Subscribe to the position of Aruco markers
        self.image_sub = rospy.Subscriber("/aruco_position", Float64MultiArray, self.aruco_position_callback)
        # Subscribe to the odometry information
        self.sub_odom = rospy.Subscriber("/turtlebot/kobuki/odom_ground_truth", Odometry, self.odom_callback)
        time.sleep(1)

    def setup(self):
        self.logger.debug("  %s [MoveRobotToPlace::setup()]" % self.name)
        try:
            # Wait for the services to be available
            rospy.wait_for_service('goal_server')
            rospy.wait_for_service('weight_server')
            rospy.wait_for_service('aruco_server')
            rospy.wait_for_service('task_server')

            # Create service proxies
            self.set_aruco = rospy.ServiceProxy('aruco_server', intervention)
            self.set_goal = rospy.ServiceProxy('goal_server', intervention)
            self.set_weight = rospy.ServiceProxy('weight_server', intervention)
            self.set_task = rospy.ServiceProxy('task_server', intervention)
            self.logger.debug("  %s [MoveRobotToPlace::setup() Server connected!]" % self.name)

            return True  # MUST return py_trees status
        except Exception as e:
            self.logger.debug(f"Error in MoveRobotToPick block: {str(e)}\n{traceback.format_exc()}")
           
            return py_trees.common.Status.FAILURE

    def initialise(self):
    
        self.goal_xyz = 0
        self.distance = 0
        # Define weights for the robot's movements
        self.weight = [1.0, 1.0, 1000.0, 1000.0, 1000.0, 1000.0]
        self.logger.debug("  %s [MoveRobotToPlace::initialise()]" % self.name)


    def update(self):
        try:
            # Set up signal handler for interrupting the process
            # signal.signal(signal.SIGINT, self.signal_handler)
            self.logger.debug("  %s [MoveRobotToPlace::update()]" % self.name)
            # Set the weight for the robot
            response = self.set_weight(self.weight)
            time.sleep(0.2)
        
            rospy.loginfo("Weight set successfully")
            # Define the goal position
            goal_position = [self.robot_state[0] + 1.5, -0.01, -0.36, 0.0]
            print('Goal_position', goal_position)
            # Send the goal position to the server
            response = self.set_goal(goal_position)
            time.sleep(1)
            rospy.loginfo("Goal set successfully")
            self.goal_xy = goal_position[0:2]
            print('Goal_position', self.goal_xy)
            threshold = 0.2
            # Calculate the distance between the goal and the current robot state
            distance = math.dist(self.goal_xy, self.robot_state)
            while distance > threshold:
                try:
                    distance = math.dist(self.goal_xy, self.robot_state)
                   
                    time.sleep(0.5)
                except KeyboardInterrupt:
                    print('Loop stopped by user')
                    break
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            self.logger.debug(f"Error in MoveRobotToPick block: {str(e)}\n{traceback.format_exc()}")
           
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        self.logger.debug("  %s [MoveRobotToPlace::terminate().terminate()][%s->%s]" %
                          (self.name, self.status, new_status))

    def ee_pose_callback(self, data):
        self.ee_pose = [data.pose.position.x, data.pose.position.y, data.pose.position.z]

    def aruco_position_callback(self, aruco_msg):
        self.aruco_pose = aruco_msg.data
        blackboard.aruco = self.aruco_pose
        print('Aruco_position', self.aruco_pose)

    def odom_callback(self, odom_data):
        quaternion = (odom_data.pose.pose.orientation.x, odom_data.pose.pose.orientation.y, odom_data.pose.pose.orientation.z, odom_data.pose.pose.orientation.w)
        euler = euler_from_quaternion(quaternion)
        self.robot_state = np.array([odom_data.pose.pose.position.x, odom_data.pose.pose.position.y])

    def signal_handler(self, signal, frame):
        print("Loop stopped by user")
        exit(0)

# Class to handle picking points
class PickPoints(py_trees.behaviour.Behaviour):

    def __init__(self, name):
        super(PickPoints, self).__init__(name)
        # Subscribe to the pose of the end-effector
        self.sub_pose_ee = rospy.Subscriber('pose_EE', PoseStamped, self.ee_pose_callback)
        # Subscribe to the odometry information
        self.sub_odom = rospy.Subscriber("/turtlebot/kobuki/odom_ground_truth", Odometry, self.odom_callback)
        time.sleep(2)



    def setup(self):
        self.logger.debug("  %s [PickPoints::setup()]" % self.name)
        try:
            # Wait for the services to be available (with timeout)
            rospy.wait_for_service('goal_server')
            rospy.wait_for_service('weight_server')
            rospy.wait_for_service('/turtlebot/swiftpro/vacuum_gripper/set_pump')

            # Create service proxies
            self.set_pump_proxy = rospy.ServiceProxy('/turtlebot/swiftpro/vacuum_gripper/set_pump', SetBool)
            self.set_goal = rospy.ServiceProxy('goal_server', intervention)
            self.set_weight = rospy.ServiceProxy('weight_server', intervention)
            self.logger.debug("  %s [PickPoints::setup() Server connected!]" % self.name)

            return True  # MUST return py_trees status now
        except Exception as e:
            self.logger.debug(f"Error in MoveRobotToPick block: {str(e)}\n{traceback.format_exc()}")
           
            return py_trees.common.Status.FAILURE
    '''
    def initialise(self):
        self.goal_xyz = 0
        self.distance = 0
        self.weight = [100000.0, 100000.0, 1.0, 1.0, 1.0, 1.0]
        self.angle = float(math.radians(0))

        self.goal_position = blackboard.aruco
        aruco_x = self.goal_position[0]
        aruco_y = self.goal_position[1]
        aruco_z = self.goal_position[2]

    # ---------- Define offsets ----------
        pre_grasp_offset = -0.10  # 7 cm above marker
        grasp_offset = -0.007      # 1 cm below marker (to make contact)
        lift_offset = -0.07       # 10 cm above marker (after pick)

    # ---------- Define 3-stage pick points ----------
        pick_pre = [aruco_x, aruco_y, aruco_z + pre_grasp_offset, self.angle]
        pick_grasp = [aruco_x, aruco_y, aruco_z + grasp_offset, self.angle]
        pick_lift = [aruco_x, aruco_y, aruco_z + lift_offset, self.angle]

    # Optional: Move back or away after lift (customize as needed)
    # pick_back = [aruco_x + 0.05, aruco_y, aruco_z + lift_offset, self.angle]

        self.point_locations = [pick_pre, pick_grasp, pick_lift]

        self.gripper_on = False
        self.logger.debug("  %s [PickPoints::initialise() updated with 3-point offset strategy]" % self.name)
        '''
    
    def initialise(self):
        self.goal_xyz = 0
        self.distance = 0
        # Define weights for the robot's movements
        self.weight = [100000.0, 100000.0, 1.0, 1.0, 1.0, 1.0]
        self.angle = 0.0
        self.goal_position = blackboard.aruco
        engage_height = self.goal_position[2]-0.28
        holding_height = self.goal_position[2]-0.5

        pick_1 = [self.goal_position[0], self.goal_position[1], engage_height, self.angle]
        pick_2 = [self.robot_state[0], self.robot_state[1] - 0.27, -0.4, self.angle]
        self.point_locations = [pick_1, pick_2]
        self.gripper_on = False
        self.logger.debug("  %s [PickPoints::initialise()]" % self.name)
    
    def update(self):
        try:
            # Set up signal handler for interrupting the process
            # signal.signal(signal.SIGINT, self.signal_handler)
            # Set the weight for the robot
            response = self.set_weight(self.weight)
            rospy.loginfo(f"Weight set to: {self.weight}")
            if self.point_locations:
                for i in range(len(self.point_locations)):
                    rospy.logerr(f'iteration: {i+1}')
                    self.logger.debug("  {}: Publishing goal position".format(self.name))
                    # Send the goal position to the server
                    response = self.set_goal(self.point_locations[i])
                    time.sleep(0.5)
                    if i == 0:
                        response = self.set_pump_proxy(True)
                        threshold = 0.005
                        time.sleep(0.5)
                    else:
                        threshold = 0.08
                    time.sleep(2)
                    rospy.loginfo("Goal set successfully")
                    self.goal_xyz = self.point_locations[i][0:3]
                    
                    # Calculate the distance between the goal and the current end-effector position
                    distance = math.dist(self.goal_xyz, self.ee_pose)
                    while distance > threshold:
                        try:
                            distance = math.dist(self.goal_xyz, self.ee_pose)
                            print(f'error: {distance}')
                            time.sleep(0.5)
                        except KeyboardInterrupt:
                            print('Loop stopped by user')
                            break
                    rospy.logwarn('Goal reached')
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.RUNNING
        except Exception as e:
            self.logger.debug(f"Error in MoveRobotToPick block: {str(e)}\n{traceback.format_exc()}")
           
            return py_trees.common.Status.FAILURE
        
    def terminate(self, new_status):
        self.logger.debug("  %s [PickPoints::terminate().terminate()][%s->%s]" %
                          (self.name, self.status, new_status))

    def ee_pose_callback(self, data):
        self.ee_pose = [data.pose.position.x, data.pose.position.y, data.pose.position.z]

    def odom_callback(self, odom_data):
        quaternion = (odom_data.pose.pose.orientation.x, odom_data.pose.pose.orientation.y, odom_data.pose.pose.orientation.z, odom_data.pose.pose.orientation.w)
        euler = euler_from_quaternion(quaternion)
        self.robot_state = np.array([odom_data.pose.pose.position.x, odom_data.pose.pose.position.y])

    def signal_handler(self, signal, frame):
        print("Loop stopped by user")
        exit(0)

# Class to handle placing points
class PlacePoints(py_trees.behaviour.Behaviour):

    def __init__(self, name):
        super(PlacePoints, self).__init__(name)
        # Subscribe to the pose of the end-effector
        self.sub_pose_ee = rospy.Subscriber('pose_EE', PoseStamped, self.ee_pose_callback)
        # Subscribe to the odometry information
        self.sub_odom = rospy.Subscriber("/turtlebot/kobuki/odom_ground_truth", Odometry, self.odom_callback)
        time.sleep(2)

 
    def setup(self):
        self.logger.debug("  %s [PlacePoints::setup()]" % self.name)
        try:
            # Wait for the services to be available (with timeout)
            rospy.wait_for_service('goal_server')
            rospy.wait_for_service('weight_server')
            rospy.wait_for_service('/turtlebot/swiftpro/vacuum_gripper/set_pump')

            # Create service proxies
            self.set_pump_proxy = rospy.ServiceProxy('/turtlebot/swiftpro/vacuum_gripper/set_pump', SetBool)
            self.set_goal = rospy.ServiceProxy('goal_server', intervention)
            self.set_weight = rospy.ServiceProxy('weight_server', intervention)
            self.logger.debug("  %s [PlacePoints::setup() Server connected!]" % self.name)

            return True  # Must return py_trees status
        except rospy.ServiceException as e:
            self.logger.error("  %s [PlacePoints::setup() Service error: %s]" % (self.name, str(e)))
            return False
        except rospy.ROSException as e:
            self.logger.error("  %s [PlacePoints::setup() Timeout or ROS error: %s]" % (self.name, str(e)))
            return False



    def initialise(self):
        self.goal_xyz = 0
        self.distance = 0
        # Define weights for the robot's movements
        self.weight = [1000.0, 1000.0, 1.0, 1.0, 1.0, 1.0]
        self.angle = float(math.radians(0))
        # Define place points
        place_1 = [self.robot_state[0] + 0.25, self.robot_state[1] - 0.175, -0.3, self.angle]
        place_2 = [self.robot_state[0] + 0.24, self.robot_state[1] - 0.175, -0.134, self.angle]
        self.point_locations = [place_1, place_2]
        self.gripper_on = True
        self.logger.debug("  %s [PlacePoints::initialise()]" % self.name)

    def update(self):
        try:
            # Set up signal handler for interrupting the process
            # signal.signal(signal.SIGINT, self.signal_handler)
            # Set the weight for the robot
            response = self.set_weight(self.weight)
            rospy.loginfo("Weight set successfully")
            if self.point_locations:
                for i in range(len(self.point_locations)):
                    self.logger.debug("  {}: Publishing goal position".format(self.name))
                    point = self.point_locations.pop(0)
                    goal_point = point
                    # Send the goal position to the server
                    response = self.set_goal(goal_point)
                    time.sleep(0.5)
                    time.sleep(2)
                    rospy.loginfo("Goal set successfully")
                    self.goal_xyz = goal_point[0:3]
                    threshold = 0.05
                    # Calculate the distance between the goal and the current end-effector position
                    distance = math.dist(self.goal_xyz, self.ee_pose)
                    while distance > threshold:
                        try:
                            distance = math.dist(self.goal_xyz, self.ee_pose)
                           
                            time.sleep(0.5)
                        except KeyboardInterrupt:
                            print('Loop stopped by user')
                            break
                    rospy.logwarn('Goal reached')
                    if i == 1:
                        response = self.set_pump_proxy(False)
                        time.sleep(0.5)
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.RUNNING
        except:
            self.logger.debug(
                "  {}: (Error  in PlacePoints block) calling service ".format(self.name))
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        self.logger.debug("  %s [PlacePoints::terminate().terminate()][%s->%s]" %
                          (self.name, self.status, new_status))

    def ee_pose_callback(self, data):
        self.ee_pose = [data.pose.position.x, data.pose.position.y, data.pose.position.z]

    def odom_callback(self, odom_data):
        quaternion = (odom_data.pose.pose.orientation.x, odom_data.pose.pose.orientation.y, odom_data.pose.pose.orientation.z, odom_data.pose.pose.orientation.w)
        euler = euler_from_quaternion(quaternion)
        self.robot_state = np.array([odom_data.pose.pose.position.x, odom_data.pose.pose.position.y])

    def signal_handler(self, signal, frame):
        print("Loop stopped by user")
        exit(0)

# Class to move the robot to the home position
class MoveToHome(py_trees.behaviour.Behaviour):

    def __init__(self, name):
        super(MoveToHome, self).__init__(name)
        # Subscribe to the pose of the end-effector
        self.sub_pose_ee = rospy.Subscriber('pose_EE', PoseStamped, self.ee_pose_callback)
        # Subscribe to the odometry information
        self.sub_odom = rospy.Subscriber("/turtlebot/kobuki/odom_ground_truth", Odometry, self.odom_callback)
       
        time.sleep(2)



    def setup(self):
        self.logger.debug("  %s [MoveToHome::setup()]" % self.name)
        try:
            # Wait for the services to be available (with timeout)
            rospy.wait_for_service('goal_server')
            rospy.wait_for_service('weight_server')

            # Create service proxies
            self.set_goal = rospy.ServiceProxy('goal_server', intervention)
            self.set_weight = rospy.ServiceProxy('weight_server', intervention)
            self.logger.debug("  %s [MoveToHome::setup() Server connected!]" % self.name)

            return True  # Must return py_trees status
        except rospy.ServiceException as e:
            self.logger.error("  %s [MoveToHome::setup() Service error: %s]" % (self.name, str(e)))
            return False
        except rospy.ROSException as e:
            self.logger.error("  %s [MoveToHome::setup() Timeout or ROS error: %s]" % (self.name, str(e)))
            return False


        
    def initialise(self):
        self.goal_xyz = 0
        self.distance = 0
        # Define the home position for the robot base 
        self.weight = [1.0, 1.0, 1000.0, 1000.0, 1000.0, 1000.0]
        self.home_position = [0.0, 0.0, 0.0, 0.0]
        self.source_frame = 'world_ned'
        self.logger.debug("  %s [MoveToHome::initialise()]" % self.name)

    def update(self):
        try:
            # Set up signal handler for interrupting the process
            # signal.signal(signal.SIGINT, self.signal_handler)
            # Move the robot base to the home position
            response=self.set_weight(self.weight)
            response = self.set_goal(self.home_position)
            time.sleep(0.5)
            rospy.loginfo("Home position goal set successfully")
            self.goal_xy = self.home_position[0:2]
            print('Goal_position',self.goal_xy)
            threshold = 0.4
            # Calculate the distance between the goal and the current end-effector position
            distance = math.dist(self.goal_xy, self.robot_state)
            while distance > threshold:
                try:
                    distance = math.dist(self.goal_xy, self.robot_state)
                   
                    time.sleep(0.5)
                    if distance<threshold:
                        rospy.logwarn('Home position reached')
                        break
                except KeyboardInterrupt:
                    print('Loop stopped by user')
                    break
            
            return py_trees.common.Status.SUCCESS
       
        except rospy.ServiceException as e:
            self.logger.debug("  %s [MoveToHome::update() Service call failed: %s]" % (self.name, str(e)))
            return py_trees.common.Status.FAILURE
    
    def terminate(self, new_status):
        self.logger.debug("  %s [MoveToHome::terminate().terminate()][%s->%s]" %
                          (self.name, self.status, new_status))

    def ee_pose_callback(self, data):
        self.ee_pose = [data.pose.position.x, data.pose.position.y, data.pose.position.z]

    def odom_callback(self, odom_data):
        quaternion = (odom_data.pose.pose.orientation.x, odom_data.pose.pose.orientation.y, odom_data.pose.pose.orientation.z, odom_data.pose.pose.orientation.w)
        euler = euler_from_quaternion(quaternion)
        self.robot_state = np.array([odom_data.pose.pose.position.x, odom_data.pose.pose.position.y])

    def signal_handler(self, signal, frame):
        print("Loop stopped by user")
        exit(0)

# Function to create the behavior tree
def create_tree():
    # Create instances of each behavior
    move_robot_to_pick = MoveRobotToPick("Move_Robot_To_Pick_Object")
    pick_points = PickPoints("Pick_Object")
    move_robot_to_place = MoveRobotToPlace("Move_Robot_To_Place_Object")
    place_points = PlacePoints("Place_Object")
    move_to_home = MoveToHome("Move_To_Home_Position")

    # Create the root of the behavior tree
    root = py_trees.composites.Sequence(name="Aruco_Pick_and_Place", memory=True)
    # Add the behaviors as children of the root
    root.add_children([move_robot_to_pick, pick_points, move_robot_to_place, place_points, move_to_home])

    return root

# Function to execute the behavior tree
def execute_behavior_tree(tick_count=1):
    """
    Sets up and executes the behavior tree for a specified number of ticks.
    
    :param tick_count: Number of times the tree should be ticked.
    """
    root = create_tree()
    behavior_tree = py_trees.trees.BehaviourTree(root)
    py_trees.display.render_dot_tree(root, name="behavior_tree")  # Save the tree structure to a dot file

    try:
        print("Setting up for all tree children...")
        behavior_tree.setup()  # Setup the tree with a timeout


        # Tick the tree for the specified number of ticks or until interrupted
        for _ in range(tick_count):
            try:
                behavior_tree.tick()
                rospy.sleep(1)  # Simulate time passing (e.g., waiting for 1 second before the next tick)
            except KeyboardInterrupt:
                print("Interrupted by the user.")
                break
    except KeyboardInterrupt:
        print("Execution interrupted.")

if __name__ == "__main__":
    py_trees.logging.level = py_trees.logging.Level.DEBUG
    blackboard = py_trees.blackboard.Blackboard()
    rospy.init_node("behavior_tree")
    execute_behavior_tree(tick_count=1)
    rospy.spin()

