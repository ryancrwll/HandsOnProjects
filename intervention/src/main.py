#!/usr/bin/env python3
import numpy as np 
import rospy 
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray,Int32
from geometry_msgs.msg import PoseStamped 
#from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import tf.transformations as tf 
from tf.transformations import*
from define import *

# “The controller node that takes a target pose, uses math to compute how to move, 
# and tells the robot exactly what to do — in real-time.”


# This is the main controller node of the project.
# It listens for a target position (goal), reads the robot’s current joint angles 
# and pose, and then computes how the base and joints should move to reach the goal. 
# It uses Jacobian-based inverse kinematics to generate smooth and coordinated 
# motions — and sends commands to both the base and the arm.


# This is the main controller that connects all the tasks we defined in define.py. 
# It receives a goal for the end-effector, and based on the robot’s current state, 
# it calculates how to move the base and joints using weighted damped least squares. 
# Then it sends the velocity commands to the robot in real time, making the 
# arm and base work together to reach the goal.

class MainIntervention:    
    def __init__(self): 
        self.wheel_base_distance = 0.23 
        self.wheel_radius =  0.035  
        self.dt = 0.1
        self.state = [0.0, 0.0, 0.0]  # [x, y, yaw]
        self.vmax = 0.2
        self.max = 0.5
        self.max_a = self.max - 0.01 
        self.min = -0.5  
        self.min_a = self.min + 0.01 
        self.goal = None 
        self.tasks = []
        self.q = np.zeros((6, 1)) # q is the joint position vector
        self.q[2] = 0.0
        self.last_control_time = rospy.Time.now().to_sec()
  
        # Publishers
        # Publishes the current end-effector pose for visualization in RViz.
        self.pose_EE_pub = rospy.Publisher('/pose_EE', PoseStamped, queue_size=10)
        # Publishes joint angles
        self.q_vals_pub = rospy.Publisher('/q_vals', Float64MultiArray, queue_size=10)
        # Publishes the goal pose (used as a visual target marker in RViz).   
        self.goal_check = rospy.Publisher('/goal_check', PoseStamped, queue_size=10) 
        # Sends a vector of joint velocities to move the SwiftPro arm. | Tells each joint how fast to rotate.
        self.joint_velocity = rospy.Publisher("/turtlebot/swiftpro/joint_velocity_controller/command", Float64MultiArray, queue_size=10) 
        # Publishes left and right wheel velocities to move the TurtleBot base.
        # Acts like Steering input and Tells wheels to move
        self.wheel_velocity = rospy.Publisher("/turtlebot/kobuki/commands/wheel_velocities", Float64MultiArray, queue_size=10)     
        # Publishes the wheel velocities in JointState format.
        # Acts like speedometer to display speed and	tells us what we're doing
        self.J_wheel_velocity = rospy.Publisher("/velocities", JointState, queue_size=10)      
        # Another way to control the robot using velocity commands (Twist messages)
        #self.cmd_vel_pub = rospy.Publisher("/turtlebot/kobuki/commands/velocity", Twist, queue_size=10)

        # Subscribers
        self.weight_sub = rospy.Subscriber('/weight_set', Float64MultiArray, self.weight_service)  
        self.goal_sub = rospy.Subscriber('/goal_set', Float64MultiArray, self.goal_service)  
        self.task_sub = rospy.Subscriber('/task_set', Int32, self.task_service)   
        self.joints_sub = rospy.Subscriber('/turtlebot/joint_states', JointState, self.JointState_callback)   
        # /odom_ground_truth gives the real-time position and rotation of the robot base, 
        # so the controller knows where it is and can plan where to go.
        self.odom_sub = rospy.Subscriber("/turtlebot/kobuki/odom_ground_truth", Odometry, self.odom_callback)

    def task_service(self, task_index):
        # This function lets the robot select which control tasks to activate, 
        # like: Move to a goal position? Respect joint limits?

        self.selected_task = task_index.data
        if self.selected_task == 0:
            rospy.loginfo('Prioritizing ee_pose')
            tasks = [
                Jointlimits3D("First Joint", np.array([0.0]), np.array([self.max, self.max_a, self.min, self.min_a]),1),  
                Jointlimits3D("Second Joint", np.array([0.0]), np.array([self.max, self.max_a, self.min, self.min_a]),2),
                # Here, only Joint 1 and Joint 2 are protected with joint limit tasks because they’re the most likely 
                # to exceed safe limits during motion. Joint 3 is not as critical, so we skip it for simplicity. 
                # But we could easily add a Jointlimits3D for Joint 3 if needed for safety or completeness.
                Position3D("End-Effector Position", np.array(self.goal[0:3]).reshape(3,1), 6)
            ]   
        elif self.selected_task == 1:
            rospy.loginfo('Prioritizing joint 3 angle then ee_pose')
            tasks = [
                Jointlimits3D("First Joint", np.array([0.0]), np.array([self.max, self.max_a, self.min, self.min_a]),1),  
                Jointlimits3D("Second Joint", np.array([0.0]), np.array([self.max, self.max_a, self.min, self.min_a]),2),
                # Here, only Joint 1 and Joint 2 are protected with joint limit tasks because they’re the most likely 
                # to exceed safe limits during motion. Joint 3 is not as critical, so we skip it for simplicity. 
                # But we could easily add a Jointlimits3D for Joint 3 if needed for safety or completeness.
                JointPosition3D("3rd Joint Position", np.array([self.goal[4]-np.pi/2]), 3),
                Position3D("End-Effector Position", np.array(self.goal[0:3]).reshape(3,1), 6)
            ]
        else:
            rospy.loginfo('Prioritizing joint 3 angle')
            tasks = [
                Jointlimits3D("First Joint", np.array([0.0]), np.array([self.max, self.max_a, self.min, self.min_a]),1),  
                Jointlimits3D("Second Joint", np.array([0.0]), np.array([self.max, self.max_a, self.min, self.min_a]),2),
                # Here, only Joint 1 and Joint 2 are protected with joint limit tasks because they’re the most likely 
                # to exceed safe limits during motion. Joint 3 is not as critical, so we skip it for simplicity. 
                # But we could easily add a Jointlimits3D for Joint 3 if needed for safety or completeness.
                JointPosition3D("3rd Joint Position", np.array([self.goal[4]-np.pi/2]), 3)
            ]
        self.tasks = tasks

    def goal_service(self, goal_msg):
        self.goal = [goal_msg.data[i] for i in range(len(goal_msg.data))]
        self.goal_pose = goal_pose(np.array(self.goal[0:3]), np.array(self.goal[3]))
        self.goal_check.publish(self.goal_pose)

        self.tasks = [
            Jointlimits3D("First Joint", np.array([0.0]), np.array([self.max, self.max_a, self.min, self.min_a]),1),  
            Jointlimits3D("Second Joint", np.array([0.0]), np.array([self.max, self.max_a, self.min, self.min_a]),2),
            # Here, only Joint 1 and Joint 2 are protected with joint limit tasks because they’re the most likely 
            # to exceed safe limits during motion. Joint 3 is not as critical, so we skip it for simplicity. 
            # But we could easily add a Jointlimits3D for Joint 3 if needed for safety or completeness.
            Position3D("End-Effector Position", np.array(self.goal[0:3]).reshape(3,1), 6)
        ]   

    def weight_service(self, weight_msg):
        # saves how much to prefer or avoid moving each joint or base part. 
        # Large weight, less move. Less weight, large move.

        # The weight_service() function stores the weights that control how much each joint 
        # or base movement is preferred during task execution.
        self.weight = weight_msg.data

    def odom_callback(self, odom_msg): 
        # reads the robot’s current base position and orientation so the controller knows where it is.
        
        # The odom_callback() function reads the robot's current position (x, y) and rotation (yaw) from the /odom_ground_truth topic, 
        # converts the orientation from quaternion to a usable yaw angle, and stores the full base pose in self.state. 
        # This pose is essential for computing transformations and controlling the robot accurately 
        # in world coordinates.
        quaternion = (
            # This pulls out the rotation of the robot’s base, represented as a quaternion (4D format).
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w
        )
        _, _, yaw = euler_from_quaternion(quaternion) 
        self.state = np.array([
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            yaw # The yaw in self.state comes from the quaternion.
        ])     

    def send_velocity(self, q):

        # This function receives a velocity vector from the task solver and splits it into joint 
        # commands and base commands. It sends joint velocities to the SwiftPro arm. Then it converts 
        # base linear/angular velocity into left/right wheel speeds, and publishes both to 
        # make the robot move. It also sends the wheel speeds as a status message so other 
        # parts of the system can use them.

        # These are joint velocities of the SwiftPro arm.
        joint_vel_msg = Float64MultiArray()
        # Index 2 - 5, The 4 joint velocity of the robot.
        joint_vel_msg.data = [float(q[2, 0]), float(q[3, 0]), float(q[4, 0]), float(q[5, 0])] 
        self.joint_velocity.publish(joint_vel_msg) # This makes the arm start moving based on task output.

        w = q[0, 0]  # Angular velocity
        v = q[1, 0]  # Linear velocity
        # These are the actual motor speeds needed to make the robot move as desired.
        v_r = (2 * v + w * self.wheel_base_distance) / (2 * self.wheel_radius) # Speed for right wheel
        v_l = (2 * v - w * self.wheel_base_distance) / (2 * self.wheel_radius) # Speed for left wheel

        # Controls wheel motion
        # This is what moves the TurtleBot base forward, backward, or makes it turn.
        wheel_vel_msg = Float64MultiArray()
        wheel_vel_msg.data = [float(v_r), float(v_l)]
        self.wheel_velocity.publish(wheel_vel_msg)

        # Shares wheel speeds with other nodes
        # This does not control anything, but simply broadcasts the same wheel 
        # speeds as a JointState message.
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.velocity = [float(v_r), float(v_l)]
        self.J_wheel_velocity.publish(joint_state_msg)
        '''
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_vel_pub.publish(cmd)
        '''

    def JointState_callback(self, msg):
        # This is a ROS subscriber callback. It gets triggered whenever the robot publishes a new 
        # JointState message about the joints' positions.

        # Every time my robot arm updates its joint angles, I check if a goal is set. 
        # If yes, I extract those angles, update the robot model, and then compute how to 
        # move next to reach the goal.

        if self.goal is not None: # This ensures that: The robot will only respond to joint states 
            # after a goal has been set If no goal exists, it skips further processing.

            names = ['turtlebot/swiftpro/joint1', 'turtlebot/swiftpro/joint2', 'turtlebot/swiftpro/joint3', 'turtlebot/swiftpro/joint4']
            if msg.name == names: # This checks if the received message exactly matches those 4 joint names.
                # Extracting the current joint angles from the message
                self.theta = np.array(msg.position, ndmin=2).T # msg.position is a list like [θ1, θ2, θ3, θ4] 
                                                               # ndmin=2 forces it into a 2D array
                self.robot = Manipulator(self.theta) # create a new instance of Manipulator class, using the current joint angles.
                self.update_robot_and_tasks()

    def update_robot_and_tasks(self):
        
        # This function computes how the robot should move based on the current active tasks. 
        # It starts with zero motion, then adds motion for each task in priority order using 
        # Jacobians and the WDLS solver, applies speed limits, sends out the commands, 
        # and updates the robot's state.
        
        dt = self.dt
        P = np.eye(self.robot.getDOF()) # P is the null-space projection matrix
        dq = np.zeros((self.robot.getDOF(), 1)) # dq is the joint + base velocity vector we're going to compute.
        self.robot.update(dq, 0.0, self.state) # This updates the robot’s internal state (Manipulator class).

        for i in range(len(self.tasks)): # Loop through all tasks
            self.tasks[i].update(self.robot) # The current robot state — compute Jacobian and error.
            if self.tasks[i].bool_is_Active(): # Only solve the task if it's active
                err = self.tasks[i].getError() # err: how far you are from the goal times a gain
                gain = 0.8 * np.identity(len(err))
                err = gain @ err
                J = self.tasks[i].getJacobian() # J: how changes in joints + base affect the task
                if self.tasks[i].name == "3rd Joint Position":
                    J = np.array([0,0,1,0,0,0]).reshape(1,6)
                J_bar = J @ P # This modifies the Jacobian so it's projected into the null space of previously completed tasks.
                J_DLS = W_DLS(J_bar, 0.01, self.weight) # Adds damping (0.1) to handle singularities
                J_pinv = np.linalg.pinv(J_bar) # Compute pseudo-inverse for null space
                dq = dq + J_DLS @ (err - J @ dq) # This computes the contribution of this task to the final dq velocity vector.
                P = P - (J_pinv @ J_bar) # Modifies P so that future tasks are solved in the null space of the current task.
                # Apply velocity limits
                largest_dq = np.max(np.abs(dq))
                scale_factor = 1.0
                if largest_dq > self.vmax:
                    scale_factor = self.vmax/largest_dq
                dq = dq*scale_factor
                dq = dq.reshape(6, 1)

        self.send_velocity(dq)
        dt = rospy.Time.now().to_sec() - self.last_control_time
        self.last_control_time = rospy.Time.now().to_sec()
        
        self.q += dq * dt
        msg = Float64MultiArray()
        msg.data = self.q.flatten().tolist()
        self.q_vals_pub.publish(msg)
        pose_ee = pose_EE(self.robot.getEETransform())
        self.pose_EE_pub.publish(pose_ee) # This publishes the current end-effector pose to RViz or logs.
        self.robot.update(dq, dt, self.state) # Updates the robot model for the next cycle 

if __name__ == '__main__': 
    try:
        rospy.init_node('main_node', anonymous=True)  
        MainIntervention()
        rospy.spin() 
    except rospy.ROSInterruptException:
        pass
