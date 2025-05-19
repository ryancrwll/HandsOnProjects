#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from tf.transformations import euler_from_quaternion
from define import Manipulator, Position3D, W_DLS, goal_pose, pose_EE

class FullManipulatorController:
    def __init__(self):
        self.dt = 0.01
        self.state = [0.0, 0.0, 0.0]
        self.theta = None
        self.goal = None
        self.tasks = []
        self.weight = [30.0, 30.0, 5.0, 5.0, 10.0, 10.0]


        # Publishers
        self.pose_pub = rospy.Publisher("pose_EE", PoseStamped, queue_size=10)
        self.marker_pub = rospy.Publisher("ee_trajectory_marker", Marker, queue_size=10)
        self.joint_vel_pub = rospy.Publisher("/swiftpro/joint_velocity_controller/command", Float64MultiArray, queue_size=10)
        self.wheel_pub = rospy.Publisher("/turtlebot/kobuki/commands/wheel_velocities", Float64MultiArray, queue_size=10)
        self.joint_state_pub = rospy.Publisher("/velocities", JointState, queue_size=10)

        # Subscribers
        self.goal_sub = rospy.Subscriber("/desired_goal", PoseStamped, self.goal_callback)
        self.joint_sub = rospy.Subscriber("/swiftpro/joint_states", JointState, self.joint_callback)
        self.odom_sub = rospy.Subscriber("/turtlebot/kobuki/odom_ground_truth", Odometry, self.odom_callback)

        self.trajectory_points = []

    def goal_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        yaw = 0.0  # Assuming flat world, no pitch/roll
        self.goal = [[x, y, z], [yaw]]
        self.goal_pose = goal_pose(np.array(self.goal[0]), np.array(self.goal[1]))
        self.tasks = [
            Position3D("End-Effector Position", np.array(self.goal[0]).reshape(3, 1), 6),
        ]

    def joint_callback(self, msg):
        expected_names = ['swiftpro/joint1', 'swiftpro/joint2', 'swiftpro/joint3', 'swiftpro/joint4']
        if all(name in msg.name for name in expected_names):
            self.theta = np.array(msg.position, ndmin=2).T
            self.robot = Manipulator(self.theta)
            self.update()

    def odom_callback(self, msg):
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        _, _, yaw = euler_from_quaternion(quaternion)
        self.state = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]

    def update(self):
        if self.goal is None or self.theta is None:
            return

        dq = np.zeros((6, 1))
        self.robot.update(dq, self.dt, self.state)
        P = np.eye(self.robot.getDOF())

        for task in self.tasks:
            task.update(self.robot)
            if task.bool_is_Active():
                err = task.getError()
                J = task.getJacobian()
                J_proj = J @ P
                J_dls = W_DLS(J_proj, 0.01, self.weight)
                dq += J_dls @ (err - J @ dq)
                J_pinv = np.linalg.pinv(J_proj)
                P -= J_pinv @ J_proj

        dq = np.clip(dq, -0.5, 0.5)
        self.robot.update(dq, self.dt, self.state)
        self.send_velocity(dq)
        self.publish_pose()

    def send_velocity(self, dq):
        # Arm joints (joint 1 to 4)
        joint_msg = Float64MultiArray()
        joint_msg.data = [float(dq[2]), float(dq[3]), float(dq[4]), float(dq[5])]
        self.joint_vel_pub.publish(joint_msg)

        # Base wheel velocities
        w = dq[0, 0]
        v = dq[1, 0]
        wheel_base = 0.23
        wheel_radius = 0.035
        v_r = (2 * v + w * wheel_base) / (2 * wheel_radius)
        v_l = (2 * v - w * wheel_base) / (2 * wheel_radius)

        wheel_msg = Float64MultiArray()
        wheel_msg.data = [float(v_r), float(v_l)]
        self.wheel_pub.publish(wheel_msg)

        joint_state_msg = JointState()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.velocity = [v_r, v_l]
        self.joint_state_pub.publish(joint_state_msg)

    def publish_pose(self):
        pose = pose_EE(self.robot.getEETransform())
        self.pose_pub.publish(pose)
        self.trajectory_points.append(pose.pose.position)
        self.publish_marker()

    def publish_marker(self):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "ee_trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale = Vector3(x=0.005, y=0.0, z=0.0)
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        marker.points = self.trajectory_points
        self.marker_pub.publish(marker)

if __name__ == '__main__':
    rospy.init_node("full_robot_with_trajectory")
    FullManipulatorController()
    rospy.spin()
