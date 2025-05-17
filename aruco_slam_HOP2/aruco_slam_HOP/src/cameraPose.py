#!/usr/bin/env python3

import rospy
import cv2
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import cv2.aruco as aruco
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from tf.transformations import euler_from_quaternion

from aruco_slam_hop.aruco_markerHandler import MarkerHandler

from geometry_msgs.msg import PoseStamped, Point, Quaternion

class MarkerNode:
    def __init__(self, marker_id, rvec, tvec):
        self.marker_id = marker_id
        self.rvec = rvec
        self.tvec = tvec


class arucoDetectorNode:
    def __init__(self):
        # Initialize the ROS node with the name 'aruco_detector'
        rospy.init_node('aruco_detector')
        
        # Create a CvBridge object to convert between ROS Image messages and OpenCV images
        self.bridge = CvBridge()
        
        # Subscribers
        self.image_sub = rospy.Subscriber('/turtlebot/kobuki/realsense/color/image_raw', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/turtlebot/kobuki/odom', Odometry, self.odom_callback, queue_size=10)
        # self.gt_odom_sub = rospy.Subscriber('/turtlebot/kobuki/odom_ground_truth', Odometry, self.gt_odom_callback)
        
        # Publishers
        self.pose_pub = rospy.Publisher('/turtlebot/kobuki/aruco_position_perception', Float64MultiArray, queue_size=10)
        self.aruco_rviz_marker_pub = rospy.Publisher('/aruco/visualization_aruco_perception', MarkerArray, queue_size=10)

        # Robot's pose [x, y, yaw] in the global (world) frame, initialized to zero
        self.state = np.zeros(3)    

        self.aruco_dict_type = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        
        # Camera intrinsic matrix
        self.camera_matrix = np.array([[1396.8086675255468, 0.0, 960.0],
                                       [0.0, 1396.8086675255468, 540.0],
                                       [0.0, 0.0, 1.0]])
        # Define distortion coefficients
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # List to store detected markers objects (each containing marker_id, rvec, tvec)
        self.markers_detected = []
        # Dictionary to store global marker positions {aruco_id: position}
        self.marker = {}

        # MarkerHandler instance to manage marker indexing and position storage
        self.marker_handler = MarkerHandler()

        # self.marker_length = 0.1875
        self.marker_length = 0.05


    def odom_callback(self, odom_msg):
        '''
            This callback function extracts the robot's current position and orientation from the odometry
            message, converts the orientation from quaternion to Euler angles, and updates the robot's pose 
            state [x, y, yaw] in the global (world) frame.
        '''
        # Extract the robot's position
        x_pose = odom_msg.pose.pose.position.x
        y_pose = odom_msg.pose.pose.position.y

        # Extract the orientation as a quaternion (x, y, z, w)
        quaternion = (odom_msg.pose.pose.orientation.x,
                      odom_msg.pose.pose.orientation.y,
                      odom_msg.pose.pose.orientation.z,
                      odom_msg.pose.pose.orientation.w)
        
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        roll, pitch, yaw = euler_from_quaternion(quaternion)

        # Update the robot's pose state [x, y, yaw] in the global (world) frame
        self.state = np.array([x_pose, y_pose, yaw])

    
    def gt_odom_callback(self, msg):
        self.x_gt = msg.pose.pose.position.x
        self.y_gt = msg.pose.pose.position.y
 
    
    def aruco_pose_estimation(self, frame, aruco_dict, camera_matrix, dist_coeffs):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters()

        corners, ids, rejected = aruco.detectMarkers(
                                                        gray, 
                                                        aruco_dict, 
                                                        parameters=parameters
                                                     )
        marker_poses = []

        if ids is not None:
            for i, corner in enumerate(corners):
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, self.marker_length, camera_matrix, dist_coeffs)
                aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

                marker_poses.append(MarkerNode(ids[i][0], rvec, tvec))
                x, y, z = tvec[0][0][0], tvec[0][0][1], tvec[0][0][2]

                cv2.putText(frame, f"X: {x:.2f}", (10, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Y: {y:.2f}", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Z: {z:.2f}", (10, 85), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

                self.aruco_callback(x, y, z, ids[i][0])

        return frame, marker_poses


    def image_callback(self, img_msg):
        '''
            This callback function recieves raw image data from the robot's camera, convert it to an OpenCV
            image, detects ArUco markers, and accumulates inverted camera poses based on the relative pose 
            of each marker. If multiple markers are seen, it averages their poses to estimate a more stable
            camera position.
        '''
        try:
            # Convert ROS Image message to OpenCV BGR8 image using CvBridge
            frame = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return
        
        cumulative_rvec = np.zeros((3,1))       # Sum of rotation vectors
        cumulative_tvec = np.zeros((3,1))       # Sum of translation vectors
        valid_marker_count = 0                  # Counter for how many valid markers are processed

        # Detect aruco markers and estimate pose of each
        frame, detected_markers = self.aruco_pose_estimation(frame, self.aruco_dict_type, self.camera_matrix, self.dist_coeffs)

        for marker in detected_markers:
            self.markers_detected.append(marker)

            # Get camera pose from marker by inverting the pose
            rvec_inv, tvec_inv = self.invert_pose(marker.rvec, marker.tvec)
            cumulative_rvec += rvec_inv
            cumulative_tvec += tvec_inv
            valid_marker_count += 1

        # Average the estimated camera pose if multiple markers are detected
        if valid_marker_count > 1:
            average_rvec = cumulative_rvec / valid_marker_count
            average_tvec = cumulative_tvec / valid_marker_count

            # rospy.logwarn(f"Real-time Camera Pose (Rotation Vector): {average_rvec.flatten()}")
            # rospy.logwarn(f"Real-time Camera Pose (Translation Vector): {average_tvec.flatten()}")


    def invert_pose(self, rvecs, tvecs):
        '''
            Inverts the pose from camera-marker to marker-camera.
            This function returns the inverted rotation and translation vectors.
        '''
        rot, _ = cv2.Rodrigues(rvecs)
        rot_inv = rot.T
        
        tvecs = tvecs.reshape((3,1))
        
        tvec_inv = -np.dot(rot_inv, tvecs)
        rvec_inv, _ = cv2.Rodrigues(rot_inv)

        return rvec_inv, tvec_inv
    

    def boxplus(self, robot_pose_global, marker_relative_pose):
        '''
            Applies a 2D pose transformation (SE2) to compute global marker position.

            Parameters:
            - robot_pose_global: np.array([x, y, theta])  # Robot pose in world frame
            - marker_relative_pose: np.array([[x], [y]])  # Marker position in robot frame

            Returns:
            - marker_global: np.array([[x], [y]])       # Marker pose in world frame
        '''
        x, y, theta = robot_pose_global
        dx, dy = marker_relative_pose[0,0], marker_relative_pose[1,0]

        marker_x = x + np.cos(theta) * dx - np.sin(theta) * dy
        marker_y = y + np.sin(theta) * dx + np.cos(theta) * dy

        return np.array([[marker_x], [marker_y]])
    

    def g(self, robot_pose, marker_relative):
        '''
            Wrapper for SE2 transformation.
            
            g(x,z):
                    x = robot pose (state)
                    z = observation measured in robot frame
        '''
        return self.boxplus(robot_pose, marker_relative)
    

    def camera_to_robot_transform(self, marker_in_camera_frame):
        '''
            Transforms a marker's position from the camera frame to the robot's base frame.
            
            Parameters:
            - marker_in_camera_frame: np.array([[z], [x], [y]])  # Position of marker in camera frame

            Returns:
            - marker_in_robot_frame: np.array([[x], [y], [z]])   # Transformed into robot's frame
        '''
        CxF_x = marker_in_camera_frame[0,0]
        CxF_y = marker_in_camera_frame[1,0]
        CxF_z = marker_in_camera_frame[2,0]

        # Known fixed offset of camera w.r.t. robot base
        RxC_x, RxC_y, RxC_z = 0.122, -0.033, 0.082
        RxC_yaw = 0.0   # Assuming no rotation between camera and robot

        # Apply transformation using 2D rotation and translation
        RxF_x = RxC_x + CxF_x * np.cos(RxC_yaw) - CxF_y * np.sin(RxC_yaw)
        RxF_y = RxC_y + CxF_x * np.sin(RxC_yaw) + CxF_y * np.cos(RxC_yaw)
        RxF_z = RxC_z + CxF_z

        return np.array([[RxF_x], [RxF_y], [RxF_z]])
    

    def get_yaw_from_rvec(self, rvec):
        '''
            Converts a rotation vector (rvec) to yaw angle (theta) from ArUco detection.
        '''
        rvec = np.array(rvec, dtype=np.float64)
        r_ct, _ = cv2.Rodrigues(rvec)
        yaw_marker = np.arctan2(r_ct[1, 0], r_ct[0, 0])
        return yaw_marker
    

    def aruco_callback(self, x, y, z, marker_id):
        '''
            Handles ArUco marker detection results, transforms them to global frame, updates
            marker map, and publishes markers to Rviz.

            Parameters:
            - x, y, z: coordinates of marker in camera frame
            - marker_id: ArUco IF of the detected marker
        '''
        self.aruco_x, self.aruco_y, self.aruco_z = x, y, z
        self.aruco_id = marker_id

        aruco_position_cam = np.array([self.aruco_x, self.aruco_y, self.aruco_z])

        # Rearrange to [z, x, y] to match expected input format (camera frame) for transform
        marker_pos_camera_frame = np.array([[self.aruco_z], [self.aruco_x], [self.aruco_y]])

        # Convert marker position to robot frame using static camera-to-robot transform
        marker_pos_robot_frame = self.camera_to_robot_transform(marker_pos_camera_frame)

        # Get robot's current pose [x, y, yaw] in world frame odometry
        robot_pose_global = self.state[0:3]

        # Convert marker position from robot frame to world frame
        marker_pos_world = self.g(robot_pose_global, marker_pos_robot_frame[0:2])

        # Store the marker globally if it's new, using the MarkerHandler
        self.marker_handler.add_marker(self.aruco_id, marker_pos_world)
        self.marker[self.aruco_id] = marker_pos_world

        # Find the corresponding rvec
        rvec_marker = np.array([[0], [0], [0]])         # default if not found

        for marker in self.markers_detected:
            if marker.marker_id == self.aruco_id:
                rvec_marker = marker.rvec
                break

        # Estimated yaw from the aruco marker
        self.yaw_marker = self.get_yaw_from_rvec(rvec_marker)

        # Get index of this marker in the observed list
        self.idx = self.marker_handler.get_index(self.aruco_id)

        # Print all observed marker IDs for debugging
        rospy.loginfo(f"Observed ArUco list: {self.marker_handler.observed_arucos}")

        # If the marker was successfully stored, visualize it in Rviz
        if self.aruco_id in self.marker_handler.observed_arucos:
            self.visualization_marker(self.idx, marker_pos_world)

        # Call localization step if the marker is known
        self.localize_robot(self.aruco_id, marker_pos_robot_frame[0:2])

    
    def visualization_marker(self, idx, aruco_position):
        '''
            Publishes RViz visualization markers for detected ArUco markers.

            This includes:
            - A cube for the marker position
            - A text label with the marker ID
            - Optional lines and distance labels between all detected markers

            Parameters:
            - idx: Index of the current ArUco marker
            - aruco_position: Position of the marker in the global (world) frame
        '''
        marker_array = MarkerArray()
        aruco_id = list(self.marker_handler.observed_arucos.keys())[idx]

        # Marker cube
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "aruco_cube"
        marker.id = aruco_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = aruco_position[0, 0]
        marker.pose.position.y = aruco_position[1, 0]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0)  # Lasts until updated or deleted
        marker_array.markers.append(marker)

        # Text label marker
        text_marker = Marker()
        text_marker.header.frame_id = "odom"
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = "aruco_label"
        text_marker.id = aruco_id + 20000  # Unique ID for text label
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = aruco_position[0, 0]
        text_marker.pose.position.y = aruco_position[1, 0]
        text_marker.pose.position.z = 0.5
        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z = 0.20
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.text = f"ArUco {aruco_id}"
        marker_array.markers.append(text_marker)

        # Line and distance between consecutive markers
        if len(self.marker) > 1:
            aruco_ids = list(self.marker.keys())
            positions = list(self.marker.values())

            for i in range(len(positions) - 1):
                # Compute euclidean distance
                dist = np.linalg.norm(positions[i] - positions[i+1])

                # Line Marker between consecutive markers
                line_marker = Marker()
                line_marker.header.frame_id = "odom"
                line_marker.header.stamp = rospy.Time.now()
                line_marker.ns = "line_markers"
                line_marker.id = 10000 + i                                  # Unique ID for each line segment (based on index)
                line_marker.type = Marker.LINE_STRIP                        # Marker type: line connecting two points
                line_marker.action = Marker.ADD
                line_marker.scale.x = 0.04  # Line thickness
                line_marker.color.r = 1.0
                line_marker.color.g = 0.0
                line_marker.color.b = 1.0
                line_marker.color.a = 1.0
                line_marker.lifetime = rospy.Duration(0)
                # Define the two endpoints of the line based on consecutive ArUco positions
                p1 = Point(positions[i][0], positions[i][1], 0.0)           # Start point of the line
                p2 = Point(positions[i + 1][0], positions[i + 1][1], 0.0)   # End point of the line
                line_marker.points = [p1, p2]                               # Add both points to the line marker
                marker_array.markers.append(line_marker)                    # Add the line marker to the marker array for publishing

                # Text Marker to show distance
                dist_marker = Marker()
                dist_marker.header.frame_id = "odom"
                dist_marker.header.stamp = rospy.Time.now()
                dist_marker.ns = "distance_labels"
                dist_marker.id = 30000 + i                                  # Unique ID for each distance label
                dist_marker.type = Marker.TEXT_VIEW_FACING
                dist_marker.action = Marker.ADD
                # Position the text label at the midpoint between the two markers
                dist_marker.pose.position.x = (positions[i][0] + positions[i + 1][0]) / 2
                dist_marker.pose.position.y = (positions[i][1] + positions[i + 1][1]) / 2
                dist_marker.pose.position.z = 0.35                          # Slightly elevated above the ground
                dist_marker.scale.z = 0.22
                dist_marker.color.r = 1.0
                dist_marker.color.g = 1.0
                dist_marker.color.b = 0.0
                dist_marker.color.a = 1.0
                dist_marker.text = f"{dist:.2f}m"
                marker_array.markers.append(dist_marker)                    # Add the text marker to the array

                # Log the distance in console
                rospy.loginfo(f"Distance between ArUco {aruco_ids[i]} and ArUco {aruco_ids[i+1]}: {dist:.2f} meters")

        # Publish the full marker array to RViz
        self.aruco_rviz_marker_pub.publish(marker_array)


    def localize_robot(self, marker_id, marker_position_robot_frame):
        """
            Correct the robot's position based on observing a known marker.
            
            :param marker_id: ID of the observed marker
            :param marker_position_robot_frame: (x, y) position of marker in robot frame
        """
        # Only proceed if the marker is already in the global map
        if marker_id in self.marker_handler.observed_arucos:
            marker_position_world = self.marker_handler.get_marker_position(marker_id)      # [ [x_world], [y_world] ]

            # The SE(2) inverse: from known world -> marker and measured robot -> marker, solve the robot pose
            dx, dy = marker_position_robot_frame[0, 0], marker_position_robot_frame[1, 0]
            mx, my = marker_position_world[0, 0], marker_position_world[1, 0]

            # Estimate robot global pose
            theta_est = self.state[2]           # Keeping odometry theta

            x_robot = mx - (np.cos(theta_est) * dx - np.sin(theta_est) * dy)
            y_robot = my - (np.sin(theta_est) * dx + np.cos(theta_est) * dy)

            # Update robot's pose
            self.state[0] = x_robot
            self.state[1] = y_robot
            self.state[2] = self.yaw_marker

            rospy.logwarn(f"Robot pose corrected using Marker {marker_id}: ({x_robot:.2f}, {y_robot:.2f})") 
            
            # Compare with the robot predicted state with the ground truth odometry
            # error = np.sqrt((self.x_gt - self.state[0])**2 + (self.y_gt - self.state[1])**2)
            # rospy.loginfo(f"Localization error: {error:.3f} meters")     
            rospy.loginfo("--------------------------------------------------")


if __name__ == "__main__":
    try:
        node = arucoDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
        
