#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import math
from sensor_msgs.msg import Image

class ArucoDetector:
    def __init__(self):
        # Publisher to send detected ArUco marker positions
        self.aruco_pub = rospy.Publisher("/aruco_position", Float64MultiArray, queue_size=10)
        
        # Subscriber to receive images from the camera
        # self.aruco_sub = rospy.Subscriber("/turtlebot/kobuki/sensors/realsense/color/image_color", Image, self.image_callback)
        self.aruco_sub = rospy.Subscriber("/turtlebot/kobuki/realsense/color/image_color", Image, self.image_callback)
        
        self.bridge = CvBridge()

        # Camera position with respect to the robot
        self.camera_x = 0.122
        self.camera_y = -0.033
        self.camera_z = 0.082
        self.camera_roll = math.pi / 2
        self.camera_pitch = 0.0
        self.camera_yaw = math.pi / 2

        self.camera_pose = np.array([self.camera_x, self.camera_y, self.camera_z, self.camera_roll, self.camera_pitch, self.camera_yaw])

        # Define dictionaries to try
        self.dictionaries = {
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000
        }

    def transform_camera_to_robot(self, x, y, z, roll, pitch, yaw):
        Transf = np.eye(4)
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(roll), -math.sin(roll)],
                       [0, math.sin(roll), math.cos(roll)]])
        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                       [0, 1, 0],
                       [-math.sin(pitch), 0, math.cos(pitch)]])
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                       [math.sin(yaw), math.cos(yaw), 0],
                       [0, 0, 1]])
        R = Rz @ Rx @ Ry
        Trans = np.array([x, y, z]).reshape(3, 1)
        Transf[0:3, 0:3] = R
        Transf[0:3, 3] = np.squeeze(Trans)
        return Trans, R, Transf

    def image_callback(self, Image_msg):
        print(" Received image, running detection...")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(Image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        marker_length = 0.05
        camera_matrix = np.array([[1396.8086675255468, 0.0, 960.0],
                                  [0.0, 1396.8086675255468, 540.0],
                                  [0.0, 0.0, 1.0]])
        dist_coeffs = np.zeros((5,))

        detection_found = False  # Flag to check if any marker was detected

        for dict_name, dict_type in self.dictionaries.items():
            dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(cv_image, dictionary)

            if marker_ids is not None:
                print(f"Detected markers using {dict_name}: {marker_ids.flatten()}")
                detection_found = True

                for i in range(len(marker_ids)):
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners[i], marker_length, camera_matrix, dist_coeffs)
                    rvecs = rvecs[0, :].reshape(1, 3)
                    tvecs = tvecs[0, :].reshape(1, 3)
                    cv2.aruco.drawDetectedMarkers(cv_image, marker_corners, marker_ids, (0, 255, 0))
                    cv2.drawFrameAxes(cv_image, camera_matrix, dist_coeffs, rvecs, tvecs, 0.05)

                    Rot, _ = cv2.Rodrigues(rvecs)
                    Transf = np.eye(4)
                    Transf[0:3, 0:3] = Rot
                    Transf[0:3, 3] = np.squeeze(tvecs)

                    Trans_r_c, rot_r_c, Transf_r_c = self.transform_camera_to_robot(
                        self.camera_pose[0], self.camera_pose[1], self.camera_pose[2],
                        self.camera_pose[3], self.camera_pose[4], self.camera_pose[5]
                    )
                    Transf_r = Transf_r_c @ Transf
                    x = float(Transf_r[0, 3])
                    y = float(Transf_r[1, 3])
                    z = float(Transf_r[2, 3])

                    point_msg = Float64MultiArray()
                    point_msg.data = [x, y, z, 0.0]
                    self.aruco_pub.publish(point_msg)
                    print(f" Published marker position: x={x:.2f}, y={y:.2f}, z={z:.2f}")

                    # OPTIONAL: Uncomment to stop after first detection
                    rospy.signal_shutdown("ArUco marker detected")

        if not detection_found:
            print(" No markers detected in this frame with any dictionary.")

        cv2.imshow("Camera", cv_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        rospy.init_node("aruco_detector")
        ArucoDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

