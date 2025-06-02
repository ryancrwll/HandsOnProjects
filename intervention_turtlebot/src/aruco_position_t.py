#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion, Twist
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import quaternion_from_matrix
import tf2_ros
import cv2.aruco as aruco


class ArucoDetector:
    def __init__(self):
        self.bridge = CvBridge()

        # Publisher to send detected ArUco marker positions
        self.aruco_pose_pub = rospy.Publisher('/aruco_position', PoseStamped, queue_size=10)
        self.aruco_rviz_marker_pub = rospy.Publisher('/aruco_marker', MarkerArray, queue_size=10)
        
        # Subscriber to receive images from the camera
        # self.aruco_sub = rospy.Subscriber("/turtlebot/kobuki/realsense/color/image_color", Image, self.image_callback)
        self.aruco_sub = rospy.Subscriber("/turtlebot/kobuki/realsense/color/image_raw", Image, self.image_callback)
        self.camera_info_sub = rospy.Subscriber('/turtlebot/kobuki/realsense/color/camera_info', CameraInfo, self.camera_info_callback)
        self.marker_length = 0.05
        self.detected_order = []
        self.aruco_dict_type = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        # self.aruco_dict_type = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
        
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame = None
        self.camera_ready = False
        self.frame_id = "odom"

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.initialized_world = False

        # 3D corner model for solvePnP
        h = self.marker_length / 2.0
        self.marker_obj_pts = np.array([
                                        [-h, h, 0],
                                        [ h, h, 0],
                                        [ h,-h, 0],
                                        [-h,-h, 0]], dtype=np.float32)
        

    def camera_info_callback(self, msg):
        rospy.sleep(0.5)
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.dist_coeffs = np.array(msg.D)
        self.camera_frame = msg.header.frame_id
        self.camera_ready = True
        self.camera_info_sub.unregister()
        rospy.loginfo(f"Camera calibration received. Using frame: {self.camera_frame}")
        rospy.loginfo(self.camera_matrix)


    def image_callback(self, msg):
        if not self.camera_ready:
            rospy.logwarn_throttle(5.0, "Waiting for camera calibration...")
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = aruco.ArucoDetector(self.aruco_dict_type, aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            print('no id')
            return

        for i, marker_id in enumerate(ids.flatten()):
            image_points = corners[i][0].astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(self.marker_obj_pts, image_points,
                                               self.camera_matrix, self.dist_coeffs,
                                               flags=cv2.SOLVEPNP_IPPE_SQUARE)

            if not success:
                rospy.logwarn(f"solvePnP failed for marker {marker_id}")
                continue

            # Broadcast transform for this marker
            t = TransformStamped()
            t.header.stamp = msg.header.stamp
            t.header.frame_id = self.camera_frame
            t.child_frame_id = f"aruco_marker_{marker_id}"
            t.transform.translation.x = tvec[0][0]
            t.transform.translation.y = tvec[1][0]
            t.transform.translation.z = tvec[2][0]

            rot_mat = cv2.Rodrigues(rvec)[0]
            T = np.eye(4)
            T[:3, :3] = rot_mat
            q = quaternion_from_matrix(T)
            t.transform.rotation.x, t.transform.rotation.y = q[0], q[1]
            t.transform.rotation.z, t.transform.rotation.w = q[2], q[3]
            self.tf_broadcaster.sendTransform(t)

            # Now lookup from world_ned to aruco_marker_X
            try:
                marker_frame = t.child_frame_id
                transform = self.tf_buffer.lookup_transform(self.frame_id, marker_frame, rospy.Time(0), rospy.Duration(0.5))

                pose_world = PoseStamped()
                pose_world.header.stamp = transform.header.stamp
                pose_world.header.frame_id = self.frame_id
                pose_world.pose.position.x = transform.transform.translation.x
                pose_world.pose.position.y = transform.transform.translation.y
                pose_world.pose.position.z = transform.transform.translation.z
                pose_world.pose.orientation = transform.transform.rotation

                # rospy.logwarn(f"Pose_world: {pose_world}")
                self.aruco_pose_pub.publish(pose_world)
                self.publish_marker(pose_world, marker_id)
                # OPTIONAL: Uncomment to stop after first detection
                rospy.signal_shutdown("ArUco marker detected")

            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"TF lookup failed for marker {marker_id}: {e}")


    def publish_marker(self, pose_world, marker_id):
        marker = Marker()
        marker.header.frame_id = pose_world.header.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "aruco_cube"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = pose_world.pose.position.x
        marker.pose.position.y = pose_world.pose.position.y
        marker.pose.position.z = -0.1
        marker.pose.orientation = pose_world.pose.orientation

        marker.scale.x = marker.scale.y = marker.scale.z = 0.2
        marker.color.r = marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        text_marker = Marker()
        text_marker.header.frame_id = pose_world.header.frame_id
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = "aruco_label"
        text_marker.id = marker_id + 1000
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        text_marker.pose.position.x = pose_world.pose.position.x
        text_marker.pose.position.y = pose_world.pose.position.y
        text_marker.pose.position.z = -0.5
        text_marker.pose.orientation = pose_world.pose.orientation

        text_marker.scale.z = 0.2
        text_marker.color.r = text_marker.color.g = text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.text = f"ArUco {marker_id}"

        self.aruco_rviz_marker_pub.publish(MarkerArray(markers=[marker, text_marker]))


if __name__ == '__main__':
    try:
        rospy.init_node("aruco_detector")
        ArucoDetector()
        print("hi")
        rospy.spin()

    except rospy.ROSInterruptException:
        pass