#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped

def publish_goal(x, y, z):
    pub = rospy.Publisher("/desired_goal", PoseStamped, queue_size=10)
    rospy.sleep(0.5)

    goal = PoseStamped()
    goal.header.frame_id = "swiftpro/manipulator_base_link"
    goal.header.stamp = rospy.Time.now()
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.position.z = z
    goal.pose.orientation.w = 1.0  # Identity quaternion

    pub.publish(goal)
    rospy.loginfo(f"[Sent] Goal published: x={x}, y={y}, z={z}")


def main():
    rospy.init_node("interactive_goal_sender", anonymous=True)

    while not rospy.is_shutdown():
        try:
            print("\nEnter new goal position:")
            x = float(input("  x: "))
            y = float(input("  y: "))
            z = float(input("  z: "))
            publish_goal(x, y, z)
        except ValueError:
            print("[Error] Please enter valid numbers.")
        except rospy.ROSInterruptException:
            break


if __name__ == "__main__":
    main()