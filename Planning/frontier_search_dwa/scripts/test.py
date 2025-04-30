import rospy
from frontier_search_dwa.msg import dwa
from geometry_msgs.msg import PoseStamped

# Create message instance
msg = dwa()

# Set boolean field
msg.replan_bool = True

# Set pose array
pose = PoseStamped()
pose.header.frame_id = "map"
msg.poses.append(pose)

print("Message created successfully!")
print("Boolean field:", msg.replan_bool)
print("Pose array length:", len(msg.poses))