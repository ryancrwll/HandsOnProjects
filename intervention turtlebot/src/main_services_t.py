#!/usr/bin/env python3


import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
from std_msgs.msg import Int32
from hands_on_intervention_t.srv import intervention_t, intervention_tResponse

# This file acts like a messenger between user commands and the robot.
# It provides ROS services that receive structured commands 
# (like goal, task, weight), and forwards them as ROS topics 
# so that main.py can pick them up and act.


# is a helper node that listens for service calls like /goal_server or 
# /weight_server, and forwards the data to topics like /goal_set and 
# /weight_set — which the main controller (main.py) subscribes to and 
# uses for motion control.

class ServiceObject():
    def __init__(self):

        self.weight = None
        self.goal = None

        self.pub_weight = rospy.Publisher('/weight_set', Float64MultiArray,
                                          queue_size=10)
        self.pub_goal = rospy.Publisher('/goal_set', Float64MultiArray, queue_size=10)
        self.pub_aruco = rospy.Publisher('/get_aruco_pose', Point, queue_size=10)

        self.pub_task = rospy.Publisher('/task_set', Int32,  queue_size=10)

        rospy.Service('weight_server', intervention_t, self.weight_service)

        rospy.Service('aruco_server', intervention_t, self.aruco_service)

        rospy.Service('goal_server', intervention_t, self.goal_service)

        rospy.Service('task_server', intervention_t, self.task_service)

    def weight_service(self, msg):
        """
        Callback function for the weight_set service.

        Parameters:
        - msg: The service message containing the weight data.

        Functionality:
        - Sets the weighted_DLS parameter.
        - Publishes the weight data.
        """
        # Extract weight data from the service message
        weight = msg.data
        # Set the parameter for weighted_DLS in the ROS parameter server
        rospy.set_param('weighted_DLS', weight)

        weighted_DLS = Float64MultiArray()
        # Assign the received value to the message
        weighted_DLS.data = weight
        # Publish the weighted DLS
        self.pub_weight.publish(weighted_DLS)
        # Return a service response
        return intervention_tResponse()

    def task_service(self, msg):
        """
        Callback function for the task_set service.

        Parameters:
        - msg: The service message containing the task index.

        Functionality:
        - Sets the task index.
        - Publishes the task data.
        """
        # Convert the received data to integer
        task_index = int(msg.data[0])
        rospy.loginfo('task_index: %d', task_index)

        task_value = Int32()
        # Assign the received value to the message
        task_value.data = task_index
        # Publish the task data
        self.pub_task.publish(task_value)
        return intervention_tResponse()

    def goal_service(self, msg):
        """
        Callback function for the goal_set service.

        Parameters:
        - msg: The service message containing the goal data.

        Functionality:
        - Sets the goal parameter.
        - Publishes the goal data.
        """
        # Extract goal data from the service message
        goal = msg.data
        rospy.loginfo('goal: %s', str(goal))

        # Set the parameter for goal in the ROS parameter server
        rospy.set_param('goal', goal)
        goal_value = Float64MultiArray()
        # Assign the received value to the message
        goal_value.data = goal
        # Publish the goal data
        self.pub_goal.publish(goal_value)
        return intervention_tResponse()

    def aruco_service(self, msg):
        """
        Callback function for the aruco_set service.

        Parameters:
        - msg: The service message for the ArUco service.

        Functionality:
        - Publishes ArUco pose.
        """
        point_msg = Point()
        #Publish the aruco data
        self.pub_aruco.publish(point_msg)
        return intervention_tResponse()

if __name__ == '__main__':
    try:
        rospy.init_node('intervention_t_service')
        ServiceObject()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

