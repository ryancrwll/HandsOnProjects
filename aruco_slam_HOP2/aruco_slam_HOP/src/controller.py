#!/usr/bin/env python3

import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import Float64MultiArray
from pynput.keyboard import Key, Listener

# Constraints for wheel speeds
FORWARD_SPEED = 1.0 * 2.5
TURN_SPEED = 0.5 * 2.5
STOP_SPEED = 0.0

def on_press(key):
    '''
        Handle the press of a key to modify wheel velocities based on the command.
    '''
    try:
        if key.char == 'w':         # Move forward
            rospy.set_param('left_wheel', FORWARD_SPEED)
            rospy.set_param('right_wheel', FORWARD_SPEED)

        elif key.char == 'a':       # Turn left
            rospy.set_param('left_wheel', STOP_SPEED)
            rospy.set_param('right_wheel', TURN_SPEED)

        elif key.char == 'd':       # Turn right
            rospy.set_param('left_wheel', TURN_SPEED)
            rospy.set_param('right_wheel', STOP_SPEED)

        elif key.char == 's':       # Stop
            rospy.set_param('left_wheel', STOP_SPEED)
            rospy.set_param('right_wheel', STOP_SPEED)

        elif key.char == 'z':       # Move backward
            rospy.set_param('left_wheel', -FORWARD_SPEED)
            rospy.set_param('right_wheel', -FORWARD_SPEED)

    except AttributeError:
        pass


def vel_move():
    '''
        Continuously move the robot based on teh wheel parameters.
    '''
    pub = rospy.Publisher('/turtlebot/kobuki/commands/wheel_velocities', Float64MultiArray, queue_size=10)
    rate = rospy.Rate(100)      # 100 Hz

    while not rospy.is_shutdown():
        w_L = rospy.get_param('left_wheel')         # Get left wheel velocity
        w_R = rospy.get_param('right_wheel')        # Get right wheel velocity
        move = Float64MultiArray()
        move.data = [w_L, w_R]          # Set the wheel speeds
        pub.publish(move)               # Publish the wheel speeds
        rate.sleep()


if __name__ == '__main__':
    rospy.init_node('velocity_command')
    
    # Initialize wheel parameters
    rospy.set_param('left_wheel', STOP_SPEED)
    rospy.set_param('right_wheel', STOP_SPEED)

    # Start the keyboard listener
    listener = Listener(on_press=on_press)
    # Start the keyboard listener in a separate background thread
    listener.start()        

    # Start the robot movement function
    vel_move()
    listener.join()
    
    rospy.spin()
