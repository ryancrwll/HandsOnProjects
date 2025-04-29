# TurtleBot in Stonefish
If you are going to work in the Stonefish environment with Stonefish_ros, follow these steps:

#### Clone and build 
- Mobile base description [Kobuki](https://bitbucket.org/udg_cirs/kobuki_description/src/master/)
- Manipulator description [SwiftPro](https://bitbucket.org/udg_cirs/swiftpro_description/src/master/)
- Mobile base + Manipulator description [Turtlebot](https://bitbucket.org/udg_cirs/turtlebot_description/src/master/)
- TurtleBot simulation environment [turtlebot_simulation](https://bitbucket.org/udg_cirs/turtlebot_simulation/src/master/)

#### ROS dependencies 
Additional needed ROS packages:
```bash
sudo apt install ros-noetic-realsense2-description # (Realsense camera)
sudo apt install ros-noetic-ros-control ros-noetic-ros-controllers
sudo apt install ros-noetic-xacro
```