# Turtlebot3
Follow these instructions if you are going to work with the **TurtleBot3** in the GAzebo sim environment.

[TurtleBot 3](https://www.turtlebot.com/) is the third version of ROS standard platform robot. There are 3 versions of robots (i.e., burger, waffle and waffle_pi). There are several ROS package related with turtlebot 3 but all of them can be installed installing only the following meta package:

```bash
sudo apt update
sudo apt install ros-noetic-turtlebot3*
```

You can run the following command to check if the installation was successful:

```bash
export TURTLEBOT3_MODEL="waffle"
roslaunch turtlebot3_gazebo turtlebot3_stage_4.launch
```