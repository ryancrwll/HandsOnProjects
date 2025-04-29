# On-line path planning

**This project belongs to the Universitat de Girona. It is forbidden to publish this project or any derivative work in any public repository.**

This lab will combine the concepts of occupancy grid mapping and path planning seen in previous sessions.
The main goal of this lab is to move a simulated or real Turtlebot-like robot from its current position to a goal position avoiding the obstacles in the environment.

## Installation

The following element are necessary to run this package. Remember that after installing a new package, you have to source your catkin workspace with `source devel/setup.bash` command.

### Simulation environment
Follow the instructions for the simulator you were instructed to use:

- [Gazebo](https://bitbucket.org/udg_cirs/turtlebot_online_path_planning/src/master/docs/gazebo.md).
- [Stonefish](https://github.com/patrykcieslak/stonefish) and [Stonefish_ros](https://github.com/patrykcieslak/stonefish_ros).

### Robot model
For each simulator we have a diferent robot definition, so please follow the instructions in line with the simulator you are going to use. This has no effect in the code or algorithms we are going work.

- For **Gazebo**, we use the **TurtleBot 3** package stack, [see here for installation process](https://bitbucket.org/udg_cirs/turtlebot_online_path_planning/src/master/docs/turtlebot3.md).
- For **Stonefish**, we use the Kuboki+SwiftPro definitions to make plain **Turtlebot**, [follow these steps](https://bitbucket.org/udg_cirs/turtlebot_online_path_planning/src/master/docs/turtlebot_stonefish.md).

### Octomap Server
The Octomap server is a ROS package that provides a 3D occupancy grid map of the environment. We will use it to obtain the 2D occupancy grid map that will be used by the path planner. To install it, run the following command:

```bash
sudo apt install ros-noetic-octomap*
```

### Check installation
You can run the following command to check if the installation was successful:

```bash
roslaunch frontier_search_dwa gazebo.launch # gazebo users
roslaunch frontier_search_dwa stonefish.launch # stonefish users
```
You should see something like this:

<table>
<thead>
  <tr>
    <td><b>Gazebo Enviroment</td>
    <td><img src="/imgs/turtle_rviz.png" width="200"></td>
    <td><img src="/imgs/turtle_gazebo.png" width="200"></td>
  </tr>
  <tr>
    <td><b>Stonefish Enviroment</td>
    <td><img src="/imgs/stonefish_gridmap.png" width="200"></td>
    <td><img src="/imgs/stonefish_fishstone.png" width="200"></td>
  </tr>
  </thead>
</table>

## Pre-lab

In this lab we are going to implement an on-line path planning algorithm. The first thing you should do is to complete the Notebook [state_validity_checker.ipynb](notebooks/state_validity_checker.ipynb).

## Online path planning architecture

To implement a path planner that can run in an unknown environment in real time several modules are required:

* A map server that publishes the occupancy grid map of the environment that has been observed till this moment.
* A path planner that computes the path from the current position to the goal.
* A controller that moves the robot along the path.

We are going to use the following algorithms for all these nodules:

* Map server: An occupancy grid map, the Octomap server, will be used. Check the documentation [here](https://wiki.ros.org/octomap_server).
* Path planner: Several path planning algorithms (i.e., A*, RRT, ...) have been implemented. Here, you can use RRT, or any similar algorithm.
* Controller: A controller that moves a robot to a specific position was already been implemented in the `turtlesim` lab.

The following figure shows the relations between all these modules.

<div style="text-align: center">
    <img src='./imgs/online_planning_architecture.png' width='600'/>
</div>

As you can see, the center piece is the `planner`, which is responsible for finding a path from the current position to the goal. The connection with the occupancy map is done through the `state validity checker` module that for each configuration examined by the `planner`, it check if there is a collision with the last available `map`. The `controller` takes the list of configurations computed by the `planner` to reach the goal (i.e., the `path`) and it is responsible for moving the robot along them. Every time that the `map server` publishes a new map, the `state validity checker` is also updated and the planned path has to be checked as new obstacles can make it invalid.

## Implementation

Once introduced the main modules that must be implemented, lets see in more detail how to implement each of them. It is very important to make each module ROS agnostic. It means that ROS specific code do not has to be included in the following modules. We will separate our code in a file containing all the functionalities of our modules and another file, a ROS node, that will make use of all these modules. This file (i.e., the ROS node) will contain all ROS related aspects: subscribe and publish all required messages, timers, configuration files, ...

### State validity checker

The Octomap server publishes a 2D occupancy grid (`nav_msgs/OccupancyGrid`) map throug the topic `/projected_map` that is updated every time a new scan is received by the Octomap server. This 2D grid will be used by the `state validity checker` to check if a configuration is valid or not.

The `state validity checker` has to implement at least 3 functions: `set`, `isValid`, and `checkPath`.

#### `set(data, resolution, origin)`: 

This function is called by the ROS node when a new occupancy grid is received. It is used to update the map that will be used to check the state validity.

* `data`: is a `int` array. To transform the `nav_msgs/OccupancyGrid/data` to a 2D matrix it has to be reshaped to `nav_msgs/OccupancyGrid/info/width x nav_msgs/OccupancyGrid/info/height` and transposed
* `resolution` is the resolution of the map (meters per cell).
* `origin` is the position of the origin of the map (in meters).

#### `is_valid(x, y, distance=0.1)`:

This function checks if the position `(x, y)` defined in meters is valid or not. To do it, it is required to transform the position from meters to the cell index that the position belongs to. If `distance` is greater than 0, it is required to check if the cells at distance `distance` from the `(x, y)` position are also free or not. Consider the map `resolution` to see how many cells around `(x, y)` have to be checked. The function returns `True` if all checked cells are free and `False` otherwise.

To transform from Cartesian position in meters ($Cartesian_p$) to cell index $Cell_p$, the following formula is used:

<div style="text-align: center">
    <img src='./imgs/equations/cartesian_eq.png'/>
</div>

$$
 Cell_p = \frac{Cartesian_p - Map_{origin}}{Map_{resolution}}
$$

If the requested position is outside the map, the function must return `True` if the parameter `is_unknown_valid` is defined as `True` and `False` otherwise. If `is_unknown_valid` argument is set to `False`, it will not be possible to send the robot to unknow positions. If it is set to `True`, planning is more difficult because when the robot reaches a position closer to the unknown cells, the map will be updated and the robot will known if the requested position is free or occupied. Therefore, to check if the path is still valid, it will necessary to call the method `check_path`. The algorithm must work with `is_unknown_valid` set at `True` and set a `False`.

#### `check_path(path, min_dist)`:

This function checks if a `path` generated by the path planner is still valid or not after an update in the occupancy map. It is especially important to check that when there are dynamic obstacles or when `is_unknown_valid` is set to `True`. If the `path` is invalid, a new `path` must be computed.

While the path is just a list of configurations (i.e., $x$, $y$ positions), the `check_path` function must check not only if these positions are valid but also that the robot can move along them. A simple solution is to discretize the *segment* between two configurations using the `min_dist` parameter and check if each intermediate configuration is valid using the `is_valid` function.

### Path Planner

The path planner function needs a `start` and a `goal` position (i.e., `x, y` in meters) and access to the `state validity checker` to compute a valid `path` from the `start` to the `goal`. The `start` position can be the current position of the robot.

Only the `compute_path` function must be implemented.

#### `compute_path(start, goal, state_validity_checker, dominion)`:

Using any path planning algorithm (i.e., RRT, RRT*, PRM...) and the state validity checker object, this function computes a valid `path` from the `start` to the `goal`. The `path` is a list of `(x, y)` positions in meters.
To implement it, you should first fill the `RRT` class provided in the skeleton with the code you have already implemented in the previous lab. Feel free to change this task for a `PRM` or `RRT*` implementation.


The path planner will need access to the `state_validity_checker` while planning. It also will need to know the `dominion` of the environment. The `dominion` is a 2D array containing the minimum bound and maximum bound where the planner will search for a solution (i.e., [$min_x$, $max_x$, $min_y$, $max_y$]).


### Controller

Once a `path` is computed, the controller is responsible for moving the robot along it. The controller only has to implement one function that must be called at a constant rate given a valid `path`:

#### `move_to_point(x, y, theta, x_goal, y_goal)`:

The `moveToPoint` function computes the desired forward velocity ($\nu_d$) and desired angular velocity ($w_d$) that the robot must apply to move from the current position ($x$, $y$, $\theta$) to the goal position ($x_{goal}$, $y_goal$). Because we want the robot to stay over the path as much as possible, the controller must compute first the desired angle between the current position and the goal ($\theta_d$) and only when the desired angle is almost equal than the current one (i.e., $\theta_d \approx \theta$) move forward with a velocity proportional to the distance with respect to the goal. When the goal position is reached with some tolerance, this waypoint has to be removed from the `path` and the next configuration in the `path` will become the new goal.

$$
\theta_d = \tan^{-1}\frac{y_g - y}{x_g - x} 
$$

$$
w_d = K_h \text{wrap\_angle}(\theta_d - \theta)
$$

$$
\nu_d = 
\begin{cases}
    K_p \sqrt{(x_g - x)^2 + (y_g - y)^2} & \text{if~} \theta_d \approx \theta \\
    0 & \text{otherwise}
\end{cases}
$$

Where $K_v > 0$ and $K_h > 0$.
