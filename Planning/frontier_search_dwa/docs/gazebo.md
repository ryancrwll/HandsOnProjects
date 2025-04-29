### Gazebo Simulator

To install Gazebo follow these instructions (from [gazebosim](http://gazebosim.org/tutorials?tut=install_ubuntu&cat=install)):

```bash
# Add a new repository
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'

# Set up the keys
wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -

# Update debian database
sudo apt update

# Install Gazebo
sudo apt install gazebo11

# Check that gazebo is correctly installed
gazebo
```

### Install gazebo ROS packages

Install the following package.

```bash
sudo apt install ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control
```

You can run the following command to check if the installation was successful:

```bash
roslaunch gazebo_ros empty_world.launch
```