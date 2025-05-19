cmake_minimum_required(VERSION 2.8.3)
project(hands_on_intervention)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

## Declare your service files
add_service_files(
  FILES
  intervention.srv
)

## Generate added messages and services
generate_messages(
  DEPENDENCIES
  std_msgs
)

## Declare the catkin package
catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs
)

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)

