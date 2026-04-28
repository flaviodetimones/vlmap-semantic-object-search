cmake_minimum_required(VERSION 3.0.2)
project(habitat_ros_bridge)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
  sensor_msgs
  geometry_msgs
  tf2_ros
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy nav_msgs sensor_msgs geometry_msgs tf2_ros
)

catkin_install_python(PROGRAMS
  scripts/habitat_ros_bridge_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
