cmake_minimum_required(VERSION 3.0.2)
project(vlmap_capture)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  std_srvs
  sensor_msgs
  geometry_msgs
  tf2_ros
  cv_bridge
  message_filters
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy std_msgs std_srvs sensor_msgs geometry_msgs tf2_ros cv_bridge message_filters
)

catkin_install_python(PROGRAMS
  scripts/vlmap_capture_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
