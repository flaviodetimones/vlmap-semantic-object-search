cmake_minimum_required(VERSION 3.0.2)
project(yoloe_verifier)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs std_msgs cv_bridge
)

catkin_install_python(PROGRAMS
  scripts/yoloe_verifier_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
