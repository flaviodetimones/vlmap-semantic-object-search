cmake_minimum_required(VERSION 3.0.2)
project(yoloe_verifier)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
  vlmap_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs std_msgs cv_bridge vlmap_msgs
)

catkin_install_python(PROGRAMS
  scripts/yoloe_verifier_node
  scripts/mock_verify_worker.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
