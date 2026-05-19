cmake_minimum_required(VERSION 3.0.2)
project(vlmap_semantic_server)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  vlmap_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy std_msgs vlmap_msgs
)

catkin_install_python(PROGRAMS
  scripts/vlmap_semantic_server_node
  scripts/publish_room_context.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
