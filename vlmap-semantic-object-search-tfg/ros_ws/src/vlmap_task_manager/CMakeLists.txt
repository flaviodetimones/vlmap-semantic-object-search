cmake_minimum_required(VERSION 3.0.2)
project(vlmap_task_manager)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  move_base_msgs
  vlmap_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs move_base_msgs vlmap_msgs
)

catkin_install_python(PROGRAMS
  scripts/vlmap_task_manager_node
  scripts/vlmap_navigation_result_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
