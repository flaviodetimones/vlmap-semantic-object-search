cmake_minimum_required(VERSION 3.0.2)
project(vlmap_bringup)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_package(
  CATKIN_DEPENDS rospy
)

catkin_install_python(PROGRAMS
  scripts/hsr_keyboard_teleop.py
  scripts/hsr_posture_keeper.py
  scripts/hsr_stack_check
  scripts/mock_move_base_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch rviz config urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
