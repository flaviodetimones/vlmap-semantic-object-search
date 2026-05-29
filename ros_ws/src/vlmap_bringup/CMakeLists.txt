cmake_minimum_required(VERSION 3.0.2)
project(vlmap_bringup)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_package(
  CATKIN_DEPENDS rospy
)

catkin_install_python(PROGRAMS
  scripts/hsr_capture_patrol.py
  scripts/hsr_cmd_vel_smoother.py
  scripts/hsr_keyboard_teleop.py
  scripts/hsr_kinematic_base_controller.py
  scripts/hsr_passive_joint_state_publisher.py
  scripts/hsr_pose_scout.py
  scripts/hsr_posture_keeper.py
  scripts/hsr_sensor_visualizer.py
  scripts/hsr_stack_check
  scripts/mock_move_base_node
  scripts/ros_contract_smoke_test.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch rviz config urdf worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
