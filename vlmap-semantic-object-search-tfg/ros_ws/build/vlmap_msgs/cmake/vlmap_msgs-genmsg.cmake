# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "vlmap_msgs: 1 messages, 2 services")

set(MSG_I_FLAGS "-Ivlmap_msgs:/ros_ws/src/vlmap_msgs/msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(vlmap_msgs_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/ros_ws/src/vlmap_msgs/msg/SemanticGoal.msg" NAME_WE)
add_custom_target(_vlmap_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "vlmap_msgs" "/ros_ws/src/vlmap_msgs/msg/SemanticGoal.msg" "geometry_msgs/Pose:geometry_msgs/PoseStamped:geometry_msgs/Quaternion:std_msgs/Header:geometry_msgs/Point"
)

get_filename_component(_filename "/ros_ws/src/vlmap_msgs/srv/QueryRoom.srv" NAME_WE)
add_custom_target(_vlmap_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "vlmap_msgs" "/ros_ws/src/vlmap_msgs/srv/QueryRoom.srv" ""
)

get_filename_component(_filename "/ros_ws/src/vlmap_msgs/srv/ResolveRoom.srv" NAME_WE)
add_custom_target(_vlmap_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "vlmap_msgs" "/ros_ws/src/vlmap_msgs/srv/ResolveRoom.srv" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(vlmap_msgs
  "/ros_ws/src/vlmap_msgs/msg/SemanticGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseStamped.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/vlmap_msgs
)

### Generating Services
_generate_srv_cpp(vlmap_msgs
  "/ros_ws/src/vlmap_msgs/srv/QueryRoom.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/vlmap_msgs
)
_generate_srv_cpp(vlmap_msgs
  "/ros_ws/src/vlmap_msgs/srv/ResolveRoom.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/vlmap_msgs
)

### Generating Module File
_generate_module_cpp(vlmap_msgs
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/vlmap_msgs
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(vlmap_msgs_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(vlmap_msgs_generate_messages vlmap_msgs_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/ros_ws/src/vlmap_msgs/msg/SemanticGoal.msg" NAME_WE)
add_dependencies(vlmap_msgs_generate_messages_cpp _vlmap_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/ros_ws/src/vlmap_msgs/srv/QueryRoom.srv" NAME_WE)
add_dependencies(vlmap_msgs_generate_messages_cpp _vlmap_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/ros_ws/src/vlmap_msgs/srv/ResolveRoom.srv" NAME_WE)
add_dependencies(vlmap_msgs_generate_messages_cpp _vlmap_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(vlmap_msgs_gencpp)
add_dependencies(vlmap_msgs_gencpp vlmap_msgs_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS vlmap_msgs_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(vlmap_msgs
  "/ros_ws/src/vlmap_msgs/msg/SemanticGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseStamped.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/vlmap_msgs
)

### Generating Services
_generate_srv_eus(vlmap_msgs
  "/ros_ws/src/vlmap_msgs/srv/QueryRoom.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/vlmap_msgs
)
_generate_srv_eus(vlmap_msgs
  "/ros_ws/src/vlmap_msgs/srv/ResolveRoom.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/vlmap_msgs
)

### Generating Module File
_generate_module_eus(vlmap_msgs
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/vlmap_msgs
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(vlmap_msgs_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(vlmap_msgs_generate_messages vlmap_msgs_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/ros_ws/src/vlmap_msgs/msg/SemanticGoal.msg" NAME_WE)
add_dependencies(vlmap_msgs_generate_messages_eus _vlmap_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/ros_ws/src/vlmap_msgs/srv/QueryRoom.srv" NAME_WE)
add_dependencies(vlmap_msgs_generate_messages_eus _vlmap_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/ros_ws/src/vlmap_msgs/srv/ResolveRoom.srv" NAME_WE)
add_dependencies(vlmap_msgs_generate_messages_eus _vlmap_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(vlmap_msgs_geneus)
add_dependencies(vlmap_msgs_geneus vlmap_msgs_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS vlmap_msgs_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(vlmap_msgs
  "/ros_ws/src/vlmap_msgs/msg/SemanticGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseStamped.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/vlmap_msgs
)

### Generating Services
_generate_srv_lisp(vlmap_msgs
  "/ros_ws/src/vlmap_msgs/srv/QueryRoom.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/vlmap_msgs
)
_generate_srv_lisp(vlmap_msgs
  "/ros_ws/src/vlmap_msgs/srv/ResolveRoom.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/vlmap_msgs
)

### Generating Module File
_generate_module_lisp(vlmap_msgs
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/vlmap_msgs
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(vlmap_msgs_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(vlmap_msgs_generate_messages vlmap_msgs_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/ros_ws/src/vlmap_msgs/msg/SemanticGoal.msg" NAME_WE)
add_dependencies(vlmap_msgs_generate_messages_lisp _vlmap_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/ros_ws/src/vlmap_msgs/srv/QueryRoom.srv" NAME_WE)
add_dependencies(vlmap_msgs_generate_messages_lisp _vlmap_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/ros_ws/src/vlmap_msgs/srv/ResolveRoom.srv" NAME_WE)
add_dependencies(vlmap_msgs_generate_messages_lisp _vlmap_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(vlmap_msgs_genlisp)
add_dependencies(vlmap_msgs_genlisp vlmap_msgs_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS vlmap_msgs_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(vlmap_msgs
  "/ros_ws/src/vlmap_msgs/msg/SemanticGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseStamped.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/vlmap_msgs
)

### Generating Services
_generate_srv_nodejs(vlmap_msgs
  "/ros_ws/src/vlmap_msgs/srv/QueryRoom.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/vlmap_msgs
)
_generate_srv_nodejs(vlmap_msgs
  "/ros_ws/src/vlmap_msgs/srv/ResolveRoom.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/vlmap_msgs
)

### Generating Module File
_generate_module_nodejs(vlmap_msgs
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/vlmap_msgs
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(vlmap_msgs_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(vlmap_msgs_generate_messages vlmap_msgs_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/ros_ws/src/vlmap_msgs/msg/SemanticGoal.msg" NAME_WE)
add_dependencies(vlmap_msgs_generate_messages_nodejs _vlmap_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/ros_ws/src/vlmap_msgs/srv/QueryRoom.srv" NAME_WE)
add_dependencies(vlmap_msgs_generate_messages_nodejs _vlmap_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/ros_ws/src/vlmap_msgs/srv/ResolveRoom.srv" NAME_WE)
add_dependencies(vlmap_msgs_generate_messages_nodejs _vlmap_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(vlmap_msgs_gennodejs)
add_dependencies(vlmap_msgs_gennodejs vlmap_msgs_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS vlmap_msgs_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(vlmap_msgs
  "/ros_ws/src/vlmap_msgs/msg/SemanticGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseStamped.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/vlmap_msgs
)

### Generating Services
_generate_srv_py(vlmap_msgs
  "/ros_ws/src/vlmap_msgs/srv/QueryRoom.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/vlmap_msgs
)
_generate_srv_py(vlmap_msgs
  "/ros_ws/src/vlmap_msgs/srv/ResolveRoom.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/vlmap_msgs
)

### Generating Module File
_generate_module_py(vlmap_msgs
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/vlmap_msgs
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(vlmap_msgs_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(vlmap_msgs_generate_messages vlmap_msgs_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/ros_ws/src/vlmap_msgs/msg/SemanticGoal.msg" NAME_WE)
add_dependencies(vlmap_msgs_generate_messages_py _vlmap_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/ros_ws/src/vlmap_msgs/srv/QueryRoom.srv" NAME_WE)
add_dependencies(vlmap_msgs_generate_messages_py _vlmap_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/ros_ws/src/vlmap_msgs/srv/ResolveRoom.srv" NAME_WE)
add_dependencies(vlmap_msgs_generate_messages_py _vlmap_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(vlmap_msgs_genpy)
add_dependencies(vlmap_msgs_genpy vlmap_msgs_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS vlmap_msgs_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/vlmap_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/vlmap_msgs
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(vlmap_msgs_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(vlmap_msgs_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/vlmap_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/vlmap_msgs
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(vlmap_msgs_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(vlmap_msgs_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/vlmap_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/vlmap_msgs
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(vlmap_msgs_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(vlmap_msgs_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/vlmap_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/vlmap_msgs
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(vlmap_msgs_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(vlmap_msgs_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/vlmap_msgs)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/vlmap_msgs\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/vlmap_msgs
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(vlmap_msgs_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(vlmap_msgs_generate_messages_py geometry_msgs_generate_messages_py)
endif()
