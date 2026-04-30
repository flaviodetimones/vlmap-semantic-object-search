execute_process(COMMAND "/ros_ws/build/yoloe_verifier/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/ros_ws/build/yoloe_verifier/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
