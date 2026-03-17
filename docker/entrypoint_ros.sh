#!/bin/bash
set -e

source /opt/ros/noetic/setup.bash

# Source workspace overlay if already built
if [ -f /ros_ws/devel/setup.bash ]; then
    source /ros_ws/devel/setup.bash
fi

echo "Container tfg-ros started."
echo "ROS_DISTRO: $ROS_DISTRO"

exec "$@"
