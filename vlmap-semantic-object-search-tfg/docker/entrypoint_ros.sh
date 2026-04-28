#!/usr/bin/env bash
# Entrypoint for the tfg-ros container. Sources the ROS Noetic underlay
# and the workspace overlay (if it has been built already) before
# delegating to the supplied command. Honours the ROS_MODE env var:
#
#   ROS_MODE=shell  (default) — drop into an interactive bash session.
#   ROS_MODE=core            — start `roscore` in the foreground.
#   ROS_MODE=launch:<file>   — `roslaunch` the given file (vlmap_bringup).
#
# Sprint 2 will replace the launch shorthand with a richer dispatcher.

set -euo pipefail

source /opt/ros/noetic/setup.bash

if [[ -f /ros_ws/devel/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /ros_ws/devel/setup.bash
fi

mode="${ROS_MODE:-shell}"
case "${mode}" in
    core)
        exec roscore
        ;;
    launch:*)
        target="${mode#launch:}"
        exec roslaunch "${target}"
        ;;
    shell|*)
        exec "$@"
        ;;
esac
