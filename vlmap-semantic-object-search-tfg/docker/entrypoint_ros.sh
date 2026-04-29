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

chmod +x /workspace/docker/ros_menu.sh
ln -sf /workspace/docker/ros_menu.sh /usr/local/bin/menu

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                TFG — ROS1 Noetic + Gazebo                      ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  ROS     : $(rosversion -d 2>/dev/null || echo 'unknown')"
echo "║  Workdir : $(pwd)"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Type  'menu'  to see ROS / Gazebo commands                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

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
