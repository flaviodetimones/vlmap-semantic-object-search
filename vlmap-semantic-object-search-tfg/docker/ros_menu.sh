#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker/docker-compose.yml"

ensure_ros_container() {
  docker compose -f "$COMPOSE_FILE" up -d tfg-ros >/dev/null
}

ask_yes_no() {
  local prompt="$1"
  local default="${2:-n}"
  local ans
  echo -n "  $prompt [y/N]: "
  read -r ans
  ans=${ans:-$default}
  [[ "$ans" =~ ^[Yy]$ ]]
}

launch_detached() {
  local name="$1"
  local command="$2"
  echo ""
  echo "► Launching $name in background..."
  docker exec tfg-ros bash -lc "pkill -f 'roslaunch vlmap_bringup' >/dev/null 2>&1 || true"
  docker exec tfg-ros bash -lc "nohup $command >/tmp/${name}.log 2>&1 &"
  echo "  Log: docker exec -it tfg-ros bash -lc 'tail -f /tmp/${name}.log'"
}

run_ros_menu() {
  while true; do
    echo ""
    echo "  ┌─────────────────────────────────────────────────────┐"
    echo "  │                 ROS / Gazebo menu                  │"
    echo "  ├─────────────────────────────────────────────────────┤"
    echo "  │  1) Start / refresh tfg-ros container              │"
    echo "  │  2) Build catkin workspace                         │"
    echo "  │  3) Launch HSR official - Gazebo only              │"
    echo "  │  4) Launch HSR official - full stack               │"
    echo "  │  5) Launch HSR proxy - full stack                  │"
    echo "  │  6) Keyboard teleop (/hsrb/command_velocity)       │"
    echo "  │  7) Tail active ROS log                            │"
    echo "  │  8) Stop active ROS/Gazebo processes               │"
    echo "  │  9) Show key HSR topics                            │"
    echo "  │  q) Quit                                           │"
    echo "  └─────────────────────────────────────────────────────┘"
    echo -n "  Select: "
    read -r opt

    case "$opt" in
      1)
        echo ""
        echo "► Starting tfg-ros..."
        docker compose -f "$COMPOSE_FILE" up -d tfg-ros
        ;;
      2)
        ensure_ros_container
        echo ""
        echo "► Building /ros_ws ..."
        docker exec -it tfg-ros bash -lc "cd /ros_ws && catkin build"
        ;;
      3)
        ensure_ros_container
        rviz_arg="use_rviz:=false"
        if ask_yes_no "Open RViz too?" "n"; then
          rviz_arg="use_rviz:=true"
        fi
        launch_detached \
          "hsr_gazebo_only" \
          "roslaunch vlmap_bringup hsr_gazebo_only.launch gui:=true ${rviz_arg}"
        ;;
      4)
        ensure_ros_container
        rviz_arg="use_rviz:=false"
        rosbridge_arg="use_rosbridge:=false"
        if ask_yes_no "Open RViz too?" "n"; then
          rviz_arg="use_rviz:=true"
        fi
        if ask_yes_no "Enable rosbridge?" "n"; then
          rosbridge_arg="use_rosbridge:=true"
        fi
        launch_detached \
          "hsr_full_stack" \
          "roslaunch vlmap_bringup hsr_gazebo_move_base.launch gui:=true ${rviz_arg} ${rosbridge_arg}"
        ;;
      5)
        ensure_ros_container
        rviz_arg="use_rviz:=false"
        rosbridge_arg="use_rosbridge:=false"
        if ask_yes_no "Open RViz too?" "n"; then
          rviz_arg="use_rviz:=true"
        fi
        if ask_yes_no "Enable rosbridge?" "n"; then
          rosbridge_arg="use_rosbridge:=true"
        fi
        launch_detached \
          "hsr_proxy_full_stack" \
          "roslaunch vlmap_bringup hsr_proxy_gazebo_move_base.launch gui:=true ${rviz_arg} ${rosbridge_arg}"
        ;;
      6)
        ensure_ros_container
        echo ""
        echo "► Starting keyboard teleop. Keys: w/s/a/d, x=stop, q=quit"
        docker exec -it tfg-ros bash -lc "rosrun vlmap_bringup hsr_keyboard_teleop.py"
        ;;
      7)
        ensure_ros_container
        echo ""
        echo "► Tailing most recent ROS launch log..."
        docker exec -it tfg-ros bash -lc "ls -1t /tmp/hsr_*.log 2>/dev/null | head -1 | xargs -r tail -f"
        ;;
      8)
        ensure_ros_container
        echo ""
        echo "► Stopping active ROS/Gazebo processes..."
        docker exec tfg-ros bash -lc "pkill -f 'roslaunch vlmap_bringup' >/dev/null 2>&1 || true; pkill -f gzserver >/dev/null 2>&1 || true; pkill -f gzclient >/dev/null 2>&1 || true"
        ;;
      9)
        ensure_ros_container
        echo ""
        echo "► Key HSR topics:"
        docker exec -it tfg-ros bash -lc "rostopic list | egrep 'hsrb|move_base|vlmap|scan|image_rect|camera_info|rectified_points' || true"
        ;;
      q|Q)
        break
        ;;
      *)
        echo "  Invalid option."
        ;;
    esac
  done
}

run_ros_menu
