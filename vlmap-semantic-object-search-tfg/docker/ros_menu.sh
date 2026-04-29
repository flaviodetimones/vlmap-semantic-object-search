#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=/tmp/tfg_ros_menu_logs
mkdir -p "$LOG_DIR"

ros_env() {
  source /opt/ros/noetic/setup.bash
  if [[ -f /ros_ws/devel/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /ros_ws/devel/setup.bash
  fi
}

show_host_start_hint() {
  echo ""
  echo "  Host-side start:"
  echo "    cd /home/mario/tfg/vlmap-semantic-object-search-tfg/docker"
  echo "    docker compose up -d tfg-ros"
  echo "    docker exec -it tfg-ros menu"
}

run_in_ros_shell() {
  local cmd="$1"
  bash -lc "source /opt/ros/noetic/setup.bash; if [[ -f /ros_ws/devel/setup.bash ]]; then source /ros_ws/devel/setup.bash; fi; ${cmd}"
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

pick_rviz_arg() {
  if ask_yes_no "Open RViz too?" "n"; then
    echo "use_rviz:=true"
  else
    echo "use_rviz:=false"
  fi
}

pick_rosbridge_arg() {
  if ask_yes_no "Enable rosbridge?" "n"; then
    echo "use_rosbridge:=true"
  else
    echo "use_rosbridge:=false"
  fi
}

launch_detached() {
  local name="$1"
  local command="$2"
  local log_path="${LOG_DIR}/${name}.log"
  local pid_path="${LOG_DIR}/${name}.pid"
  echo ""
  echo "► Launching $name in background..."
  pkill -f "roslaunch vlmap_bringup" >/dev/null 2>&1 || true
  pkill -f gzserver >/dev/null 2>&1 || true
  pkill -f gzclient >/dev/null 2>&1 || true
  nohup bash -lc "source /opt/ros/noetic/setup.bash; if [[ -f /ros_ws/devel/setup.bash ]]; then source /ros_ws/devel/setup.bash; fi; ${command}" >"$log_path" 2>&1 &
  echo $! >"$pid_path"
  sleep 1
  echo "  PID: $(cat "$pid_path")"
  echo "  Log: tail -f $log_path"
}

tail_latest_log() {
  local latest
  latest="$(ls -1t "$LOG_DIR"/hsr_*.log 2>/dev/null | head -1 || true)"
  if [[ -z "$latest" ]]; then
    echo "  No ROS launch logs found yet."
    return 0
  fi
  tail -f "$latest"
}

stop_active_stack() {
  echo ""
  echo "► Stopping active ROS/Gazebo processes..."
  pkill -f "roslaunch vlmap_bringup" >/dev/null 2>&1 || true
  pkill -f "rosrun rviz rviz" >/dev/null 2>&1 || true
  pkill -f gzserver >/dev/null 2>&1 || true
  pkill -f gzclient >/dev/null 2>&1 || true
  pkill -f "hsr_keyboard_teleop.py" >/dev/null 2>&1 || true
  rm -f "$LOG_DIR"/*.pid
}

show_status() {
  echo ""
  echo "► Process status:"
  pgrep -af "roslaunch vlmap_bringup|gzserver|gzclient|rviz|hsr_keyboard_teleop.py" || echo "  No active ROS/Gazebo processes."
}

run_ros_menu() {
  while true; do
    echo ""
    echo "  ┌─────────────────────────────────────────────────────┐"
    echo "  │                 ROS / Gazebo menu                  │"
    echo "  ├─────────────────────────────────────────────────────┤"
    echo "  │  1) Show host-side start commands                  │"
    echo "  │  2) Build catkin workspace                         │"
    echo "  │  3) Launch HSR official - Gazebo only              │"
    echo "  │  4) Launch HSR official - full stack               │"
    echo "  │  5) Launch HSR proxy - full stack                  │"
    echo "  │  6) Keyboard teleop (/hsrb/command_velocity)       │"
    echo "  │  7) Tail active ROS log                            │"
    echo "  │  8) Stop active ROS/Gazebo processes               │"
    echo "  │  9) Show key HSR topics                            │"
    echo "  │  s) Show running ROS/Gazebo processes              │"
    echo "  │  q) Quit                                           │"
    echo "  └─────────────────────────────────────────────────────┘"
    echo -n "  Select: "
    read -r opt

    case "$opt" in
      1)
        show_host_start_hint
        ;;
      2)
        echo ""
        echo "► Building /ros_ws ..."
        run_in_ros_shell "cd /ros_ws && catkin build"
        ;;
      3)
        rviz_arg="$(pick_rviz_arg)"
        launch_detached \
          "hsr_gazebo_only" \
          "roslaunch vlmap_bringup hsr_gazebo_only.launch gui:=true ${rviz_arg}"
        ;;
      4)
        rviz_arg="$(pick_rviz_arg)"
        rosbridge_arg="$(pick_rosbridge_arg)"
        launch_detached \
          "hsr_full_stack" \
          "roslaunch vlmap_bringup hsr_gazebo_move_base.launch gui:=true ${rviz_arg} ${rosbridge_arg}"
        ;;
      5)
        rviz_arg="$(pick_rviz_arg)"
        rosbridge_arg="$(pick_rosbridge_arg)"
        launch_detached \
          "hsr_proxy_full_stack" \
          "roslaunch vlmap_bringup hsr_proxy_gazebo_move_base.launch gui:=true ${rviz_arg} ${rosbridge_arg}"
        ;;
      6)
        echo ""
        echo "► Starting keyboard teleop. Keys: w/s/a/d, x=stop, q=quit"
        run_in_ros_shell "rosrun vlmap_bringup hsr_keyboard_teleop.py"
        ;;
      7)
        echo ""
        echo "► Tailing most recent ROS launch log..."
        tail_latest_log
        ;;
      8)
        stop_active_stack
        ;;
      9)
        echo ""
        echo "► Key HSR topics:"
        run_in_ros_shell "rostopic list | egrep 'hsrb|move_base|vlmap|scan|image_rect|camera_info|rectified_points' || true"
        ;;
      s|S)
        show_status
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
