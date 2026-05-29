#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=/tmp/tfg_ros_menu_logs
mkdir -p "$LOG_DIR"

run_in_ros_shell() {
  local cmd="$1"
  bash -lc "source /opt/ros/noetic/setup.bash; if [[ -f /ros_ws/devel/setup.bash ]]; then source /ros_ws/devel/setup.bash; fi; ${cmd}"
}

launch_foreground() {
  local name="$1"
  local command="$2"
  local log_path="${LOG_DIR}/${name}.log"
  echo ""
  echo "  ► Live log (also saved to ${log_path})"
  echo "  ► Press Ctrl+C to stop this scene and return to the menu."
  echo ""
  # Use `script -f` so roslaunch sees a real pseudo-tty and keeps
  # line-buffered output (otherwise Python block-buffers when stdout is a
  # pipe and you see nothing live). `script -f` flushes after each write,
  # mirrors output to the terminal AND writes to the log file in one shot.
  # `|| true` so a Ctrl+C doesn't kill the menu via set -e.
  script -qf -c "bash -lc 'source /opt/ros/noetic/setup.bash; if [[ -f /ros_ws/devel/setup.bash ]]; then source /ros_ws/devel/setup.bash; fi; ${command}'" "$log_path" || true
  # When roslaunch dies (Ctrl+C or crash), make sure no orphan Gazebo
  # processes survive before the menu prompts again.
  stop_stack_processes
}

stop_stack_processes() {
  pkill -f "roslaunch vlmap_bringup" >/dev/null 2>&1 || true
  pkill -f "roslaunch aws_robomaker" >/dev/null 2>&1 || true
  pkill -f "roslaunch gazebo_ros" >/dev/null 2>&1 || true
  pkill -f "rosrun rviz rviz" >/dev/null 2>&1 || true
  pkill -f gzserver >/dev/null 2>&1 || true
  pkill -f gzclient >/dev/null 2>&1 || true
  rm -f "$LOG_DIR"/*.pid
}

pick_scene() {
  local ans
  {
    echo ""
    echo "  Select scene:"
    echo "    1) house     — AWS RoboMaker Small House (residencial occidental)"
    echo "    2) wrs       — Toyota WRS 2020 apartment (HSR-native)"
    echo "    3) tb3_house — TurtleBot3 House (apartamento pequeño amueblado)"
    echo "    4) pal_home  — PAL Robotics home (multi-room amueblado: cocina, salón, dormitorio)"
    printf "  Scene [1-4]: "
  } >&2
  read -r ans </dev/tty
  case "${ans:-1}" in
    1) echo "house" ;;
    2) echo "wrs" ;;
    3) echo "tb3_house" ;;
    4) echo "pal_home" ;;
    *) echo "house" ;;
  esac
}

pick_capture_scene() {
  local ans
  {
    echo ""
    echo "  Select scene for VLMap capture:"
    echo "    1) small_house — AWS RoboMaker Small House"
    echo "    2) wrs         — Toyota WRS 2020 apartment"
    echo "    3) tb3_house   — TurtleBot3 House"
    echo "    4) pal_home    — PAL Robotics home"
    printf "  Scene [1-4]: "
  } >&2
  read -r ans </dev/tty
  case "${ans:-1}" in
    1) echo "small_house" ;;
    2) echo "wrs" ;;
    3) echo "tb3_house" ;;
    4) echo "pal_home" ;;
    *) echo "small_house" ;;
  esac
}

prompt_run_id() {
  local ans
  local default_id
  default_id="run_$(date +%Y%m%d_%H%M%S)"
  {
    echo ""
    printf "  Run id [default: %s]: " "$default_id"
  } >&2
  read -r ans </dev/tty
  echo "${ans:-$default_id}"
}

launch_world_only() {
  local scene
  scene="$(pick_scene)"
  echo ""
  echo "► Stopping previous stack and launching world '${scene}' (no robot)..."
  stop_stack_processes
  case "$scene" in
    house)
      launch_foreground "world_${scene}" \
        "roslaunch aws_robomaker_small_house_world small_house.launch gui:=true"
      ;;
    wrs)
      launch_foreground "world_${scene}" \
        "roslaunch gazebo_ros empty_world.launch gui:=true world_name:=\$(rospack find tmc_wrs_gazebo_worlds)/worlds/wrs2020.world"
      ;;
    tb3_house)
      launch_foreground "world_${scene}" \
        "roslaunch vlmap_bringup world_tb3_house.launch gui:=true"
      ;;
    pal_home)
      launch_foreground "world_${scene}" \
        "roslaunch vlmap_bringup world_pal_home.launch gui:=true"
      ;;
  esac
}

launch_vlmap_capture() {
  local scene run_id
  scene="$(pick_capture_scene)"
  run_id="$(prompt_run_id)"
  echo ""
  echo "► Stopping previous stack and starting VLMap capture on '${scene}' (run_id=${run_id})..."
  echo "  Output: /shared/captures/${scene}/${run_id}/"
  echo "  This window stays in the foreground showing the capture log."
  echo "  Open ONE more terminal and drive the robot:"
  echo "       docker exec -it tfg-ros menu   ->  6  (HSR keyboard teleop)"
  echo "  When you finish driving, just Ctrl+C the teleop: the capture is"
  echo "  flushed to /shared/captures automatically. Then Ctrl+C here."
  echo "  (Option 5 flush / option 7 auto patrol remain available if needed.)"
  stop_stack_processes
  launch_foreground "vlmap_capture_${scene}_${run_id}" \
    "roslaunch vlmap_bringup vlmap_capture_session.launch world:=${scene} run_id:=${run_id}"
}

stop_vlmap_capture() {
  echo ""
  echo "► Calling /vlmap_capture/stop to flush the active capture..."
  run_in_ros_shell "rosservice call /vlmap_capture/stop '{}'" || \
    echo "  (No active capture node found, or service call failed.)"
}

launch_teleop() {
  echo ""
  echo "► HSR keyboard teleop (ROS standard keys, k=STOP). Ctrl+C to exit."
  echo "  Requires a running session from option 4 (or 2/3) in another terminal."
  echo ""
  echo "  When you exit teleop (Ctrl+C), any active VLMap capture is flushed"
  echo "  to disk automatically — no need to call the flush option by hand."
  echo ""
  launch_foreground "hsr_teleop" \
    "roslaunch vlmap_bringup hsr_teleop_keyboard.launch"
  # On exit from teleop, auto-flush the capture so the recorded run is written
  # to /shared/captures without a separate manual step. Harmless if there is no
  # active capture (the service call just reports it could not find the node).
  echo ""
  echo "► Teleop closed — flushing any active VLMap capture..."
  stop_vlmap_capture
}

launch_capture_patrol() {
  local scene
  scene="$(pick_capture_scene)"
  echo ""
  echo "► Automated HSR capture patrol for '${scene}'. Ctrl+C to abort."
  echo "  Requires a running capture session from option 4 in another terminal."
  echo "  The robot will visit configured waypoints and perform a 360° scan at each one."
  echo ""
  launch_foreground "hsr_capture_patrol_${scene}" \
    "roslaunch vlmap_bringup hsr_capture_patrol.launch world:=${scene} stop_capture_on_finish:=true"
}

launch_with_robot() {
  local with_rviz="$1"
  local scene
  local rviz_arg
  scene="$(pick_scene)"
  if [[ "$with_rviz" == "yes" ]]; then
    rviz_arg="use_rviz:=true"
  else
    rviz_arg="use_rviz:=false"
  fi
  echo ""
  echo "► Stopping previous stack and launching '${scene}' with HSR (rviz=${with_rviz})..."
  stop_stack_processes
  case "$scene" in
    house)
      launch_foreground "hsr_${scene}" \
        "roslaunch vlmap_bringup hsr_gazebo_small_house.launch gui:=true ${rviz_arg}"
      ;;
    wrs)
      launch_foreground "hsr_${scene}" \
        "roslaunch vlmap_bringup hsr_gazebo_wrs.launch gui:=true ${rviz_arg}"
      ;;
    tb3_house)
      launch_foreground "hsr_${scene}" \
        "roslaunch vlmap_bringup hsr_gazebo_tb3_house.launch gui:=true ${rviz_arg}"
      ;;
    pal_home)
      launch_foreground "hsr_${scene}" \
        "roslaunch vlmap_bringup hsr_gazebo_pal_home.launch gui:=true ${rviz_arg}"
      ;;
  esac
}

run_ros_menu() {
  while true; do
    echo ""
    echo "  ┌──────────────────────────────────────────────┐"
    echo "  │            ROS / Gazebo iteration menu       │"
    echo "  ├──────────────────────────────────────────────┤"
    echo "  │  1) World only (no robot)                    │"
    echo "  │  2) World + HSR robot                        │"
    echo "  │  3) World + HSR robot + RViz                 │"
    echo "  │  4) VLMap capture session (no teleop inside) │"
    echo "  │  5) Stop active VLMap capture (flush)        │"
    echo "  │  6) HSR keyboard teleop (ROS standard)       │"
    echo "  │  7) Auto capture patrol (waypoints + 360s)   │"
    echo "  │  q) Quit                                     │"
    echo "  └──────────────────────────────────────────────┘"
    printf "  Select: "
    read -r opt

    case "$opt" in
      1) launch_world_only ;;
      2) launch_with_robot "no" ;;
      3) launch_with_robot "yes" ;;
      4) launch_vlmap_capture ;;
      5) stop_vlmap_capture ;;
      6) launch_teleop ;;
      7) launch_capture_patrol ;;
      q|Q)
        echo ""
        echo "► Stopping active ROS/Gazebo processes..."
        stop_stack_processes
        break
        ;;
      *) echo "  Invalid option." ;;
    esac
  done
}

run_ros_menu
