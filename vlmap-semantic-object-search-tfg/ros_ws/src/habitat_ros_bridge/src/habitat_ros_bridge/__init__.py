"""habitat_ros_bridge — Habitat-Sim ↔ ROS topic bridge.

Sprint 1: stub. Sprint 2 will:
  - run inside tfg-ros (subscribes via rosbridge to a publisher inside tfg-sim,
    OR runs as a sidecar in tfg-sim with rospy connecting to tfg-ros's master)
  - publish /map (nav_msgs/OccupancyGrid) from VLMap obstacles_map
  - publish /odom (nav_msgs/Odometry) from robot.curr_pos_on_map
  - publish /tf for map -> base_link
  - publish /scan derived from depth or the configured lidar_sensor
  - publish /camera/color/image_raw and depth
"""
from .file_bridge import (
    MapMeta,
    Pose2D,
    flatten_occupancy_for_ros,
    load_occupancy_array,
    load_pose_json,
    rowcol_to_xy,
    xy_to_rowcol,
)

__all__ = [
    "MapMeta",
    "Pose2D",
    "flatten_occupancy_for_ros",
    "load_occupancy_array",
    "load_pose_json",
    "rowcol_to_xy",
    "xy_to_rowcol",
]
