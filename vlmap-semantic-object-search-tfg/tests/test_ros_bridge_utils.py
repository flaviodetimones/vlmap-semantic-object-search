"""Offline tests for ROS migration helper modules.

These tests validate the file-backed bridge and lightweight task-manager parser
without requiring a ROS master or the Habitat runtime.
"""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros_ws" / "src" / "habitat_ros_bridge" / "src"))
sys.path.insert(0, str(ROOT / "ros_ws" / "src" / "vlmap_task_manager" / "src"))

from habitat_ros_bridge import (
    MapMeta,
    flatten_occupancy_for_ros,
    load_occupancy_array,
    load_pose_json,
    rowcol_to_xy,
    xy_to_rowcol,
)
from vlmap_task_manager import parse_instruction


def test_rowcol_xy_round_trip():
    meta = MapMeta(resolution=0.5, origin_x=-1.0, origin_y=2.0)
    x, y = rowcol_to_xy(3, 4, meta)
    assert (x, y) == (1.25, 3.75)
    assert xy_to_rowcol(x, y, meta) == (3, 4)


def test_flatten_occupancy_for_ros():
    occ = np.array([[0, 100], [-1, 0]], dtype=np.int32)
    assert flatten_occupancy_for_ros(occ) == [0, 100, -1, 0]


def test_loaders_and_parser(tmp_path):
    occ_path = tmp_path / "occ.json"
    occ_path.write_text('{"data": [[0, 100], [0, -1]]}\n', encoding="utf-8")
    pose_path = tmp_path / "pose.json"
    pose_path.write_text('{"x": 1.5, "y": -2.0, "yaw": 0.3}\n', encoding="utf-8")

    occ = load_occupancy_array(occ_path)
    pose = load_pose_json(pose_path)

    assert occ.shape == (2, 2)
    assert pose.x == 1.5
    assert pose.y == -2.0
    assert pose.yaw == 0.3

    parsed = parse_instruction("find the laptop in the office")
    assert parsed.target == "laptop"
    assert parsed.explicit_room == "office"

    parsed2 = parse_instruction("search for mug")
    assert parsed2.target == "mug"
    assert parsed2.explicit_room is None
