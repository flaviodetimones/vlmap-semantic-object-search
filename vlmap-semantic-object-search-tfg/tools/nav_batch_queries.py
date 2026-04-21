#!/usr/bin/env python3
"""
Generate room/object query batches for interactive_object_nav.py.

Outputs all navigable rooms first, then all present objects in the scene.
Intended to be run inside the Docker container.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "third_party" / "vlmaps") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "vlmaps"))

CONFIG_DIR = str(REPO_ROOT / "third_party" / "vlmaps" / "config")

from vlmaps.robot.habitat_lang_robot import HabitatLanguageRobot
from vlmaps.utils.matterport3d_categories import get_categories
from vlmaps.utils.search_state import SearchState


def _build_present_categories(robot, score_thresh: float = 0.3, min_voxels: int = 10):
    cats = getattr(robot.map, "categories", [])
    scores_mat = getattr(robot.map, "scores_mat", None)
    structural = {"void", "wall", "floor", "ceiling"}
    if scores_mat is None:
        return [c for c in cats if c not in structural]

    max_ids = np.argmax(scores_mat, axis=1)
    present = []
    for i, c in enumerate(cats):
        if c in structural:
            continue
        mask = (max_ids == i) & (scores_mat[:, i] > score_thresh)
        if int(mask.sum()) >= min_voxels:
            present.append(c)
    return present


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", type=int, default=1)
    parser.add_argument("--dataset-type", default="hssd")
    parser.add_argument(
        "--scene-dataset-config-file",
        default="/workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json",
    )
    parser.add_argument("--data-paths", default="hssd")
    parser.add_argument("--min-room-navigable", type=float, default=0.25)
    args = parser.parse_args()

    overrides = [
        f"scene_id={args.scene_id}",
        f"dataset_type={args.dataset_type}",
        f"scene_dataset_config_file={args.scene_dataset_config_file}",
        f"data_paths={args.data_paths}",
    ]
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="object_goal_navigation_cfg", overrides=overrides)

    robot = HabitatLanguageRobot(cfg)
    robot.setup_scene(cfg.scene_id)
    robot.map.init_categories(get_categories(args.dataset_type))

    room_provider = getattr(robot, "room_provider", None)
    search_state = SearchState("__batch__", room_provider, robot.map.obstacles_map)

    room_queries = []
    if room_provider and room_provider.is_available():
        for room_name, rs in search_state.rooms.items():
            if rs.explored_ratio >= args.min_room_navigable:
                room_queries.append(room_name)

    object_queries = _build_present_categories(robot)

    for q in room_queries:
        print(q)
    for q in object_queries:
        print(q)


if __name__ == "__main__":
    main()
