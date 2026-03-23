"""
project_detections_3d.py  (Step 5)
===================================
Project YOLOE 2D bounding-box detections to 3D world positions using
depth maps and camera poses.

Reads:  <scene_dir>/yoloe_detections.json   (from Step 4)
        <scene_dir>/depth/*.npy
        <scene_dir>/poses.txt

Writes: <scene_dir>/detections_3d.json

Usage (inside Docker):
  python src/project_detections_3d.py data_paths=docker scene_id=0
"""

import json
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from vlmaps.utils.mapping_utils import (
    cvt_pose_vec2tf,
    base_pos2grid_id_3d,
)

# Import pure helpers from shared utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    CAM_CALIB,
    BASE_TRANSFORM,
    BASE2CAM_TF,
    backproject_bbox_center,
    cam_to_world,
    cluster_detections,
)


@hydra.main(
    version_base=None,
    config_path="../third_party/vlmaps/config",
    config_name="object_goal_navigation_cfg.yaml",
)
def main(config: DictConfig) -> None:
    gs = config.params.gs   # 1000
    cs = config.params.cs   # 0.05

    # ── Resolve scene directory ──────────────────────────────────────────
    data_dir = Path(config.data_paths.vlmaps_data_dir)
    scene_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if config.scene_id >= len(scene_dirs):
        print(f"ERROR: scene_id={config.scene_id} but only {len(scene_dirs)} scenes found.")
        sys.exit(1)
    scene_dir = scene_dirs[config.scene_id]

    # ── Load 2D detections ───────────────────────────────────────────────
    det_path = scene_dir / "yoloe_detections.json"
    if not det_path.exists():
        print(f"ERROR: {det_path} not found. Run run_yoloe_detect.py first (Step 4).")
        sys.exit(1)

    with open(det_path) as f:
        data = json.load(f)
    detections_2d = data["detections"]
    print(f"Loaded {len(detections_2d)} 2D detections from {det_path.name}")

    # ── Load poses ───────────────────────────────────────────────────────
    poses_path = scene_dir / "poses.txt"
    poses = np.loadtxt(str(poses_path))  # (N, 7): px py pz qx qy qz qw
    print(f"Loaded {len(poses)} poses")

    # Compute initial base transform (same as vlmap_builder)
    init_habitat_tf = cvt_pose_vec2tf(poses[0])
    init_base_tf = BASE_TRANSFORM @ init_habitat_tf @ np.linalg.inv(BASE_TRANSFORM)
    init_base_tf_inv = np.linalg.inv(init_base_tf)

    # ── Project each detection to 3D ─────────────────────────────────────
    detections_3d = []
    skipped = 0

    for det in tqdm(detections_2d, desc="Projecting to 3D"):
        frame_id = det["frame_id"]
        depth_path = scene_dir / "depth" / f"{frame_id:06d}.npy"
        if not depth_path.exists():
            skipped += 1
            continue

        depth_map = np.load(str(depth_path))

        # Back-project bbox center to camera frame
        p_cam = backproject_bbox_center(det["bbox_xyxy"], depth_map)
        if p_cam is None:
            skipped += 1
            continue

        # Transform to world frame
        pose_tf = cvt_pose_vec2tf(poses[frame_id])
        p_world = cam_to_world(p_cam, pose_tf, init_base_tf_inv)

        # Convert to grid coordinates
        row, col, height = base_pos2grid_id_3d(gs, cs, p_world[0], p_world[1], p_world[2])

        det_3d = {
            "frame_id": frame_id,
            "class_name": det["class_name"],
            "confidence": det["confidence"],
            "bbox_xyxy": det["bbox_xyxy"],
            "pos_3d_world": [round(float(x), 4) for x in p_world],
            "pos_grid": [int(row), int(col)],
        }
        detections_3d.append(det_3d)

    print(f"\nProjected: {len(detections_3d)}  Skipped: {skipped}")

    # ── Cluster nearby detections ────────────────────────────────────────
    merged = cluster_detections(detections_3d, merge_radius_m=0.5)
    print(f"After clustering: {len(merged)} unique candidates")

    # ── Summary ──────────────────────────────────────────────────────────
    counts = {}
    for d in merged:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:20s}: {cnt}")

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = scene_dir / "detections_3d.json"
    payload = {
        "metadata": {
            "scene": scene_dir.name,
            "total_2d": len(detections_2d),
            "projected": len(detections_3d),
            "merged": len(merged),
            "skipped": skipped,
            "gs": gs,
            "cs": cs,
        },
        "detections_raw": detections_3d,
        "detections_merged": merged,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
