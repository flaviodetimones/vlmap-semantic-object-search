"""
convert_labelme_rooms.py
========================
Convert a LabelMe JSON annotation file into the VLMap room segmentation
format: room_map.npy + regions.json.

Workflow:
  1. Export top-down map:  python src/export_topdown_map.py data_paths=docker scene_id=0
  2. Open the PNG in LabelMe and draw polygons for each room.
     Use room category names as labels (e.g., kitchen, bedroom, hallway).
  3. Save the LabelMe JSON, then run this script:

Usage (inside Docker):
  python src/convert_labelme_rooms.py data_paths=docker scene_id=0 \
      +labelme_json=data/vlmaps_dataset/<scene>/topdown_rgb.json

Output (in <scene_dir>/room_map/):
  room_map.npy        — (gs, gs) int32, category index per cell (-1 = unlabeled)
  regions.json        — region metadata (centroids, areas, labels)
  room_map_viz.png    — visualization overlay
"""

import json
import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig

from vlmaps.utils.mapping_utils import load_3d_map
from vlmaps.utils.visualize_utils import pool_3d_rgb_to_2d
from vlmaps.utils.room_map_utils import (
    MP3D_ROOM_CATEGORIES,
    _normalize_query,
    room_color,
    save_room_map,
)


def _find_crop_bounds(rgb_2d, gs, pad=10):
    """Return (rmin, rmax, cmin, cmax) of the non-black region."""
    rows, cols = np.where(np.any(rgb_2d > 0, axis=2))
    if len(rows) == 0:
        return 0, gs - 1, 0, gs - 1
    rmin = max(0, rows.min() - pad)
    rmax = min(gs - 1, rows.max() + pad)
    cmin = max(0, cols.min() - pad)
    cmax = min(gs - 1, cols.max() + pad)
    return rmin, rmax, cmin, cmax


@hydra.main(
    version_base=None,
    config_path="../third_party/vlmaps/config",
    config_name="map_indexing_cfg.yaml",
)
def main(config: DictConfig) -> None:
    labelme_path = config.get("labelme_json", None)
    if not labelme_path:
        print("ERROR: pass +labelme_json=<path_to_labelme_json>")
        sys.exit(1)
    labelme_path = Path(labelme_path)
    if not labelme_path.exists():
        print(f"ERROR: LabelMe JSON not found: {labelme_path}")
        sys.exit(1)

    # ── Resolve scene dir ─────────────────────────────────────────────────
    data_dir = Path(config.data_paths.vlmaps_data_dir)
    scene_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if config.scene_id >= len(scene_dirs):
        print(f"ERROR: scene_id={config.scene_id} but only {len(scene_dirs)} scenes.")
        sys.exit(1)
    scene_dir = scene_dirs[config.scene_id]

    # ── Load VLMap to get grid dimensions and top-down RGB ────────────────
    map_path = scene_dir / "vlmap" / "vlmaps.h5df"
    if not map_path.exists():
        print(f"ERROR: VLMap not found at {map_path}")
        sys.exit(1)

    print(f"Loading VLMap from {map_path} ...")
    _, _, grid_pos, _, _, grid_rgb = load_3d_map(str(map_path))
    gs = config.params.gs
    rgb_2d = pool_3d_rgb_to_2d(grid_rgb, grid_pos, gs)

    # ── Crop bounds (must match export_topdown_map.py) ────────────────────
    rmin, rmax, cmin, cmax = _find_crop_bounds(rgb_2d, gs)
    crop_h = rmax - rmin + 1
    crop_w = cmax - cmin + 1
    print(f"Crop bounds: rows [{rmin}, {rmax}]  cols [{cmin}, {cmax}]")
    print(f"Crop size: {crop_w}x{crop_h}")

    # ── Parse LabelMe JSON ────────────────────────────────────────────────
    with open(labelme_path, encoding="utf-8") as f:
        lm = json.load(f)

    categories = MP3D_ROOM_CATEGORIES
    room_map = np.full((gs, gs), -1, dtype=np.int32)
    regions = {}
    region_id = 0

    print(f"\nParsing {len(lm.get('shapes', []))} shapes from LabelMe ...")

    for shape in lm.get("shapes", []):
        label_raw = shape["label"]
        category = _normalize_query(label_raw)

        if category not in categories:
            print(f"  [warn] Unknown category '{label_raw}' → '{category}', skipping.")
            continue

        cat_idx = categories.index(category)
        points = np.array(shape["points"], dtype=np.float32)

        # LabelMe polygon is in cropped-image coords (col, row).
        # Convert to full-map coords.
        polygon_full = points.copy()
        polygon_full[:, 0] += cmin  # col offset
        polygon_full[:, 1] += rmin  # row offset

        # Draw polygon onto the full room_map
        # cv2.fillPoly expects (col, row) = (x, y) format, which is what we have
        mask = np.zeros((gs, gs), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_full.astype(np.int32)], 1)
        room_map[mask == 1] = cat_idx

        # Compute region info
        coords = np.argwhere(mask == 1)
        if len(coords) == 0:
            continue
        centroid = (int(round(coords[:, 0].mean())), int(round(coords[:, 1].mean())))
        area = int(mask.sum())

        regions[region_id] = {
            "category": category,
            "label": label_raw,
            "instance_idx": region_id,
            "centroid": [centroid[0], centroid[1]],
            "raw_centroid": [float(coords[:, 0].mean()), float(coords[:, 1].mean())],
            "area": area,
            "confidence": 1.0,
        }
        print(f"  {label_raw:20s} → {category:15s}  (cat_idx={cat_idx}, area={area})")
        region_id += 1

    if region_id == 0:
        print("\nERROR: No valid room polygons found.")
        sys.exit(1)

    # ── Visualize ─────────────────────────────────────────────────────────
    canvas = np.zeros((gs, gs, 3), dtype=np.uint8)
    # Background: dim top-down RGB
    bgr_bg = cv2.cvtColor(rgb_2d, cv2.COLOR_RGB2BGR)
    canvas = (bgr_bg * 0.35).astype(np.uint8)

    for i, cat in enumerate(categories):
        color = room_color(i)
        canvas[room_map == i] = color

    for rid, info in regions.items():
        r, c = info["centroid"]
        cv2.circle(canvas, (c, r), 5, (255, 255, 255), -1)
        cv2.putText(
            canvas, info["label"], (c + 7, r + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # ── Save ──────────────────────────────────────────────────────────────
    save_room_map(scene_dir, room_map, categories, regions, canvas)

    labeled = np.sum(room_map >= 0)
    total = np.sum(np.any(rgb_2d > 0, axis=2))
    coverage = labeled / total * 100 if total > 0 else 0
    print(f"\nRoom map: {region_id} regions, {labeled} labeled cells "
          f"({coverage:.1f}% of building footprint)")
    print("Done.")


if __name__ == "__main__":
    main()
