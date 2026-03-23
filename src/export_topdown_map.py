"""
export_topdown_map.py
=====================
Export the VLMap top-down RGB image as a PNG file, ready for room
labeling in LabelMe or any other annotation tool.

Usage (inside Docker):
  python src/export_topdown_map.py data_paths=docker scene_id=0

Output:
  data/vlmaps_dataset/<scene>/topdown_rgb.png
"""

import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig

from vlmaps.utils.mapping_utils import load_3d_map
from vlmaps.utils.visualize_utils import pool_3d_rgb_to_2d


@hydra.main(
    version_base=None,
    config_path="../third_party/vlmaps/config",
    config_name="map_indexing_cfg.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir)
    scene_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if config.scene_id >= len(scene_dirs):
        print(f"ERROR: scene_id={config.scene_id} but only {len(scene_dirs)} scenes found.")
        sys.exit(1)

    scene_dir = scene_dirs[config.scene_id]
    map_path = scene_dir / "vlmap" / "vlmaps.h5df"
    if not map_path.exists():
        print(f"ERROR: VLMap not found at {map_path}")
        sys.exit(1)

    print(f"Loading VLMap from {map_path} ...")
    _, _, grid_pos, _, _, grid_rgb = load_3d_map(str(map_path))

    gs = config.params.gs
    rgb_2d = pool_3d_rgb_to_2d(grid_rgb, grid_pos, gs)

    # Crop to building footprint (non-black region)
    rows, cols = np.where(np.any(rgb_2d > 0, axis=2))
    if len(rows) == 0:
        print("ERROR: top-down map is entirely black.")
        sys.exit(1)
    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()
    pad = 10
    rmin, cmin = max(0, rmin - pad), max(0, cmin - pad)
    rmax, cmax = min(gs - 1, rmax + pad), min(gs - 1, cmax + pad)
    cropped = rgb_2d[rmin:rmax + 1, cmin:cmax + 1]

    # Save
    out_path = scene_dir / "topdown_rgb.png"
    bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr)

    print(f"Exported {cropped.shape[1]}x{cropped.shape[0]} top-down map to:")
    print(f"  {out_path}")
    print(f"\nCrop bounds: rows [{rmin}, {rmax}]  cols [{cmin}, {cmax}]")
    print("Open this PNG in LabelMe to label rooms.")

    # Show the exported map
    cv2.imshow("Top-down RGB map", bgr)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
