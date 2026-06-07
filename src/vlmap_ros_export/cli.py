"""CLI entry point: load a saved VLMap and dump per-category 2D heatmaps.

Usage (from the tfg-sim container):

    python -m vlmap_ros_export.cli \\
        --data-dir /workspace/data/vlmaps_dataset/scene_X/ \\
        --output-dir /shared/hssd/scene_X/heatmaps \\
        --categories mug bottle laptop drill

The categories list is the union of (a) the queries you want to visualize
in RViz and (b) any class you expect ``/vlmap/query_room`` to receive.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

from .heatmap_dumper import dump_heatmap_npy, pool_voxel_scores_to_2d


def _load_vlmap(data_dir: str):
    """Lazy import so the rest of the package stays torch-free."""
    # third_party/vlmaps must be on sys.path; the tfg-sim entrypoint already
    # arranges this. Keep a defensive fallback for ad-hoc invocations.
    submodule = Path("/workspace/third_party/vlmaps")
    if submodule.exists() and str(submodule) not in sys.path:
        sys.path.insert(0, str(submodule))
    from vlmaps.map.vlmap import VLMap  # noqa: WPS433 - intentional lazy import
    from omegaconf import OmegaConf

    cfg_path = submodule / "config" / "map_config" / "vlmaps.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"VLMap config not found at {cfg_path}. "
            "Pass --config to override.",
        )
    cfg = OmegaConf.load(str(cfg_path))
    # map_config/vlmaps.yaml interpolates grid_size/cell_size from ${params.gs}
    # and ${params.cs}. Those live in the params group, which Hydra composes at
    # build time but is absent when the map_config is loaded standalone here.
    # Inject the same params/default.yaml the map was built with so the
    # interpolations resolve to the matching grid geometry (gs=1000, cs=0.05).
    params_path = submodule / "config" / "params" / "default.yaml"
    if params_path.exists():
        cfg.params = OmegaConf.load(str(params_path))
    vlmap = VLMap(cfg)
    if not vlmap.load_map(data_dir):
        raise RuntimeError(f"VLMap.load_map returned False for {data_dir}")
    vlmap._init_clip()  # noqa: SLF001 - public surface forces this anyway
    # Do not call init_categories here: main() scores each requested category
    # on its own via _scores_for_category(). Calling it with an empty list (the
    # state right after load_map) drives get_lseg_score with no landmarks and
    # raises IndexError on landmarks_other[-1].
    return vlmap


def _scores_for_category(vlmap, category: str) -> np.ndarray:
    """Return a per-voxel similarity score for ``category`` using the CLIP model."""
    from vlmaps.utils.clip_utils import get_lseg_score  # noqa: WPS433

    scores_mat = get_lseg_score(
        vlmap.clip_model,
        [category],
        vlmap.grid_feat,
        vlmap.clip_feat_dim,
        use_multiple_templates=True,
        add_other=True,
    )  # shape (N, 2): [category, "other"]
    # Take the category column. Higher = more similar.
    return np.asarray(scores_mat[:, 0], dtype=np.float32)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump VLMap 2D heatmaps for ROS.")
    parser.add_argument("--data-dir", required=True, help="Directory passed to VLMap.load_map.")
    parser.add_argument("--output-dir", required=True, help="Where to write <category>.npy files.")
    parser.add_argument("--categories", nargs="+", required=True, help="Categories to dump.")
    parser.add_argument(
        "--rotate-to-ros",
        action="store_true",
        help="Transpose+flip to match the ROS occupancy convention.",
    )
    args = parser.parse_args(argv)

    print(f"[vlmap_ros_export] loading VLMap from {args.data_dir} ...")
    vlmap = _load_vlmap(args.data_dir)
    gs = int(vlmap.gs)
    grid_pos = np.asarray(vlmap.grid_pos)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cat in args.categories:
        scores = _scores_for_category(vlmap, cat)
        heatmap_2d = pool_voxel_scores_to_2d(scores, grid_pos, gs)
        path = dump_heatmap_npy(
            heatmap_2d,
            str(out_dir),
            cat,
            rotate_to_ros=args.rotate_to_ros,
        )
        print(f"[vlmap_ros_export] {cat}: {heatmap_2d.shape} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
