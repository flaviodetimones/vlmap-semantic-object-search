"""Project per-voxel CLIP scores onto a 2D grid and dump as ``.npy``.

The output format matches what ``vlmap_semantic_server.heatmap`` reads on
the ROS side: a single ``<category>.npy`` file holding a 2D float32 array
on the same grid as the bridge's occupancy map.

Pure NumPy — no torch, no CLIP, no h5py. Callers (typically the
``cli`` module of this package) are responsible for producing the raw
per-voxel scores using the existing ``vlmaps`` utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def pool_voxel_scores_to_2d(
    scores: np.ndarray,
    grid_pos: np.ndarray,
    gs: int,
) -> np.ndarray:
    """Max-pool per-voxel scalar scores onto a ``gs x gs`` 2D grid.

    Args:
        scores: shape ``(N,)`` — one scalar score per occupied voxel.
        grid_pos: shape ``(N, 3)`` — voxel ``(row, col, height)`` indices,
            same convention used by ``vlmaps.utils.visualize_utils``.
        gs: grid size (one side of the square 2D grid).

    Returns:
        ``(gs, gs)`` float32 array. Cells with no voxel above them stay 0.
    """
    if scores.shape[0] != grid_pos.shape[0]:
        raise ValueError(
            f"scores and grid_pos must have the same length "
            f"(got {scores.shape[0]} vs {grid_pos.shape[0]})"
        )
    out = np.zeros((gs, gs), dtype=np.float32)
    for i in range(scores.shape[0]):
        row, col, _h = grid_pos[i]
        s = float(scores[i])
        if s > out[row, col]:
            out[row, col] = s
    return out


def dump_heatmap_npy(
    heatmap_2d: np.ndarray,
    output_dir: str,
    category: str,
    *,
    rotate_to_ros: bool = False,
) -> Path:
    """Persist a 2D heatmap as ``<output_dir>/<category>.npy``.

    Args:
        heatmap_2d: 2D float array.
        output_dir: target directory (created if missing).
        category: filename stem; spaces and slashes are replaced.
        rotate_to_ros: if True, transpose+flip so the array is in the same
            row/col convention used by the ``OccupancyGrid`` published by
            ``habitat_ros_bridge``. Off by default — set when the saved
            occupancy map already follows that convention.

    Returns:
        Path to the written file.
    """
    if heatmap_2d.ndim != 2:
        raise ValueError(f"heatmap must be 2D (got shape {heatmap_2d.shape})")
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    safe = category.strip().replace(" ", "_").replace("/", "_")
    if not safe:
        raise ValueError("category must produce a non-empty filename")
    arr = heatmap_2d.astype(np.float32, copy=False)
    if rotate_to_ros:
        arr = np.flipud(arr.T).copy()
    out_path = target_dir / f"{safe}.npy"
    np.save(out_path, arr)
    return out_path


def normalize_heatmap(arr: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    """Min-max normalize to [0, 1]. Returns zeros if the array is constant."""
    a = arr.astype(np.float32)
    amin = float(a.min())
    amax = float(a.max())
    if amax - amin < eps:
        return np.zeros_like(a)
    return (a - amin) / (amax - amin)
