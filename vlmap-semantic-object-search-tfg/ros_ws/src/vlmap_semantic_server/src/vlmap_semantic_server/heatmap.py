"""File-backed heatmap loader for the VLMap semantic server.

Architectural rationale: torch + CLIP + the full VLMap (h5df) live in the
tfg-sim container, where the search pipeline already runs. To avoid pulling
that stack into tfg-ros, the simulator writes per-category similarity
heatmaps to a shared volume as ``<category>.npy`` (2D float array, same
grid as ``occupancy_path``). The semantic server reads them on demand and
republishes them as standard ROS messages so RViz can render them on top
of ``/map``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def load_heatmap_array(heatmap_dir: str, category: str) -> Optional[np.ndarray]:
    """Load ``<category>.npy`` from ``heatmap_dir``. Returns None if missing."""
    if not heatmap_dir:
        return None
    candidate = Path(heatmap_dir) / f"{category}.npy"
    if not candidate.exists():
        return None
    arr = np.load(candidate)
    return np.asarray(arr, dtype=np.float32)


def normalize_to_occupancy(arr: np.ndarray) -> np.ndarray:
    """Scale ``arr`` to int8 in [0, 100], the OccupancyGrid convention."""
    a = arr.astype(np.float32)
    amin = float(a.min())
    amax = float(a.max())
    if amax > amin:
        scaled = (a - amin) / (amax - amin) * 100.0
    else:
        scaled = np.zeros_like(a)
    return scaled.astype(np.int8)


def extract_top_candidates(
    arr: np.ndarray,
    n: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    min_distance_cells: int = 5,
) -> List[Tuple[float, float, float]]:
    """Pick the top-N peaks of ``arr`` with a simple non-max suppression.

    Returns a list of ``(x, y, score)`` tuples in the ``/map`` frame.
    """
    if n <= 0 or arr.size == 0:
        return []
    h, w = arr.shape
    flat_idx = np.argsort(arr.ravel())[::-1]
    picked: List[Tuple[float, float, float]] = []
    chosen_rc: List[Tuple[int, int]] = []
    for idx in flat_idx:
        if len(picked) >= n:
            break
        r = int(idx // w)
        c = int(idx % w)
        if any(
            abs(r - r2) < min_distance_cells and abs(c - c2) < min_distance_cells
            for r2, c2 in chosen_rc
        ):
            continue
        chosen_rc.append((r, c))
        x = float(origin_x) + (c + 0.5) * float(resolution)
        y = float(origin_y) + (r + 0.5) * float(resolution)
        picked.append((x, y, float(arr[r, c])))
    return picked
