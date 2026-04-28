"""File-backed helpers for the Habitat -> ROS bridge.

The bridge is intentionally decoupled from Habitat internals. For Sprint 2 it
can read:

- occupancy grids from ``.npy`` or ``.json``
- robot pose from a tiny ``.json`` sidecar

This keeps the HSSD/Habitat pipeline untouched while still giving the ROS
container a stable interface that can later be replaced by:

- Gazebo topics
- HSR drivers
- a live Habitat publisher
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class MapMeta:
    resolution: float
    origin_x: float = 0.0
    origin_y: float = 0.0
    frame_id: str = "map"
    free_values: Tuple[int, ...] = (0,)


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float
    frame_id: str = "map"


def load_occupancy_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"occupancy file not found: {path}")
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        payload = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
        arr = np.asarray(payload, dtype=np.int32)
    else:
        raise ValueError(f"unsupported occupancy file format: {path.suffix}")
    if arr.ndim != 2:
        raise ValueError(f"occupancy array must be 2D, got shape {arr.shape}")
    return arr.astype(np.int32, copy=False)


def load_pose_json(path: str | Path) -> Pose2D:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"pose file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return Pose2D(
        x=float(payload["x"]),
        y=float(payload["y"]),
        yaw=float(payload["yaw"]),
        frame_id=str(payload.get("frame_id", "map")),
    )


def rowcol_to_xy(row: float, col: float, meta: MapMeta) -> Tuple[float, float]:
    """Convert occupancy-grid indices into /map-frame metres.

    Convention:
    - ``row`` increases downward in the matrix
    - ``col`` increases to the right
    - origin corresponds to the bottom-left corner of cell (0,0)
    """

    x = meta.origin_x + (float(col) + 0.5) * meta.resolution
    y = meta.origin_y + (float(row) + 0.5) * meta.resolution
    return x, y


def xy_to_rowcol(x: float, y: float, meta: MapMeta) -> Tuple[int, int]:
    col = int(np.floor((float(x) - meta.origin_x) / meta.resolution))
    row = int(np.floor((float(y) - meta.origin_y) / meta.resolution))
    return row, col


def flatten_occupancy_for_ros(occupancy: np.ndarray, *, occupied_value: int = 100) -> list:
    """Convert a 2D occupancy grid into the row-major 1D ROS payload."""

    occ = np.asarray(occupancy, dtype=np.int32)
    flat = occ.reshape(-1)
    result = []
    for value in flat:
        if value < 0:
            result.append(-1)
        elif value == 0:
            result.append(0)
        else:
            result.append(int(occupied_value))
    return result
