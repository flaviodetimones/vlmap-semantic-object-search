"""Centroid-based room labeler for the ROS migration.

This module implements the room annotation scheme described in
``project_migration_to_ros.pdf``:

1. The user marks one semantic centroid per room on the occupancy map.
2. Free cells are assigned to the closest centroid (Voronoi partition).
3. Obstacles / unknown cells remain unlabeled.

The implementation is intentionally dependency-light so it can be reused by:

- the future ROS node ``vlmap_semantic_server_node``
- offline tooling / tests on the host
- a future RViz click-based annotation tool

It does not depend on ROS, Habitat, torch, or the VLMaps submodule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


UNASSIGNED_ROOM_ID = -1


@dataclass(frozen=True)
class RoomCentroid:
    """Single semantic room anchor expressed in occupancy-grid coordinates."""

    label: str
    row: float
    col: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> Tuple[float, float]:
        return float(self.row), float(self.col)


def _parse_centroid_payload(payload: Any) -> List[RoomCentroid]:
    if isinstance(payload, dict):
        payload = payload.get("rooms", [])
    if not isinstance(payload, list):
        raise ValueError("centroid payload must be a list or {'rooms': [...]} dict")

    centroids: List[RoomCentroid] = []
    seen_labels = set()
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"centroid entry must be a dict, got {type(item)!r}")
        label = str(item.get("label", "")).strip()
        if not label:
            raise ValueError("centroid entry is missing a non-empty 'label'")
        if label.lower() in seen_labels:
            raise ValueError(f"duplicate centroid label: {label!r}")
        seen_labels.add(label.lower())
        centroids.append(
            RoomCentroid(
                label=label,
                row=float(item["row"]),
                col=float(item["col"]),
                metadata=dict(item.get("metadata") or {}),
            )
        )
    if not centroids:
        raise ValueError("at least one centroid is required")
    return centroids


def load_room_centroids(path: Path) -> List[RoomCentroid]:
    """Load room centroids from JSON.

    Accepted shapes:
    - ``[{"label": "...", "row": 10, "col": 20}, ...]``
    - ``{"rooms": [...same entries...]}``
    """

    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return _parse_centroid_payload(payload)


def dump_room_centroids(path: Path, centroids: Sequence[RoomCentroid]) -> None:
    """Write centroids in the canonical JSON schema used by this module."""

    rows = [
        {
            "label": c.label,
            "row": float(c.row),
            "col": float(c.col),
            "metadata": dict(c.metadata or {}),
        }
        for c in centroids
    ]
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump({"rooms": rows}, fh, indent=2, sort_keys=True)
        fh.write("\n")


def build_voronoi_room_map(
    occupancy_map: np.ndarray,
    centroids: Sequence[RoomCentroid],
    *,
    free_values: Iterable[int] = (0,),
    unassigned_value: int = UNASSIGNED_ROOM_ID,
) -> Tuple[np.ndarray, List[str], Dict[int, Dict[str, Any]]]:
    """Partition free occupancy cells by nearest room centroid.

    Parameters
    ----------
    occupancy_map:
        2D occupancy grid following the ROS convention by default:
        ``0=free``, ``100=occupied``, ``-1=unknown``.
    centroids:
        One semantic anchor per room.
    free_values:
        Cell values considered navigable and therefore assignable.
    unassigned_value:
        Integer used for obstacles / unknown cells in the returned room map.

    Returns
    -------
    room_map:
        ``int32`` array with one room id per free cell and ``unassigned_value``
        elsewhere.
    labels:
        Ordered room labels matching the ids stored in ``room_map``.
    regions:
        Region metadata keyed by integer room id. The shape is intentionally
        similar to the existing VLMaps room-map metadata to ease integration.
    """

    occ = np.asarray(occupancy_map)
    if occ.ndim != 2:
        raise ValueError(f"occupancy_map must be 2D, got shape {occ.shape}")
    parsed = _parse_centroid_payload(
        [
            {
                "label": c.label,
                "row": c.row,
                "col": c.col,
                "metadata": c.metadata,
            }
            for c in centroids
        ]
    )

    free_mask = np.isin(occ, list(free_values))
    room_map = np.full(occ.shape, int(unassigned_value), dtype=np.int32)
    free_cells = np.argwhere(free_mask)
    labels = [c.label for c in parsed]
    if free_cells.size == 0:
        regions = {
            idx: {
                "label": c.label,
                "centroid": [float(c.row), float(c.col)],
                "area": 0,
                "metadata": dict(c.metadata or {}),
            }
            for idx, c in enumerate(parsed)
        }
        return room_map, labels, regions

    centroid_arr = np.asarray([[c.row, c.col] for c in parsed], dtype=np.float32)
    # Deterministic tie-breaking comes from np.argmin choosing the first
    # centroid in input order when distances are equal.
    dists = ((free_cells[:, None, :] - centroid_arr[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argmin(dists, axis=1)
    room_map[free_cells[:, 0], free_cells[:, 1]] = nearest.astype(np.int32)

    regions: Dict[int, Dict[str, Any]] = {}
    for idx, centroid in enumerate(parsed):
        assigned = free_cells[nearest == idx]
        regions[idx] = {
            "label": centroid.label,
            "centroid": [float(centroid.row), float(centroid.col)],
            "area": int(len(assigned)),
            "metadata": dict(centroid.metadata or {}),
        }
    return room_map, labels, regions


class VoronoiRoomProvider:
    """Room-provider style wrapper over a centroid/Voronoi partition."""

    def __init__(
        self,
        occupancy_map: np.ndarray,
        centroids: Sequence[RoomCentroid],
        *,
        free_values: Iterable[int] = (0,),
    ) -> None:
        self._room_map, self._labels, self._regions = build_voronoi_room_map(
            occupancy_map, centroids, free_values=free_values
        )
        self._label_lookup = {label.lower(): idx for idx, label in enumerate(self._labels)}
        self._available = True

    def is_available(self) -> bool:
        return self._available

    def list_rooms(self) -> List[str]:
        return list(self._labels)

    def resolve_room_name(self, room_query: str) -> Optional[str]:
        idx = self._label_lookup.get(str(room_query or "").strip().lower())
        return None if idx is None else self._labels[idx]

    def get_room_at_cell(self, row: int, col: int) -> Optional[str]:
        if row < 0 or col < 0 or row >= self._room_map.shape[0] or col >= self._room_map.shape[1]:
            return None
        idx = int(self._room_map[row, col])
        if idx == UNASSIGNED_ROOM_ID:
            return None
        return self._labels[idx]

    def get_room_centroid(self, room_name: str) -> Optional[Tuple[float, float]]:
        resolved = self.resolve_room_name(room_name)
        if resolved is None:
            return None
        idx = self._label_lookup[resolved.lower()]
        centroid = self._regions[idx]["centroid"]
        return float(centroid[0]), float(centroid[1])

    def to_regions(self) -> Dict[int, Dict[str, Any]]:
        return {int(k): dict(v) for k, v in self._regions.items()}
