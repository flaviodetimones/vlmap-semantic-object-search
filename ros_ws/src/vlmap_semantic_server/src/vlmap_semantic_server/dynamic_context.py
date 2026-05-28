"""Helpers for feeding live semantic room context into the ROS stack.

The dynamic context is intentionally JSON-based so that tfg-sim can publish it
through rosbridge without depending on custom ROS messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class DynamicRoomEntry:
    room_id: str
    x: float
    y: float
    aliases: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


class DynamicRoomContext:
    def __init__(
        self,
        rooms: List[DynamicRoomEntry],
        priors: Optional[Mapping[str, List[Tuple[str, float]]]] = None,
    ) -> None:
        self._rooms = list(rooms)
        self._room_by_key: Dict[str, DynamicRoomEntry] = {}
        for room in self._rooms:
            self._room_by_key[self._norm(room.room_id)] = room
            for alias in room.aliases:
                self._room_by_key.setdefault(self._norm(alias), room)
        self._priors: Dict[str, List[Tuple[str, float]]] = {}
        for category, ranked in (priors or {}).items():
            self._priors[self._norm(category)] = [
                (str(room_id), float(score)) for room_id, score in ranked if str(room_id).strip()
            ]

    @staticmethod
    def _norm(text: str) -> str:
        return str(text or "").strip().lower()

    def is_available(self) -> bool:
        return bool(self._rooms)

    def list_rooms(self) -> List[str]:
        return [room.room_id for room in self._rooms]

    def resolve_room_name(self, room: str) -> Optional[str]:
        entry = self._room_by_key.get(self._norm(room))
        return None if entry is None else entry.room_id

    def resolve_room_pose(self, room: str) -> Optional[Tuple[float, float]]:
        entry = self._room_by_key.get(self._norm(room))
        return None if entry is None else (float(entry.x), float(entry.y))

    def room_at_pose(
        self, x: float, y: float, *, max_distance: Optional[float] = None
    ) -> Optional[str]:
        """Return the room whose centroid is closest to the (x, y) pose.

        This lets tfg-ros derive the *actual* room the robot ended up in from
        its final pose, instead of echoing the requested room. ``max_distance``
        (metres) optionally rejects matches that are implausibly far from any
        known centroid; ``None`` always returns the nearest room (or None when
        no rooms are loaded).
        """
        if not self._rooms:
            return None
        px, py = float(x), float(y)
        best_room: Optional[str] = None
        best_d2 = float("inf")
        for room in self._rooms:
            dx = float(room.x) - px
            dy = float(room.y) - py
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_room = room.room_id
        if best_room is None:
            return None
        if max_distance is not None and best_d2 > float(max_distance) ** 2:
            return None
        return best_room

    def rank_rooms_for_category(self, category: str) -> Tuple[List[str], List[float]]:
        key = self._norm(category)
        ranked = self._priors.get(key) or self._priors.get("*")
        if ranked:
            filtered = []
            for room_id, score in ranked:
                resolved = self.resolve_room_name(room_id)
                if resolved is not None:
                    filtered.append((resolved, float(score)))
            if filtered:
                return [room_id for room_id, _ in filtered], [score for _, score in filtered]
        rooms = self.list_rooms()
        if not rooms:
            return [], []
        uniform = 1.0 / float(len(rooms))
        return rooms, [uniform for _ in rooms]


def _row_col_to_xy(
    payload: Mapping[str, Any], resolution: float, origin_x: float, origin_y: float
) -> Tuple[float, float]:
    row = float(payload["row"])
    col = float(payload["col"])
    x = origin_x + (col + 0.5) * float(resolution)
    y = origin_y + (row + 0.5) * float(resolution)
    return float(x), float(y)


def load_dynamic_room_context(
    raw_text: str,
    *,
    resolution: float = 0.05,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> DynamicRoomContext:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("empty room context payload")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("room context must be a JSON object")

    rooms_payload = payload.get("rooms") or []
    if not isinstance(rooms_payload, list):
        raise ValueError("'rooms' must be a list")

    rooms: List[DynamicRoomEntry] = []
    for item in rooms_payload:
        if not isinstance(item, dict):
            raise ValueError("each room entry must be an object")
        room_id = str(item.get("room_id") or item.get("label") or "").strip()
        if not room_id:
            raise ValueError("room entry missing room_id/label")
        if "x" in item and "y" in item:
            x, y = float(item["x"]), float(item["y"])
        elif "row" in item and "col" in item:
            x, y = _row_col_to_xy(item, resolution, origin_x, origin_y)
        else:
            raise ValueError(f"room {room_id!r} missing pose (x/y or row/col)")
        aliases = tuple(str(alias).strip() for alias in item.get("aliases", []) if str(alias).strip())
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        rooms.append(
            DynamicRoomEntry(
                room_id=room_id,
                x=x,
                y=y,
                aliases=aliases,
                metadata=dict(metadata),
            )
        )

    priors_payload = payload.get("priors") or {}
    priors: Dict[str, List[Tuple[str, float]]] = {}
    if not isinstance(priors_payload, dict):
        raise ValueError("'priors' must be an object")
    for category, ranked in priors_payload.items():
        if not isinstance(ranked, list):
            raise ValueError(f"priors for {category!r} must be a list")
        parsed_ranked: List[Tuple[str, float]] = []
        for entry in ranked:
            if isinstance(entry, dict):
                room_id = str(entry.get("room_id") or entry.get("label") or "").strip()
                score = float(entry.get("score", 0.0))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                room_id = str(entry[0]).strip()
                score = float(entry[1])
            else:
                raise ValueError(f"invalid prior entry for {category!r}: {entry!r}")
            if room_id:
                parsed_ranked.append((room_id, score))
        priors[str(category)] = parsed_ranked

    return DynamicRoomContext(rooms, priors)
