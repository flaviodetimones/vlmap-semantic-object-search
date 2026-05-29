"""SemanticGoal dataclass — the unique adapter between the strategic
policy ("decide what to look for") and the navigation backend
("execute the goal"). Mirrors the ROS message
``vlmap_msgs/SemanticGoal``.

Kept dependency-free on purpose: importing this module must not pull in
torch, ROS, or the VLMaps submodule. That allows both the
Habitat-side code and the ROS-side bridge to use the same schema
without coupling environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class GoalType(str, Enum):
    """Possible high-level goal types the strategist can emit."""

    ROOM = "ROOM"           # navigate to a room interior point
    APPROACH = "APPROACH"   # navigate to an approach point near a candidate
    INSPECT = "INSPECT"     # arrive + run a YOLOE check
    VERIFY = "VERIFY"       # local angular re-check after inspection failed


@dataclass
class SemanticGoal:
    """Schema shared by Habitat and ROS backends.

    Fields:
        type:          high-level goal kind, see :class:`GoalType`.
        map_pose:      (row, col) goal cell in the map grid (Habitat) or
                       (x, y) in metres in the /map frame (ROS). The
                       interpretation is fixed per backend at construction.
        room_id:       semantic room label assigned by the room labeler.
                       May be ``None`` for direct object goals.
        object_class:  target category for verification. Required for
                       INSPECT and VERIFY, optional for ROOM/APPROACH.
        metadata:      free-form dict for extra context (priors, scores,
                       parent component id, ...). Serialised as JSON when
                       crossing the ROS bridge.
    """

    type: GoalType
    map_pose: Tuple[float, float]
    room_id: Optional[str] = None
    object_class: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a primitive dict (JSON-safe)."""
        d = asdict(self)
        d["type"] = self.type.value
        d["map_pose"] = list(self.map_pose)
        return d

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SemanticGoal":
        """Construct from a primitive dict (round-trip with :meth:`to_dict`)."""
        pose = payload["map_pose"]
        if not isinstance(pose, (list, tuple)) or len(pose) != 2:
            raise ValueError(f"map_pose must be a 2-tuple, got {pose!r}")
        return cls(
            type=GoalType(payload["type"]),
            map_pose=(float(pose[0]), float(pose[1])),
            room_id=payload.get("room_id"),
            object_class=payload.get("object_class"),
            metadata=dict(payload.get("metadata") or {}),
        )
