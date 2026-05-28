"""Helpers for bridging ROS navigation results back into the shared contract.

This module is intentionally ROS-light so that the interesting logic can be
tested offline. The ROS node only performs message I/O and delegates payload
construction here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Optional


_SUCCESS_STATUS_CODES = {3}
_FAILURE_STATUS_CODES = {4, 5, 8, 9}


@dataclass(frozen=True)
class ActiveSemanticGoal:
    token: str
    room_id: str
    object_class: str
    metadata: Dict[str, Any]


def decode_semantic_goal_metadata(metadata_json: str) -> Dict[str, Any]:
    text = str(metadata_json or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"raw_metadata": text}
    return payload if isinstance(payload, dict) else {"raw_metadata": payload}


def active_goal_from_fields(room_id: str, object_class: str, metadata_json: str) -> Optional[ActiveSemanticGoal]:
    metadata = decode_semantic_goal_metadata(metadata_json)
    token = str(metadata.get("navigation_token") or metadata.get("token") or "").strip()
    if not token:
        return None
    return ActiveSemanticGoal(
        token=token,
        room_id=str(room_id or ""),
        object_class=str(object_class or ""),
        metadata=metadata,
    )


def move_base_status_to_success(status_code: int) -> Optional[bool]:
    code = int(status_code)
    if code in _SUCCESS_STATUS_CODES:
        return True
    if code in _FAILURE_STATUS_CODES:
        return False
    return None


def build_navigation_result_payload(
    active_goal: ActiveSemanticGoal,
    *,
    status_code: int,
    status_text: str,
    source: str = "/move_base/result",
    resolved_room: Optional[str] = None,
) -> Dict[str, Any]:
    success = move_base_status_to_success(status_code)
    if success is None:
        raise ValueError(f"status code {status_code} is not terminal")
    metadata = dict(active_goal.metadata)
    metadata.update(
        {
            "result_source": source,
            "result_status_code": int(status_code),
        }
    )
    # Prefer the room derived from the robot's actual final pose (live room
    # context). Fall back to the requested room when no live context is
    # available, and record both so the difference is auditable.
    requested_room = active_goal.room_id or None
    actual_room = resolved_room if resolved_room else requested_room
    if resolved_room:
        metadata["requested_room"] = requested_room
        metadata["room_source"] = "pose_resolved"
    else:
        metadata["room_source"] = "requested_echo"
    return {
        "token": active_goal.token,
        "success": success,
        "found": success,
        "actual_room": actual_room,
        "message": str(status_text or ""),
        "metadata": metadata,
        "object_class": active_goal.object_class or None,
    }
