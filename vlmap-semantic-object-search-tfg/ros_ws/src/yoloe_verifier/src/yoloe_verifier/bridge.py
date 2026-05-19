"""ROS-light helpers for request/response visual verification bridging."""

from __future__ import annotations

import json
from typing import Any, Dict


def build_verification_request_payload(
    *,
    request_id: str,
    object_class: str,
    rgb_topic: str,
    depth_topic: str,
    camera_info_topic: str,
    point_cloud_topic: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "request_id": str(request_id),
        "object_class": str(object_class),
        "rgb_topic": str(rgb_topic),
        "depth_topic": str(depth_topic),
        "camera_info_topic": str(camera_info_topic),
        "point_cloud_topic": str(point_cloud_topic),
        "metadata": dict(metadata),
    }


def decode_verification_result_payload(raw_text: str) -> Dict[str, Any]:
    payload = json.loads(str(raw_text or ""))
    if not isinstance(payload, dict):
        raise ValueError("verification result must decode to a JSON object")
    request_id = str(payload.get("request_id") or "").strip()
    if not request_id:
        raise ValueError("verification result missing request_id")
    return {
        "request_id": request_id,
        "success": bool(payload.get("success", payload.get("found", False))),
        "found": bool(payload.get("found", payload.get("success", False))),
        "message": str(payload.get("message", "")),
        "metadata": dict(payload.get("metadata") or {}),
    }


def make_verification_response(
    *,
    request_id: str,
    success: bool,
    found: bool,
    message: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "request_id": str(request_id),
        "success": bool(success),
        "found": bool(found),
        "message": str(message),
        "metadata": dict(metadata),
    }
