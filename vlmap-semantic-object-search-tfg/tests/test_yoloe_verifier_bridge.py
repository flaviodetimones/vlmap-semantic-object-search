"""Offline tests for the ROS-side visual verification bridge helpers."""

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros_ws" / "src" / "yoloe_verifier" / "src"))

from yoloe_verifier import (
    build_verification_request_payload,
    decode_verification_result_payload,
    make_verification_response,
)


def test_build_verification_request_payload_round_trip():
    payload = build_verification_request_payload(
        request_id="req-1",
        object_class="bottle",
        rgb_topic="/rgb",
        depth_topic="/depth",
        camera_info_topic="/cam_info",
        point_cloud_topic="/points",
        metadata={"token": "nav-1"},
    )
    assert payload["request_id"] == "req-1"
    assert payload["object_class"] == "bottle"
    assert payload["metadata"]["token"] == "nav-1"


def test_decode_verification_result_payload_accepts_found_only():
    payload = decode_verification_result_payload(
        '{"request_id":"req-2","found":true,"message":"ok","metadata":{"score":0.8}}'
    )
    assert payload["request_id"] == "req-2"
    assert payload["success"] is True
    assert payload["found"] is True
    assert payload["metadata"]["score"] == pytest.approx(0.8)


def test_make_verification_response_is_json_ready():
    payload = make_verification_response(
        request_id="req-3",
        success=True,
        found=False,
        message="not visible",
        metadata={"worker": "mock"},
    )
    assert payload["request_id"] == "req-3"
    assert payload["success"] is True
    assert payload["found"] is False
    assert payload["metadata"]["worker"] == "mock"
