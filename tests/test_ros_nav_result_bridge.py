"""Offline tests for the ROS navigation-result bridge helpers."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros_ws" / "src" / "vlmap_task_manager" / "src"))

from vlmap_task_manager import (
    active_goal_from_fields,
    build_navigation_result_payload,
    decode_semantic_goal_metadata,
    move_base_status_to_success,
)


def test_decode_semantic_goal_metadata_handles_json_and_garbage():
    assert decode_semantic_goal_metadata("") == {}
    assert decode_semantic_goal_metadata('{"navigation_token":"tok","room":"office"}') == {
        "navigation_token": "tok",
        "room": "office",
    }
    assert decode_semantic_goal_metadata("not-json") == {"raw_metadata": "not-json"}


def test_active_goal_requires_navigation_token():
    assert active_goal_from_fields("office", "laptop", '{"foo": 1}') is None
    goal = active_goal_from_fields(
        "office",
        "laptop",
        '{"navigation_token":"tok-1","selected_room_score":0.9}',
    )
    assert goal is not None
    assert goal.token == "tok-1"
    assert goal.room_id == "office"
    assert goal.object_class == "laptop"


def test_move_base_status_mapping_and_payload():
    goal = active_goal_from_fields(
        "kitchen",
        "mug",
        '{"navigation_token":"tok-2","parsed_target":"mug"}',
    )
    assert goal is not None
    assert move_base_status_to_success(3) is True
    assert move_base_status_to_success(4) is False
    assert move_base_status_to_success(1) is None

    payload = build_navigation_result_payload(
        goal,
        status_code=3,
        status_text="Goal reached.",
    )
    assert payload["token"] == "tok-2"
    assert payload["success"] is True
    assert payload["found"] is True
    assert payload["actual_room"] == "kitchen"
    assert payload["object_class"] == "mug"
    assert payload["metadata"]["parsed_target"] == "mug"
    assert payload["metadata"]["result_status_code"] == 3
