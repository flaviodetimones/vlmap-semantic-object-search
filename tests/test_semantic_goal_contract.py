"""Contract tests for the backend-neutral semantic navigation layer."""

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfg_nav_contracts import (
    GoalType,
    HabitatFollowerBackend,
    NavigationBackend,
    NavigationResult,
    RosNavigationBackend,
    SemanticGoal,
)


def test_semantic_goal_to_dict_round_trip():
    goal = SemanticGoal(
        type=GoalType.INSPECT,
        map_pose=(123.5, 45.0),
        room_id="kitchen",
        object_class="bottle",
        metadata={"prior": 0.7, "parent_component": 3},
    )
    payload = goal.to_dict()
    encoded = json.dumps(payload)  # must be JSON-serialisable
    decoded = json.loads(encoded)
    restored = SemanticGoal.from_dict(decoded)

    assert restored.type is GoalType.INSPECT
    assert restored.map_pose == (123.5, 45.0)
    assert restored.room_id == "kitchen"
    assert restored.object_class == "bottle"
    assert restored.metadata["prior"] == pytest.approx(0.7)
    assert restored.metadata["parent_component"] == 3


def test_semantic_goal_defaults_and_validation():
    goal = SemanticGoal(type=GoalType.ROOM, map_pose=(0.0, 0.0))
    assert goal.room_id is None
    assert goal.object_class is None
    assert goal.metadata == {}

    bad = goal.to_dict()
    bad["map_pose"] = "not-a-pose"
    with pytest.raises(ValueError):
        SemanticGoal.from_dict(bad)


def test_habitat_backend_is_stubbed():
    backend = HabitatFollowerBackend()
    assert isinstance(backend, NavigationBackend)
    goal = SemanticGoal(type=GoalType.APPROACH, map_pose=(1.0, 2.0))
    with pytest.raises(NotImplementedError):
        backend.submit_goal(goal)
    with pytest.raises(NotImplementedError):
        backend.wait_result("token-123")
    with pytest.raises(NotImplementedError):
        backend.cancel("token-123")


def test_navigation_result_defaults():
    res = NavigationResult(success=True)
    assert res.success is True
    assert res.found is False
    assert res.actual_room is None
    assert res.metadata == {}


class _FakeClient:
    def __init__(self):
        self.is_connected = False
        self.run_calls = 0

    def run(self):
        self.is_connected = True
        self.run_calls += 1


class _FakeTopic:
    def __init__(self, name, type_name):
        self.name = name
        self.type_name = type_name
        self.messages = []
        self._callback = None

    def publish(self, message):
        self.messages.append(message)

    def subscribe(self, callback):
        self._callback = callback

    def emit(self, message):
        assert self._callback is not None
        self._callback(message)


def _make_topic_factory(registry):
    def _factory(_client, name, type_name):
        topic = registry.get(name)
        if topic is None:
            topic = _FakeTopic(name, type_name)
            registry[name] = topic
        return topic

    return _factory


def test_ros_backend_publishes_semantic_and_nav_goals():
    registry = {}
    backend = RosNavigationBackend(
        auto_connect=True,
        client=_FakeClient(),
        topic_factory=_make_topic_factory(registry),
        message_factory=lambda payload: payload,
        uuid_factory=lambda: "tok-1",
    )
    goal = SemanticGoal(
        type=GoalType.ROOM,
        map_pose=(1.25, -2.5),
        room_id="office",
        object_class="laptop",
        metadata={"prior": 0.91},
    )

    token = backend.submit_goal(goal)

    assert token == "tok-1"
    assert registry["/vlmap/semantic_goal"].messages
    semantic_payload = registry["/vlmap/semantic_goal"].messages[0]
    nav_payload = registry["/move_base_simple/goal"].messages[0]

    assert json.loads(semantic_payload["metadata"])["navigation_token"] == "tok-1"
    assert semantic_payload["room_id"] == "office"
    assert semantic_payload["object_class"] == "laptop"
    assert nav_payload["pose"]["position"]["x"] == pytest.approx(1.25)
    assert nav_payload["pose"]["position"]["y"] == pytest.approx(-2.5)


def test_ros_backend_uses_metadata_yaw_for_pose_orientation():
    registry = {}
    backend = RosNavigationBackend(
        auto_connect=True,
        client=_FakeClient(),
        topic_factory=_make_topic_factory(registry),
        message_factory=lambda payload: payload,
        uuid_factory=lambda: "tok-yaw",
    )
    goal = SemanticGoal(
        type=GoalType.APPROACH,
        map_pose=(0.5, 1.0),
        metadata={"yaw": math.pi / 2.0},
    )

    backend.submit_goal(goal)

    nav_payload = registry["/move_base_simple/goal"].messages[0]
    orientation = nav_payload["pose"]["orientation"]
    assert orientation["z"] == pytest.approx(math.sin(math.pi / 4.0))
    assert orientation["w"] == pytest.approx(math.cos(math.pi / 4.0))


def test_ros_backend_waits_on_json_result_topic():
    registry = {}
    backend = RosNavigationBackend(
        client=_FakeClient(),
        topic_factory=_make_topic_factory(registry),
        message_factory=lambda payload: payload,
        uuid_factory=lambda: "tok-json",
        result_topic="/vlmap/navigation_result",
        result_topic_type="std_msgs/String",
    )
    token = backend.submit_goal(SemanticGoal(type=GoalType.INSPECT, map_pose=(0.0, 0.0)))

    registry["/vlmap/navigation_result"].emit(
        {
            "data": json.dumps(
                {
                    "token": token,
                    "success": True,
                    "found": True,
                    "actual_room": "kitchen",
                    "message": "verified",
                    "metadata": {"source": "test"},
                }
            )
        }
    )
    result = backend.wait_result(token, timeout_s=0.01)
    assert result.success is True
    assert result.found is True
    assert result.actual_room == "kitchen"
    assert result.metadata["source"] == "test"


def test_ros_backend_maps_move_base_results_to_oldest_pending_goal():
    registry = {}
    backend = RosNavigationBackend(
        client=_FakeClient(),
        topic_factory=_make_topic_factory(registry),
        message_factory=lambda payload: payload,
        uuid_factory=lambda: "tok-move-base",
        sleep_fn=lambda _secs: None,
        result_topic="/move_base/result",
        result_topic_type="move_base_msgs/MoveBaseActionResult",
    )
    token = backend.submit_goal(SemanticGoal(type=GoalType.APPROACH, map_pose=(3.0, 4.0)))

    registry["/move_base/result"].emit(
        {
            "status": {
                "status": 3,
                "text": "Goal reached.",
                "goal_id": {"id": ""},
            }
        }
    )
    result = backend.wait_result(token, timeout_s=0.01)
    assert result.success is True
    assert result.found is True
    assert result.metadata["status_code"] == 3


def test_ros_backend_cancel_publishes_goal_id_and_returns_cancelled_result():
    registry = {}
    backend = RosNavigationBackend(
        client=_FakeClient(),
        topic_factory=_make_topic_factory(registry),
        message_factory=lambda payload: payload,
        uuid_factory=lambda: "tok-cancel",
    )
    token = backend.submit_goal(SemanticGoal(type=GoalType.ROOM, map_pose=(0.0, 1.0)))
    backend.cancel(token)

    cancel_payload = registry["/move_base/cancel"].messages[0]
    assert cancel_payload["id"] == token

    result = backend.wait_result(token, timeout_s=0.01)
    assert result.success is False
    assert result.message == "Cancelled before completion."
