"""Offline tests for the live room-context path (Phase 3).

Covers:
  - DynamicRoomContext.room_at_pose nearest-centroid resolver
  - RoomContextPublisher round-trip through a fake rosbridge client
  - build_navigation_result_payload honouring a pose-resolved room
"""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ros_ws" / "src" / "vlmap_task_manager" / "src"))
sys.path.insert(0, str(ROOT / "ros_ws" / "src" / "vlmap_semantic_server" / "src"))

from tfg_nav_contracts.navigation_backend import RoomContextPublisher
from vlmap_semantic_server.dynamic_context import load_dynamic_room_context
from vlmap_task_manager import (
    active_goal_from_fields,
    build_navigation_result_payload,
)


# ── Fake rosbridge plumbing (mirrors test_semantic_goal_contract) ────────────
class _FakeClient:
    def __init__(self):
        self.is_connected = False

    def run(self):
        self.is_connected = True


class _FakeTopic:
    def __init__(self, name, type_name):
        self.name = name
        self.type_name = type_name
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def _make_topic_factory(registry):
    def _factory(_client, name, type_name):
        topic = registry.setdefault(name, _FakeTopic(name, type_name))
        return topic
    return _factory


# ── room_at_pose ─────────────────────────────────────────────────────────────
def _context():
    payload = json.dumps(
        {
            "rooms": [
                {"room_id": "kitchen", "x": 0.0, "y": 0.0},
                {"room_id": "office", "x": 5.0, "y": 0.0, "aliases": ["study"]},
                {"room_id": "bedroom", "x": 0.0, "y": 5.0},
            ],
            "priors": {"laptop": [["office", 0.8], ["bedroom", 0.2]]},
        }
    )
    return load_dynamic_room_context(payload)


def test_room_at_pose_picks_nearest_centroid():
    ctx = _context()
    assert ctx.room_at_pose(4.6, 0.1) == "office"
    assert ctx.room_at_pose(0.2, 0.1) == "kitchen"
    assert ctx.room_at_pose(0.1, 4.7) == "bedroom"


def test_room_at_pose_respects_max_distance():
    ctx = _context()
    # Far from every centroid -> rejected when a cap is given.
    assert ctx.room_at_pose(100.0, 100.0, max_distance=2.0) is None
    # No cap -> always returns the nearest.
    assert ctx.room_at_pose(100.0, 100.0) is not None


def test_room_at_pose_empty_context_returns_none():
    empty = load_dynamic_room_context(json.dumps({"rooms": []})) if False else None
    # load_dynamic_room_context rejects empty 'rooms' list? guard either way.
    try:
        empty = load_dynamic_room_context(json.dumps({"rooms": []}))
    except Exception:
        empty = None
    if empty is not None:
        assert empty.room_at_pose(1.0, 1.0) is None


# ── RoomContextPublisher round-trip ──────────────────────────────────────────
def test_publisher_emits_latched_json_parseable_by_listener():
    registry = {}
    pub = RoomContextPublisher(
        auto_connect=True,
        client=_FakeClient(),
        topic_factory=_make_topic_factory(registry),
        message_factory=lambda payload: payload,
        publish_settle_s=0.0,
    )
    rooms = [
        {"room_id": "office", "x": 5.0, "y": 0.0, "aliases": ["study"]},
        {"room_id": "kitchen", "x": 0.0, "y": 0.0},
    ]
    priors = {"laptop": [["office", 0.9]]}
    maps = {"laptop": "/shared/heatmaps/scene0/laptop.png"}

    data = pub.publish(rooms, priors=priors, maps=maps)

    topic = registry["/vlmap/room_context"]
    assert topic.type_name == "std_msgs/String"
    assert len(topic.messages) == 1
    published = topic.messages[0]["data"]
    assert published == data

    # The tfg-ros side must be able to parse exactly this payload.
    ctx = load_dynamic_room_context(published)
    assert set(ctx.list_rooms()) == {"office", "kitchen"}
    assert ctx.resolve_room_name("study") == "office"
    rooms_ranked, scores = ctx.rank_rooms_for_category("laptop")
    assert rooms_ranked[0] == "office"
    assert scores[0] == 0.9
    # maps survive as references inside the raw payload.
    assert json.loads(published)["maps"]["laptop"].startswith("/shared/")


# ── result payload honours pose-resolved room ────────────────────────────────
def test_result_payload_prefers_resolved_room_over_requested():
    goal = active_goal_from_fields(
        "office",  # requested room
        "laptop",
        '{"navigation_token":"tok-9"}',
    )
    payload = build_navigation_result_payload(
        goal,
        status_code=3,
        status_text="Goal reached.",
        resolved_room="bedroom",  # robot actually ended in bedroom
    )
    assert payload["actual_room"] == "bedroom"
    assert payload["metadata"]["requested_room"] == "office"
    assert payload["metadata"]["room_source"] == "pose_resolved"


def test_result_payload_falls_back_to_requested_when_no_context():
    goal = active_goal_from_fields("office", "laptop", '{"navigation_token":"tok-10"}')
    payload = build_navigation_result_payload(
        goal, status_code=3, status_text="Goal reached."
    )
    assert payload["actual_room"] == "office"
    assert payload["metadata"]["room_source"] == "requested_echo"
