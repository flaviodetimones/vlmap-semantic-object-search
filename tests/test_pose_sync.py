"""Offline tests for the Phase 4 pose-sync bridge (ROS -> Habitat).

Covers:
  - Rigid2D transform (identity, translation, rotation, round-trip inverse)
  - pose_from_result extraction from a NavigationResult and from a metadata dict
  - PoseSyncBridge.verify_at invokes the injected verifier at the Habitat pose
  - the tfg-ros side actually emits metadata.final_pose
"""

from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ros_ws" / "src" / "vlmap_task_manager" / "src"))

from tfg_nav_contracts import (
    NavigationResult,
    PoseSyncBridge,
    Rigid2D,
    pose_from_result,
)
from vlmap_task_manager import active_goal_from_fields, build_navigation_result_payload


def _close(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# ── Rigid2D ──────────────────────────────────────────────────────────────────
def test_identity_transform_is_noop():
    t = Rigid2D()
    assert _close(t.apply((1.0, 2.0, 0.5)), (1.0, 2.0, 0.5))


def test_translation_only():
    t = Rigid2D(tx=10.0, ty=-3.0)
    assert _close(t.apply((1.0, 2.0, 0.0)), (11.0, -1.0, 0.0))


def test_rotation_90deg():
    t = Rigid2D(theta=math.pi / 2)
    x, y, yaw = t.apply((1.0, 0.0, 0.0))
    assert _close((x, y), (0.0, 1.0), tol=1e-9)
    assert abs(yaw - math.pi / 2) < 1e-9


def test_inverse_round_trip():
    t = Rigid2D(tx=2.5, ty=-1.0, theta=0.7, scale=1.0)
    inv = t.inverse()
    p = (3.0, 4.0, 0.2)
    back = inv.apply(t.apply(p))
    assert _close(back, p, tol=1e-9)


# ── pose_from_result ─────────────────────────────────────────────────────────
def test_pose_from_navigation_result_object():
    res = NavigationResult(
        success=True,
        actual_room="office",
        metadata={"final_pose": {"x": 2.5, "y": 0.0, "yaw": 0.1}},
    )
    assert pose_from_result(res) == (2.5, 0.0, 0.1)


def test_pose_from_plain_metadata_dict():
    assert pose_from_result({"final_pose": {"x": 1.0, "y": -1.0}}) == (1.0, -1.0, 0.0)


def test_pose_absent_returns_none():
    assert pose_from_result(NavigationResult(success=True)) is None
    assert pose_from_result({"metadata": {}}) is None


# ── PoseSyncBridge ───────────────────────────────────────────────────────────
def test_verify_at_invokes_verifier_with_transformed_pose():
    calls = []

    def fake_verifier(pose, result):
        calls.append((pose, result))
        return {"found": True, "label": "laptop"}

    bridge = PoseSyncBridge(
        verifier=fake_verifier,
        transform=Rigid2D(tx=100.0, ty=0.0),  # map -> habitat offset
    )
    res = NavigationResult(
        success=True, metadata={"final_pose": {"x": 2.5, "y": 0.0, "yaw": 0.0}}
    )
    out = bridge.verify_at(res)
    assert out == {"found": True, "label": "laptop"}
    assert len(calls) == 1
    (pose, passed_result) = calls[0]
    assert _close(pose, (102.5, 0.0, 0.0))
    assert passed_result is res


def test_verify_at_without_pose_or_verifier_returns_none():
    # No verifier configured.
    bridge_no_verifier = PoseSyncBridge()
    res = NavigationResult(success=True, metadata={"final_pose": {"x": 1.0, "y": 1.0}})
    assert bridge_no_verifier.verify_at(res) is None
    # No pose in the result.
    bridge = PoseSyncBridge(verifier=lambda p, r: "x")
    assert bridge.verify_at(NavigationResult(success=True)) is None


# ── tfg-ros side emits final_pose ────────────────────────────────────────────
def test_result_payload_includes_final_pose():
    goal = active_goal_from_fields("office", "laptop", '{"navigation_token":"tok-fp"}')
    payload = build_navigation_result_payload(
        goal,
        status_code=3,
        status_text="Goal reached.",
        final_pose=(2.5, 0.0, 0.25),
    )
    assert payload["metadata"]["final_pose"] == {"x": 2.5, "y": 0.0, "yaw": 0.25}
    # And the full bridge can read it straight back.
    assert pose_from_result(payload) == (2.5, 0.0, 0.25)


def test_result_payload_tolerates_xy_only_pose():
    goal = active_goal_from_fields("kitchen", "mug", '{"navigation_token":"tok-xy"}')
    payload = build_navigation_result_payload(
        goal, status_code=3, status_text="ok", final_pose=(1.0, -2.0)
    )
    assert payload["metadata"]["final_pose"] == {"x": 1.0, "y": -2.0, "yaw": 0.0}
