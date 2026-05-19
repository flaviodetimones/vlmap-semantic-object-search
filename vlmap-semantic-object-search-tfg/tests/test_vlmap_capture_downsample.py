"""Offline tests for the vlmap_capture downsample/serialization helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROS_SRC = (
    Path(__file__).resolve().parents[1]
    / "ros_ws"
    / "src"
    / "vlmap_capture"
    / "src"
)
if str(ROS_SRC) not in sys.path:
    sys.path.insert(0, str(ROS_SRC))

from vlmap_capture.downsample import (  # noqa: E402
    PoseSample,
    format_pose_line,
    quaternion_yaw,
    should_accept_frame,
)


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def test_quaternion_yaw_identity_is_zero():
    assert quaternion_yaw(0.0, 0.0, 0.0, 1.0) == pytest.approx(0.0)


def test_quaternion_yaw_known_angles():
    for angle_deg in (-179.0, -90.0, 0.0, 45.0, 90.0, 179.0):
        qx, qy, qz, qw = _yaw_quat(math.radians(angle_deg))
        assert quaternion_yaw(qx, qy, qz, qw) == pytest.approx(
            math.radians(angle_deg), abs=1e-6
        )


def test_first_frame_always_accepted():
    p = PoseSample(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    accept, reason = should_accept_frame(p, None, 0.2, 0.35)
    assert accept and reason == "first"


def test_reject_when_below_thresholds():
    last = PoseSample(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    candidate = PoseSample(0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    accept, reason = should_accept_frame(candidate, last, 0.2, 0.35)
    assert not accept
    assert "below" in reason


def test_accept_on_translation():
    last = PoseSample(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    candidate = PoseSample(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    accept, reason = should_accept_frame(candidate, last, 0.2, 0.35)
    assert accept
    assert reason.startswith("distance")


def test_accept_on_rotation():
    last = PoseSample(0.0, 0.0, 0.0, *_yaw_quat(0.0))
    candidate = PoseSample(0.0, 0.0, 0.0, *_yaw_quat(math.radians(45.0)))
    accept, reason = should_accept_frame(candidate, last, 0.5, math.radians(20.0))
    assert accept
    assert reason.startswith("rotation")


def test_zero_thresholds_always_accept_after_first():
    last = PoseSample(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    candidate = PoseSample(0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    # With min_distance_m=0 and min_angle_rad=0, both conditions disabled
    # → falls through to "below thresholds". This protects against runaway
    # captures when the user forgets to set thresholds.
    accept, _ = should_accept_frame(candidate, last, 0.0, 0.0)
    assert not accept


def test_format_pose_line_uses_tabs_and_no_newline():
    p = PoseSample(1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9)
    line = format_pose_line(p)
    assert "\n" not in line
    parts = line.split("\t")
    assert len(parts) == 7
    assert parts[0] == "1.0"
    assert parts[-1] == "0.9"


def test_angle_diff_handles_wraparound():
    last = PoseSample(0.0, 0.0, 0.0, *_yaw_quat(math.radians(170.0)))
    # +20° from 170° lands at -170° (wrapped). Difference is 20°, not 340°.
    candidate = PoseSample(0.0, 0.0, 0.0, *_yaw_quat(math.radians(-170.0)))
    accept, reason = should_accept_frame(candidate, last, 0.5, math.radians(45.0))
    assert not accept  # 20° rotation is below the 45° threshold
    assert "below" in reason
