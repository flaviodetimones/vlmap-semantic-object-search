"""Pure-Python helpers for the vlmap_capture node (no rospy / cv_bridge here).

Keeping these out of the node script lets us test acceptance and serialization
logic offline, without spinning up a ROS master.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PoseSample:
    """Minimal 3D pose sample. Quaternion is ``(qx, qy, qz, qw)``."""

    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


def quaternion_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw (rotation around Z) extracted from a quaternion, radians."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _angle_diff(a: float, b: float) -> float:
    """Smallest signed difference ``a - b`` wrapped to ``[-pi, pi]``."""
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def should_accept_frame(
    candidate: PoseSample,
    last_accepted: Optional[PoseSample],
    min_distance_m: float,
    min_angle_rad: float,
) -> Tuple[bool, str]:
    """Decide whether to keep ``candidate`` based on travel since last accepted.

    Returns ``(accept, reason)``. The first frame is always accepted. After
    that, we accept if the robot has translated by ``min_distance_m`` OR
    rotated (yaw) by ``min_angle_rad``. Set both to 0 to keep every frame
    (useful for short scripted runs).
    """
    if last_accepted is None:
        return True, "first"
    dx = candidate.x - last_accepted.x
    dy = candidate.y - last_accepted.y
    travel = math.hypot(dx, dy)
    if min_distance_m > 0 and travel >= min_distance_m:
        return True, f"distance {travel:.3f}m"
    yaw_a = quaternion_yaw(candidate.qx, candidate.qy, candidate.qz, candidate.qw)
    yaw_b = quaternion_yaw(
        last_accepted.qx, last_accepted.qy, last_accepted.qz, last_accepted.qw
    )
    rot = abs(_angle_diff(yaw_a, yaw_b))
    if min_angle_rad > 0 and rot >= min_angle_rad:
        return True, f"rotation {math.degrees(rot):.1f}deg"
    return False, "below thresholds"


def format_pose_line(pose: PoseSample) -> str:
    """Single line of ``poses.txt`` in the VLMapBuilder convention.

    Tab-separated ``tx\\tty\\ttz\\tqx\\tqy\\tqz\\tqw``, no trailing newline.
    """
    return (
        f"{pose.x}\t{pose.y}\t{pose.z}\t"
        f"{pose.qx}\t{pose.qy}\t{pose.qz}\t{pose.qw}"
    )
