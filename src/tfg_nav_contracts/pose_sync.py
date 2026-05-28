"""Pose synchronisation between the ROS navigation frame and Habitat.

Phase 4, Option A: ROS only navigates; perception/verification stays in
Habitat (tfg-sim). After the HSR reaches a goal in Gazebo, ``move_base``
returns the robot's real final pose inside ``navigation_result.metadata``
(``final_pose``). This module turns that pose into a Habitat-frame pose and
hands it to an injected verifier (e.g. teleport the Habitat agent + run YOLOE).

Everything here is pure and ROS-free so it can be unit-tested offline; the
heavy Habitat/YOLOE integration is supplied as a callback by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Optional, Tuple

# (x, y, yaw) in some 2D frame.
Pose2D = Tuple[float, float, float]


@dataclass(frozen=True)
class Rigid2D:
    """A 2D rigid (optionally scaled) transform: rotate by ``theta`` (rad),
    scale by ``scale``, then translate by ``(tx, ty)``.

    Used to express ``T_habitat_from_map``: the relation between the ROS
    ``map`` frame (where ``move_base`` reports poses) and the Habitat world
    frame (where the agent and the placed objects live). For the current
    empty-Gazebo setup the two frames coincide, so the identity is the
    sensible default; a non-trivial transform is only needed once Gazebo
    loads a map with a different origin/orientation.
    """

    tx: float = 0.0
    ty: float = 0.0
    theta: float = 0.0
    scale: float = 1.0

    def apply(self, pose: Pose2D) -> Pose2D:
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        c, s = math.cos(self.theta), math.sin(self.theta)
        rx = self.scale * (c * x - s * y)
        ry = self.scale * (s * x + c * y)
        nx = rx + self.tx
        ny = ry + self.ty
        nyaw = _wrap_angle(yaw + self.theta)
        return (nx, ny, nyaw)

    def inverse(self) -> "Rigid2D":
        # Inverse of: p' = R(theta)*scale*p + t  =>  p = (1/scale) R(-theta) (p' - t)
        inv_scale = 1.0 / self.scale if self.scale else 1.0
        c, s = math.cos(-self.theta), math.sin(-self.theta)
        # New translation maps origin back: t_inv = -(1/scale) R(-theta) t
        tx = -inv_scale * (c * self.tx - s * self.ty)
        ty = -inv_scale * (s * self.tx + c * self.ty)
        return Rigid2D(tx=tx, ty=ty, theta=-self.theta, scale=inv_scale)


def _wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def pose_from_result(result: Any) -> Optional[Pose2D]:
    """Extract ``(x, y, yaw)`` from a NavigationResult or its metadata dict.

    Looks for ``metadata['final_pose'] = {x, y, yaw}``. Returns None when the
    pose is absent (e.g. older ROS side that did not emit it).
    """
    metadata: Optional[Mapping[str, Any]]
    if isinstance(result, Mapping):
        metadata = result.get("metadata", result)
    else:
        metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    fp = metadata.get("final_pose")
    if not isinstance(fp, Mapping):
        return None
    try:
        return (float(fp["x"]), float(fp["y"]), float(fp.get("yaw", 0.0)))
    except (KeyError, TypeError, ValueError):
        return None


class PoseSyncBridge:
    """Bridge a ROS navigation result into a Habitat-frame verification call.

    Parameters
    ----------
    verifier:
        Callback ``verifier(habitat_pose, result) -> Any``. In production this
        teleports the Habitat agent to ``habitat_pose`` and runs YOLOE; in
        tests it is a fake that just records the pose. May be omitted to use
        the bridge purely for the frame transform.
    transform:
        ``T_habitat_from_map``. Defaults to identity.
    """

    def __init__(
        self,
        *,
        verifier: Optional[Callable[[Pose2D, Any], Any]] = None,
        transform: Optional[Rigid2D] = None,
    ) -> None:
        self.verifier = verifier
        self.transform = transform or Rigid2D()

    def habitat_pose(self, result: Any) -> Optional[Pose2D]:
        """Return the Habitat-frame pose for a navigation result, or None."""
        map_pose = pose_from_result(result)
        if map_pose is None:
            return None
        return self.transform.apply(map_pose)

    def verify_at(self, result: Any) -> Any:
        """Resolve the Habitat pose and invoke the verifier there.

        Returns the verifier's output, or None when there is no pose to verify
        or no verifier was configured.
        """
        pose = self.habitat_pose(result)
        if pose is None or self.verifier is None:
            return None
        return self.verifier(pose, result)
