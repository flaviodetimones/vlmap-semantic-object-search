#!/usr/bin/env python3
"""Keep the simulated HSR upper body in a stable posture for nav-only demos."""

from __future__ import annotations

import math
from typing import Dict, List

import rospy
from gazebo_msgs.srv import GetModelState, SetModelConfiguration


_PROFILES: Dict[str, Dict[str, float]] = {
    # Compact, visually stable pose for navigation and sensor validation.
    "nav": {
        "arm_lift_joint": 0.0,
        "arm_flex_joint": 0.0,
        "arm_roll_joint": 0.0,
        "wrist_flex_joint": -1.57,
        "wrist_roll_joint": 0.0,
        "head_pan_joint": 0.0,
        "head_tilt_joint": -0.35,
    },
    # Slightly more open display pose for Gazebo/RViz inspection.
    "display": {
        "arm_lift_joint": 0.12,
        "arm_flex_joint": -0.55,
        "arm_roll_joint": 0.0,
        "wrist_flex_joint": -1.10,
        "wrist_roll_joint": 0.0,
        "head_pan_joint": 0.0,
        "head_tilt_joint": -0.28,
    },
}


class HsrPostureKeeper:
    def __init__(self) -> None:
        self.model_name = rospy.get_param("~model_name", "hsrb")
        self.profile_name = rospy.get_param("~profile", "display")
        self.rate_hz = float(rospy.get_param("~rate_hz", 2.0))
        self.ready_timeout = float(rospy.get_param("~ready_timeout_sec", 30.0))
        self._joint_map = _PROFILES.get(self.profile_name, _PROFILES["display"])
        self._joint_names: List[str] = list(self._joint_map.keys())
        self._joint_positions: List[float] = [self._joint_map[name] for name in self._joint_names]

        rospy.loginfo(
            "[hsr_posture_keeper] profile=%s model=%s joints=%s",
            self.profile_name,
            self.model_name,
            ",".join(self._joint_names),
        )

        rospy.wait_for_service("/gazebo/get_model_state")
        rospy.wait_for_service("/gazebo/set_model_configuration")
        self._get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self._set_model_configuration = rospy.ServiceProxy(
            "/gazebo/set_model_configuration", SetModelConfiguration
        )

    def _model_exists(self) -> bool:
        try:
            resp = self._get_model_state(self.model_name, "")
            return bool(resp.success)
        except rospy.ServiceException:
            return False

    def _wait_until_spawned(self) -> bool:
        deadline = rospy.Time.now() + rospy.Duration.from_sec(self.ready_timeout)
        rate = rospy.Rate(5.0)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self._model_exists():
                return True
            rate.sleep()
        return False

    def _apply(self) -> None:
        try:
            self._set_model_configuration(
                model_name=self.model_name,
                urdf_param_name="robot_description",
                joint_names=self._joint_names,
                joint_positions=self._joint_positions,
            )
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(5.0, "[hsr_posture_keeper] set_model_configuration failed: %s", exc)

    def spin(self) -> None:
        if not self._wait_until_spawned():
            rospy.logwarn("[hsr_posture_keeper] model '%s' not found within timeout", self.model_name)
            return

        rate = rospy.Rate(max(self.rate_hz, 0.2))
        while not rospy.is_shutdown():
            self._apply()
            rate.sleep()


def main() -> None:
    rospy.init_node("hsr_posture_keeper")
    HsrPostureKeeper().spin()


if __name__ == "__main__":
    main()
