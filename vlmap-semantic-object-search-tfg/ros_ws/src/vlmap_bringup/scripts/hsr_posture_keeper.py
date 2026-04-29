#!/usr/bin/env python3
"""Keep the simulated HSR upper body in a stable posture for demos."""

from __future__ import annotations

from typing import Dict

import rospy
from std_msgs.msg import Float64


_PROFILES: Dict[str, Dict[str, float]] = {
    # Compact, visually stable pose for navigation and sensor validation.
    "nav": {
        "base_roll_joint": 0.0,
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
        "base_roll_joint": 0.0,
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
        self.controller_namespace = rospy.get_param("~controller_namespace", "/hsrb")
        self.profile_name = rospy.get_param("~profile", "display")
        self.rate_hz = float(rospy.get_param("~rate_hz", 2.0))
        self._joint_map = _PROFILES.get(self.profile_name, _PROFILES["display"])
        self._publishers = {}

        rospy.loginfo(
            "[hsr_posture_keeper] profile=%s ns=%s joints=%s",
            self.profile_name,
            self.controller_namespace,
            ",".join(self._joint_map.keys()),
        )

        for joint_name in self._joint_map:
            controller_name = joint_name.replace("_joint", "_controller")
            topic = f"{self.controller_namespace}/{controller_name}/command"
            self._publishers[joint_name] = rospy.Publisher(topic, Float64, queue_size=1)

    def _apply(self) -> None:
        for joint_name, target in self._joint_map.items():
            self._publishers[joint_name].publish(Float64(data=float(target)))

    def spin(self) -> None:
        rate = rospy.Rate(max(self.rate_hz, 0.2))
        while not rospy.is_shutdown():
            self._apply()
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break


def main() -> None:
    rospy.init_node("hsr_posture_keeper")
    HsrPostureKeeper().spin()


if __name__ == "__main__":
    main()
