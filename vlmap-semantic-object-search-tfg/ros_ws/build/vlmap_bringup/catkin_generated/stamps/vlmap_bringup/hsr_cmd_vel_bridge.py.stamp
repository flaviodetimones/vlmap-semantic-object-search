#!/usr/bin/env python3
"""Bridge the project-level HSR cmd_vel topic into the active base controller."""

from __future__ import annotations

import rospy
from geometry_msgs.msg import Twist


class CmdVelBridge:
    def __init__(self) -> None:
        self.input_topic = rospy.get_param("~input_topic", "/hsrb/command_velocity")
        self.output_topic = rospy.get_param(
            "~output_topic", "/hsrb/base_velocity_controller/cmd_vel"
        )
        self.publisher = rospy.Publisher(self.output_topic, Twist, queue_size=1)
        self.subscriber = rospy.Subscriber(self.input_topic, Twist, self._callback, queue_size=1)
        rospy.loginfo(
            "[hsr_cmd_vel_bridge] %s -> %s",
            self.input_topic,
            self.output_topic,
        )

    def _callback(self, msg: Twist) -> None:
        self.publisher.publish(msg)


def main() -> None:
    rospy.init_node("hsr_cmd_vel_bridge")
    CmdVelBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
