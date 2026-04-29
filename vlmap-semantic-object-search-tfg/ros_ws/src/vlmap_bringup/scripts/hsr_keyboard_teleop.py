#!/usr/bin/env python3
"""Simple WASD keyboard teleop for the simulated HSR base."""

from __future__ import annotations

import select
import sys
import termios
import tty

import rospy
from geometry_msgs.msg import Twist


HELP = """
HSR keyboard teleop
  w / s : forward / backward
  a / d : turn left / right
  x     : stop
  q     : quit
"""


def _get_key(timeout: float) -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.read(1)
        return ""
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main() -> None:
    rospy.init_node("hsr_keyboard_teleop")
    topic = rospy.get_param("~cmd_vel_topic", "/hsrb/command_velocity")
    linear_speed = float(rospy.get_param("~linear_speed", 0.25))
    angular_speed = float(rospy.get_param("~angular_speed", 0.7))
    publish_rate = float(rospy.get_param("~publish_rate_hz", 10.0))

    pub = rospy.Publisher(topic, Twist, queue_size=1)
    twist = Twist()
    rate = rospy.Rate(max(publish_rate, 1.0))

    print(HELP.strip())
    print(f"\nPublishing to {topic}")

    while not rospy.is_shutdown():
        key = _get_key(0.1)
        if key == "w":
            twist.linear.x = linear_speed
            twist.angular.z = 0.0
        elif key == "s":
            twist.linear.x = -linear_speed
            twist.angular.z = 0.0
        elif key == "a":
            twist.linear.x = 0.0
            twist.angular.z = angular_speed
        elif key == "d":
            twist.linear.x = 0.0
            twist.angular.z = -angular_speed
        elif key == "x":
            twist = Twist()
        elif key == "q":
            pub.publish(Twist())
            print("\nTeleop stopped.")
            return

        pub.publish(twist)
        rate.sleep()


if __name__ == "__main__":
    main()
