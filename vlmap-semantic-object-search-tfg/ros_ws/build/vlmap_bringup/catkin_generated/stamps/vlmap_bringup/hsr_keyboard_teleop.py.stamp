#!/usr/bin/env python3
"""Keyboard teleop for the official simulated HSR base."""

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
  e / q : forward-left / forward-right
  c / z : backward-left / backward-right
  x     : stop
  r / f : increase / decrease linear speed
  t / g : increase / decrease angular speed
  v     : print current speeds
  p     : quit
"""


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


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
    linear_step = float(rospy.get_param("~linear_step", 0.05))
    angular_step = float(rospy.get_param("~angular_step", 0.1))
    max_linear_speed = float(rospy.get_param("~max_linear_speed", 0.6))
    max_angular_speed = float(rospy.get_param("~max_angular_speed", 1.5))
    publish_rate = float(rospy.get_param("~publish_rate_hz", 10.0))

    pub = rospy.Publisher(topic, Twist, queue_size=1)
    twist = Twist()
    rate = rospy.Rate(max(publish_rate, 1.0))

    print(HELP.strip())
    print(f"\nPublishing to {topic}")
    print(f"Initial speeds: linear={linear_speed:.2f} angular={angular_speed:.2f}")

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
        elif key == "e":
            twist.linear.x = linear_speed
            twist.angular.z = angular_speed
        elif key == "q":
            twist.linear.x = linear_speed
            twist.angular.z = -angular_speed
        elif key == "z":
            twist.linear.x = -linear_speed
            twist.angular.z = angular_speed
        elif key == "c":
            twist.linear.x = -linear_speed
            twist.angular.z = -angular_speed
        elif key == "x":
            twist = Twist()
        elif key == "r":
            linear_speed = _clamp(linear_speed + linear_step, linear_step, max_linear_speed)
            print(f"\rlinear speed -> {linear_speed:.2f} m/s      ", end="", flush=True)
        elif key == "f":
            linear_speed = _clamp(linear_speed - linear_step, linear_step, max_linear_speed)
            print(f"\rlinear speed -> {linear_speed:.2f} m/s      ", end="", flush=True)
        elif key == "t":
            angular_speed = _clamp(angular_speed + angular_step, angular_step, max_angular_speed)
            print(f"\rangular speed -> {angular_speed:.2f} rad/s      ", end="", flush=True)
        elif key == "g":
            angular_speed = _clamp(angular_speed - angular_step, angular_step, max_angular_speed)
            print(f"\rangular speed -> {angular_speed:.2f} rad/s      ", end="", flush=True)
        elif key == "v":
            print(f"\rlinear={linear_speed:.2f} angular={angular_speed:.2f}      ", end="", flush=True)
        elif key == "p":
            pub.publish(Twist())
            print("\nTeleop stopped.")
            return

        pub.publish(twist)
        rate.sleep()


if __name__ == "__main__":
    main()
