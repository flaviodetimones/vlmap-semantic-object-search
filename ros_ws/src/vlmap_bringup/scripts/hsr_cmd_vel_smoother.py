#!/usr/bin/env python3
"""Clamp and smooth HSR velocity commands before they reach Gazebo."""

from __future__ import annotations

import math

import rospy
from geometry_msgs.msg import Twist


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _approach(current: float, target: float, max_delta: float) -> float:
    return current + _clamp(target - current, -max_delta, max_delta)


class HsrCmdVelSmoother:
    def __init__(self) -> None:
        self.input_topic = rospy.get_param("~input_topic", "/hsrb/command_velocity_raw")
        self.output_topic = rospy.get_param("~output_topic", "/hsrb/command_velocity")
        self.rate_hz = float(rospy.get_param("~rate_hz", 20.0))
        self.timeout = float(rospy.get_param("~timeout", 0.35))
        self.max_linear = float(rospy.get_param("~max_linear", 0.12))
        self.max_angular = float(rospy.get_param("~max_angular", 0.09))
        self.max_linear_accel = float(rospy.get_param("~max_linear_accel", 0.08))
        self.max_angular_accel = float(rospy.get_param("~max_angular_accel", 0.06))

        self._target = Twist()
        self._current = Twist()
        self._last_msg_time = rospy.Time(0)
        self._last_tick = rospy.Time.now()

        self._pub = rospy.Publisher(self.output_topic, Twist, queue_size=1)
        self._sub = rospy.Subscriber(self.input_topic, Twist, self._on_twist, queue_size=1)
        rospy.loginfo(
            "[hsr_cmd_vel_smoother] %s -> %s max=(%.2f m/s, %.2f rad/s) accel=(%.2f, %.2f)",
            self.input_topic,
            self.output_topic,
            self.max_linear,
            self.max_angular,
            self.max_linear_accel,
            self.max_angular_accel,
        )

    def _on_twist(self, msg: Twist) -> None:
        self._target.linear.x = _clamp(float(msg.linear.x), -self.max_linear, self.max_linear)
        self._target.angular.z = _clamp(float(msg.angular.z), -self.max_angular, self.max_angular)
        self._last_msg_time = rospy.Time.now()

    def _zero_target_if_stale(self, now: rospy.Time) -> None:
        if self._last_msg_time == rospy.Time(0):
            return
        if (now - self._last_msg_time).to_sec() > self.timeout:
            self._target.linear.x = 0.0
            self._target.angular.z = 0.0

    def _tick(self) -> None:
        now = rospy.Time.now()
        dt = max((now - self._last_tick).to_sec(), 1.0 / max(self.rate_hz, 1.0))
        self._last_tick = now
        self._zero_target_if_stale(now)

        self._current.linear.x = _approach(
            self._current.linear.x,
            self._target.linear.x,
            self.max_linear_accel * dt,
        )
        self._current.angular.z = _approach(
            self._current.angular.z,
            self._target.angular.z,
            self.max_angular_accel * dt,
        )

        # Avoid tiny residual commands that keep Gazebo controllers twitching.
        if math.fabs(self._current.linear.x) < 1e-3:
            self._current.linear.x = 0.0
        if math.fabs(self._current.angular.z) < 1e-3:
            self._current.angular.z = 0.0

        out = Twist()
        out.linear.x = self._current.linear.x
        out.angular.z = self._current.angular.z
        self._pub.publish(out)

    def spin(self) -> None:
        rate = rospy.Rate(max(self.rate_hz, 1.0))
        while not rospy.is_shutdown():
            self._tick()
            rate.sleep()


def main() -> None:
    rospy.init_node("hsr_cmd_vel_smoother")
    HsrCmdVelSmoother().spin()


if __name__ == "__main__":
    main()
