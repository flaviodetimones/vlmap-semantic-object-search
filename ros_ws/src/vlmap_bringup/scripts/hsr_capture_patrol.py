#!/usr/bin/env python3
"""Drive the HSR through capture waypoints and rotate in place at each stop."""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, Iterable, List, Tuple

import actionlib
import rospy
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_srvs.srv import Trigger


Waypoint = Tuple[float, float, float]


def _yaw_to_quat(yaw: float) -> Tuple[float, float]:
    half = 0.5 * yaw
    return math.sin(half), math.cos(half)


def _parse_waypoint(raw: Any) -> Waypoint:
    if isinstance(raw, dict):
        return (
            float(raw.get("x", 0.0)),
            float(raw.get("y", 0.0)),
            float(raw.get("yaw", raw.get("theta", 0.0))),
        )
    if isinstance(raw, str):
        parts = [float(part.strip()) for part in raw.split(",")]
        if len(parts) not in (2, 3):
            raise ValueError(f"waypoint string must be x,y[,yaw], got {raw!r}")
        return (parts[0], parts[1], parts[2] if len(parts) == 3 else 0.0)
    if isinstance(raw, (list, tuple)):
        if len(raw) not in (2, 3):
            raise ValueError(f"waypoint list must have 2 or 3 values, got {raw!r}")
        return (float(raw[0]), float(raw[1]), float(raw[2]) if len(raw) == 3 else 0.0)
    raise ValueError(f"unsupported waypoint format: {raw!r}")


def _load_waypoints() -> List[Waypoint]:
    explicit = rospy.get_param("~waypoints", [])
    if explicit:
        return [_parse_waypoint(item) for item in explicit]

    world = rospy.get_param("~world", "small_house")
    routes = rospy.get_param("~routes", {})
    route = routes.get(world, [])
    if not route:
        available = ", ".join(sorted(routes.keys())) or "<none>"
        raise RuntimeError(f"no capture route for world={world!r}; available routes: {available}")
    return [_parse_waypoint(item) for item in route]


class HsrCapturePatrol:
    def __init__(self) -> None:
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.move_base_action = rospy.get_param("~move_base_action", "/move_base")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/hsrb/command_velocity_raw")
        self.goal_timeout_sec = float(rospy.get_param("~goal_timeout_sec", 90.0))
        self.server_timeout_sec = float(rospy.get_param("~server_timeout_sec", 20.0))
        self.settle_sec = float(rospy.get_param("~settle_sec", 1.0))
        self.scan_each_waypoint = bool(rospy.get_param("~scan_each_waypoint", True))
        self.scan_turns = float(rospy.get_param("~scan_turns", 1.0))
        self.scan_angular_speed = float(rospy.get_param("~scan_angular_speed", 0.30))
        self.scan_rate_hz = float(rospy.get_param("~scan_rate_hz", 10.0))
        self.continue_on_failure = bool(rospy.get_param("~continue_on_failure", True))
        self.stop_capture_on_finish = bool(rospy.get_param("~stop_capture_on_finish", False))
        self.stop_capture_service = rospy.get_param("~stop_capture_service", "/vlmap_capture/stop")
        self.waypoints = _load_waypoints()

        self.client = actionlib.SimpleActionClient(self.move_base_action, MoveBaseAction)
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)

    def _make_goal(self, waypoint: Waypoint) -> MoveBaseGoal:
        x, y, yaw = waypoint
        z, w = _yaw_to_quat(yaw)
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.frame_id
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.z = z
        goal.target_pose.pose.orientation.w = w
        return goal

    def _publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())

    def _scan_360(self, index: int) -> None:
        if not self.scan_each_waypoint:
            return
        speed = abs(self.scan_angular_speed)
        if speed < 1e-3 or self.scan_turns <= 0.0:
            return

        duration = (2.0 * math.pi * self.scan_turns) / speed
        twist = Twist()
        twist.angular.z = speed
        rate = rospy.Rate(max(self.scan_rate_hz, 1.0))
        end_time = rospy.Time.now() + rospy.Duration(duration)
        rospy.loginfo(
            "[hsr_capture_patrol] waypoint %d: scanning %.1f turn(s) at %.2f rad/s",
            index,
            self.scan_turns,
            speed,
        )
        while not rospy.is_shutdown() and rospy.Time.now() < end_time:
            self.cmd_pub.publish(twist)
            rate.sleep()
        self._publish_stop()
        rospy.sleep(0.5)

    def _stop_capture(self) -> None:
        if not self.stop_capture_on_finish:
            return
        try:
            rospy.wait_for_service(self.stop_capture_service, timeout=5.0)
            response = rospy.ServiceProxy(self.stop_capture_service, Trigger)()
            rospy.loginfo("[hsr_capture_patrol] capture stop: %s", response.message)
        except Exception as exc:  # pragma: no cover - exercised in ROS runtime.
            rospy.logwarn("[hsr_capture_patrol] capture stop failed: %s", exc)

    def run(self) -> int:
        rospy.loginfo(
            "[hsr_capture_patrol] waiting for %s; waypoints=%d cmd_vel=%s",
            self.move_base_action,
            len(self.waypoints),
            self.cmd_vel_topic,
        )
        if not self.client.wait_for_server(rospy.Duration(self.server_timeout_sec)):
            rospy.logerr("[hsr_capture_patrol] move_base action server not available")
            return 2

        failures = 0
        for index, waypoint in enumerate(self.waypoints, start=1):
            if rospy.is_shutdown():
                break
            x, y, yaw = waypoint
            rospy.loginfo(
                "[hsr_capture_patrol] waypoint %d/%d -> x=%.2f y=%.2f yaw=%.2f",
                index,
                len(self.waypoints),
                x,
                y,
                yaw,
            )
            self.client.send_goal(self._make_goal(waypoint))
            finished = self.client.wait_for_result(rospy.Duration(self.goal_timeout_sec))
            if not finished:
                self.client.cancel_goal()
                failures += 1
                rospy.logwarn("[hsr_capture_patrol] waypoint %d timed out", index)
                if not self.continue_on_failure:
                    break
                continue

            state = self.client.get_state()
            if state != GoalStatus.SUCCEEDED:
                failures += 1
                rospy.logwarn("[hsr_capture_patrol] waypoint %d failed with state=%d", index, state)
                if not self.continue_on_failure:
                    break
                continue

            rospy.sleep(max(self.settle_sec, 0.0))
            self._scan_360(index)

        self._publish_stop()
        self._stop_capture()
        rospy.loginfo("[hsr_capture_patrol] finished; failures=%d", failures)
        return 1 if failures else 0


def main() -> None:
    rospy.init_node("hsr_capture_patrol")
    try:
        code = HsrCapturePatrol().run()
    except Exception as exc:
        rospy.logerr("[hsr_capture_patrol] fatal: %s", exc)
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
