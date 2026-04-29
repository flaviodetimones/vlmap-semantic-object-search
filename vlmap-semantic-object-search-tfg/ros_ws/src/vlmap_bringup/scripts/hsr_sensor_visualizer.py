#!/usr/bin/env python3
"""Publish RViz markers for HSR IMU and wrist force/torque sensors."""

from __future__ import annotations

import math

import rospy
from geometry_msgs.msg import Point, WrenchStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class HsrSensorVisualizer:
    def __init__(self) -> None:
        self.marker_topic = rospy.get_param("~marker_topic", "/hsrb/sensor_markers")
        self.imu_topic = rospy.get_param("~imu_topic", "/hsrb/base_imu/data")
        self.wrench_topic = rospy.get_param("~wrench_topic", "/hsrb/wrist_wrench")
        self.imu_scale = float(rospy.get_param("~imu_scale", 0.35))
        self.force_scale = float(rospy.get_param("~force_scale", 0.015))
        self.torque_scale = float(rospy.get_param("~torque_scale", 0.08))
        self._imu_msg = None
        self._wrench_msg = None

        self._pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1)
        rospy.Subscriber(self.imu_topic, Imu, self._on_imu, queue_size=1)
        rospy.Subscriber(self.wrench_topic, WrenchStamped, self._on_wrench, queue_size=1)

    def _on_imu(self, msg: Imu) -> None:
        self._imu_msg = msg

    def _on_wrench(self, msg: WrenchStamped) -> None:
        self._wrench_msg = msg

    @staticmethod
    def _arrow(marker_id: int, frame_id: str, ns: str, color: ColorRGBA, scale_xyz, p0, p1) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = scale_xyz[0]
        marker.scale.y = scale_xyz[1]
        marker.scale.z = scale_xyz[2]
        marker.color = color
        marker.points = [p0, p1]
        return marker

    def _build_imu_marker(self) -> Marker | None:
        msg = self._imu_msg
        if msg is None:
            return None

        q = msg.orientation
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if norm < 1e-6:
            return None
        x = q.x / norm
        y = q.y / norm
        z = q.z / norm
        w = q.w / norm

        # Rotate unit X axis by the IMU quaternion to show facing/orientation.
        vx = 1.0 - 2.0 * (y * y + z * z)
        vy = 2.0 * (x * y + w * z)
        vz = 2.0 * (x * z - w * y)

        p0 = Point(0.0, 0.0, 0.0)
        p1 = Point(vx * self.imu_scale, vy * self.imu_scale, vz * self.imu_scale)
        return self._arrow(
            1,
            msg.header.frame_id or "base_imu_frame",
            "imu",
            ColorRGBA(0.2, 0.8, 1.0, 0.95),
            (0.025, 0.05, 0.08),
            p0,
            p1,
        )

    def _build_wrench_markers(self) -> list[Marker]:
        msg = self._wrench_msg
        if msg is None:
            return []

        force = msg.wrench.force
        torque = msg.wrench.torque
        markers = []
        p0 = Point(0.0, 0.0, 0.0)
        p_force = Point(force.x * self.force_scale, force.y * self.force_scale, force.z * self.force_scale)
        p_torque = Point(torque.x * self.torque_scale, torque.y * self.torque_scale, torque.z * self.torque_scale)
        markers.append(
            self._arrow(
                2,
                msg.header.frame_id or "wrist_roll_link",
                "wrench_force",
                ColorRGBA(1.0, 0.2, 0.2, 0.95),
                (0.018, 0.035, 0.06),
                p0,
                p_force,
            )
        )
        markers.append(
            self._arrow(
                3,
                msg.header.frame_id or "wrist_roll_link",
                "wrench_torque",
                ColorRGBA(1.0, 0.85, 0.2, 0.95),
                (0.014, 0.03, 0.05),
                p0,
                p_torque,
            )
        )
        return markers

    def spin(self) -> None:
        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            marker_array = MarkerArray()
            imu_marker = self._build_imu_marker()
            if imu_marker is not None:
                marker_array.markers.append(imu_marker)
            marker_array.markers.extend(self._build_wrench_markers())
            self._pub.publish(marker_array)
            rate.sleep()


def main() -> None:
    rospy.init_node("hsr_sensor_visualizer")
    HsrSensorVisualizer().spin()


if __name__ == "__main__":
    main()
