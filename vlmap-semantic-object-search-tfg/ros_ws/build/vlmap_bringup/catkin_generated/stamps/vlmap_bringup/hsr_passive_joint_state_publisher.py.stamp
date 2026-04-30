#!/usr/bin/env python3
"""Publish passive caster joint states for the official HSR in Gazebo."""

from __future__ import annotations

from typing import List

import rospy
from gazebo_msgs.srv import GetJointProperties, GetJointPropertiesRequest
from sensor_msgs.msg import JointState


class HsrPassiveJointStatePublisher:
    def __init__(self) -> None:
        self.topic = rospy.get_param("~topic", "/joint_states")
        self.rate_hz = float(rospy.get_param("~rate_hz", 15.0))
        self.joint_names: List[str] = list(
            rospy.get_param(
                "~joint_names",
                [
                    "base_r_passive_wheel_x_frame_joint",
                    "base_r_passive_wheel_y_frame_joint",
                    "base_r_passive_wheel_z_joint",
                    "base_l_passive_wheel_x_frame_joint",
                    "base_l_passive_wheel_y_frame_joint",
                    "base_l_passive_wheel_z_joint",
                ],
            )
        )
        rospy.wait_for_service("/gazebo/get_joint_properties")
        self._get_joint_properties = rospy.ServiceProxy("/gazebo/get_joint_properties", GetJointProperties)
        self._pub = rospy.Publisher(self.topic, JointState, queue_size=10)
        rospy.loginfo(
            "[hsr_passive_joint_state_publisher] topic=%s joints=%s",
            self.topic,
            ",".join(self.joint_names),
        )

    def _read_joint(self, joint_name: str):
        req = GetJointPropertiesRequest()
        req.joint_name = joint_name
        try:
            resp = self._get_joint_properties(req)
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(2.0, "[hsr_passive_joint_state_publisher] %s query failed: %s", joint_name, exc)
            return 0.0, 0.0
        position = float(resp.position[0]) if resp.position else 0.0
        velocity = float(resp.rate[0]) if resp.rate else 0.0
        return position, velocity

    def spin(self) -> None:
        rate = rospy.Rate(max(self.rate_hz, 1.0))
        while not rospy.is_shutdown():
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            for joint_name in self.joint_names:
                position, velocity = self._read_joint(joint_name)
                msg.name.append(joint_name)
                msg.position.append(position)
                msg.velocity.append(velocity)
                msg.effort.append(0.0)
            self._pub.publish(msg)
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break


def main() -> None:
    rospy.init_node("hsr_passive_joint_state_publisher")
    HsrPassiveJointStatePublisher().spin()


if __name__ == "__main__":
    main()
