#!/usr/bin/env python3
"""Republish Gazebo ground-truth odometry as the project's /odom contract."""

from __future__ import annotations

import rospy
import tf
from nav_msgs.msg import Odometry


class HsrGroundTruthOdomBridge:
    def __init__(self) -> None:
        self.input_topic = rospy.get_param("~input_topic", "/hsrb/odom_ground_truth")
        self.output_topic = rospy.get_param("~output_topic", "/odom")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_footprint")
        self.publish_tf = bool(rospy.get_param("~publish_tf", True))

        self._odom_pub = rospy.Publisher(self.output_topic, Odometry, queue_size=10)
        self._tf_broadcaster = tf.TransformBroadcaster()
        rospy.Subscriber(self.input_topic, Odometry, self._on_odom, queue_size=10)

        rospy.loginfo(
            "[hsr_ground_truth_odom_bridge] %s -> %s (tf=%s, frames=%s->%s)",
            self.input_topic,
            self.output_topic,
            self.publish_tf,
            self.odom_frame,
            self.base_frame,
        )

    def _on_odom(self, msg: Odometry) -> None:
        bridged = Odometry()
        bridged.header = msg.header
        bridged.header.frame_id = self.odom_frame
        bridged.child_frame_id = self.base_frame
        bridged.pose = msg.pose
        bridged.twist = msg.twist
        self._odom_pub.publish(bridged)

        if self.publish_tf:
            pose = bridged.pose.pose
            self._tf_broadcaster.sendTransform(
                (pose.position.x, pose.position.y, pose.position.z),
                (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
                bridged.header.stamp if bridged.header.stamp != rospy.Time() else rospy.Time.now(),
                self.base_frame,
                self.odom_frame,
            )


def main() -> None:
    rospy.init_node("hsr_ground_truth_odom_bridge")
    HsrGroundTruthOdomBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
