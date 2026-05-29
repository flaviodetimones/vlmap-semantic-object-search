#!/usr/bin/env python3
"""Stable Gazebo-only base driver for the HSR migration demos.

The stock HSR Gazebo caster dynamics are unstable in this container when the
base receives angular velocity commands.  This node keeps the ROS contract
simple (`cmd_vel` in, `/odom` + TF out) while moving the Gazebo model
kinematically through `/gazebo/set_model_state`.
"""

from __future__ import annotations

import math
from typing import Optional

import rospy
import tf2_ros
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class HsrKinematicBaseController:
    def __init__(self) -> None:
        self.model_name = rospy.get_param("~model_name", "hsrb")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/hsrb/command_velocity")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_footprint")
        self.rate_hz = float(rospy.get_param("~rate_hz", 30.0))
        self.cmd_timeout = float(rospy.get_param("~cmd_timeout", 0.35))
        # z_height < 0 (default) => auto: let the model settle on the floor by
        # gravity for ``settle_time`` seconds, then capture its resting z and
        # hold it. A fixed >= 0 value overrides and pins the model at that z.
        self.z_height = float(rospy.get_param("~z_height", -1.0))
        self.settle_time = float(rospy.get_param("~settle_time", 1.5))
        self.max_linear = float(rospy.get_param("~max_linear", 0.30))
        self.max_angular = float(rospy.get_param("~max_angular", 0.70))
        self.publish_tf = bool(rospy.get_param("~publish_tf", True))

        self._last_cmd = Twist()
        self._last_cmd_time = rospy.Time(0)
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._last_step: Optional[rospy.Time] = None

        rospy.loginfo("[hsr_kinematic_base] waiting for Gazebo model state service")
        rospy.wait_for_service("/gazebo/get_model_state")
        self._get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self._load_initial_pose()

        self._model_state_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self._odom_pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=10)
        self._tf_pub = tf2_ros.TransformBroadcaster()
        self._cmd_sub = rospy.Subscriber(self.cmd_vel_topic, Twist, self._on_cmd, queue_size=1)

        rospy.loginfo(
            "[hsr_kinematic_base] model=%s cmd=%s odom=%s max=(%.2f m/s, %.2f rad/s)",
            self.model_name,
            self.cmd_vel_topic,
            self.odom_topic,
            self.max_linear,
            self.max_angular,
        )

    def _load_initial_pose(self) -> None:
        rate = rospy.Rate(5.0)
        state = None
        while not rospy.is_shutdown():
            state = self._get_model_state(self.model_name, "world")
            if state.success:
                break
            rospy.loginfo_throttle(
                2.0,
                "[hsr_kinematic_base] waiting for Gazebo model '%s': %s",
                self.model_name,
                state.status_message,
            )
            rate.sleep()
        if state is None or not state.success:
            raise rospy.ROSInterruptException("ROS shutdown before Gazebo model was available")

        # Auto z: let the model drop to its physical resting height before we
        # start pinning it. Without this settle window the controller would
        # freeze the robot at its (possibly floating) spawn height.
        if self.z_height < 0.0 and self.settle_time > 0.0:
            rospy.loginfo(
                "[hsr_kinematic_base] letting model settle %.1fs before capturing z",
                self.settle_time,
            )
            rospy.sleep(self.settle_time)
            settled = self._get_model_state(self.model_name, "world")
            if settled.success:
                state = settled

        self._x = state.pose.position.x
        self._y = state.pose.position.y
        if self.z_height < 0.0:
            self.z_height = float(state.pose.position.z)
            rospy.loginfo("[hsr_kinematic_base] captured resting z=%.3f", self.z_height)
        quat = state.pose.orientation
        _, _, self._yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

    def _on_cmd(self, msg: Twist) -> None:
        self._last_cmd = msg
        self._last_cmd_time = rospy.Time.now()

    def _active_cmd(self, now: rospy.Time) -> tuple[float, float]:
        if self._last_cmd_time == rospy.Time(0):
            return 0.0, 0.0
        if (now - self._last_cmd_time).to_sec() > self.cmd_timeout:
            return 0.0, 0.0
        linear = _clamp(float(self._last_cmd.linear.x), self.max_linear)
        angular = _clamp(float(self._last_cmd.angular.z), self.max_angular)
        return linear, angular

    def _publish_odom(self, stamp: rospy.Time, linear: float, angular: float) -> None:
        quat = quaternion_from_euler(0.0, 0.0, self._yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self._x
        odom.pose.pose.position.y = self._y
        odom.pose.pose.position.z = self.z_height
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]
        odom.twist.twist.linear.x = linear
        odom.twist.twist.angular.z = angular
        self._odom_pub.publish(odom)

        if not self.publish_tf:
            return
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self.odom_frame
        tf_msg.child_frame_id = self.base_frame
        tf_msg.transform.translation.x = self._x
        tf_msg.transform.translation.y = self._y
        tf_msg.transform.translation.z = self.z_height
        tf_msg.transform.rotation.x = quat[0]
        tf_msg.transform.rotation.y = quat[1]
        tf_msg.transform.rotation.z = quat[2]
        tf_msg.transform.rotation.w = quat[3]
        self._tf_pub.sendTransform(tf_msg)

    def _push_model_state(self, linear: float, angular: float) -> None:
        quat = quaternion_from_euler(0.0, 0.0, self._yaw)
        state = ModelState()
        state.model_name = self.model_name
        state.reference_frame = "world"
        state.pose.position.x = self._x
        state.pose.position.y = self._y
        state.pose.position.z = self.z_height
        state.pose.orientation.x = quat[0]
        state.pose.orientation.y = quat[1]
        state.pose.orientation.z = quat[2]
        state.pose.orientation.w = quat[3]
        state.twist.linear.x = linear
        state.twist.angular.z = angular
        self._model_state_pub.publish(state)

    def spin(self) -> None:
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if self._last_step is None:
                dt = 0.0
            else:
                dt = max(0.0, min(0.2, (now - self._last_step).to_sec()))
            self._last_step = now

            linear, angular = self._active_cmd(now)
            self._yaw = math.atan2(math.sin(self._yaw + angular * dt), math.cos(self._yaw + angular * dt))
            self._x += linear * math.cos(self._yaw) * dt
            self._y += linear * math.sin(self._yaw) * dt

            self._publish_odom(now, linear, angular)
            self._push_model_state(linear, angular)
            rate.sleep()


def main() -> None:
    rospy.init_node("hsr_kinematic_base_controller")
    try:
        HsrKinematicBaseController().spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
