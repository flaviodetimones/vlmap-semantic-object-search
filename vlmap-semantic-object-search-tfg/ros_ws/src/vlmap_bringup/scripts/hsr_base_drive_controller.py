#!/usr/bin/env python3
"""Drive the official HSR base in Gazebo using wheel effort control."""

from __future__ import annotations

from typing import Dict

import rospy
from gazebo_msgs.srv import ApplyJointEffort, ApplyJointEffortRequest
from gazebo_msgs.srv import JointRequest, JointRequestRequest
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState


def _param(name: str, default, namespace: str):
    private_name = f"~{name}"
    if rospy.has_param(private_name):
        return rospy.get_param(private_name)
    scoped_name = f"{namespace.rstrip('/')}/{name}"
    if namespace and rospy.has_param(scoped_name):
        return rospy.get_param(scoped_name)
    return default


class HsrBaseDriveController:
    def __init__(self) -> None:
        self.config_namespace = rospy.get_param("~config_namespace", "/hsrb/base_drive")
        self.cmd_vel_topic = _param("cmd_vel_topic", "/hsrb/command_velocity", self.config_namespace)
        self.joint_states_topic = _param("joint_states_topic", "/joint_states", self.config_namespace)
        self.left_joint = _param("left_wheel_joint", "base_l_drive_wheel_joint", self.config_namespace)
        self.right_joint = _param("right_wheel_joint", "base_r_drive_wheel_joint", self.config_namespace)
        self.wheel_radius = float(_param("wheel_radius", 0.040, self.config_namespace))
        self.wheel_separation = float(_param("wheel_separation", 0.266, self.config_namespace))
        self.max_effort = float(_param("max_effort", 11.067, self.config_namespace))
        self.kp = float(_param("kp", 7.5, self.config_namespace))
        self.kff = float(_param("kff", 0.7, self.config_namespace))
        self.control_rate_hz = float(_param("control_rate_hz", 20.0, self.config_namespace))
        self.cmd_timeout = float(_param("cmd_timeout", 0.40, self.config_namespace))

        self._joint_velocity: Dict[str, float] = {}
        self._last_cmd = Twist()
        self._last_cmd_stamp = rospy.Time(0)

        rospy.Subscriber(self.joint_states_topic, JointState, self._on_joint_states, queue_size=1)
        rospy.Subscriber(self.cmd_vel_topic, Twist, self._on_cmd_vel, queue_size=1)

        rospy.wait_for_service("/gazebo/apply_joint_effort")
        rospy.wait_for_service("/gazebo/clear_joint_forces")
        self._apply_effort = rospy.ServiceProxy("/gazebo/apply_joint_effort", ApplyJointEffort)
        self._clear_effort = rospy.ServiceProxy("/gazebo/clear_joint_forces", JointRequest)

        rospy.on_shutdown(self._stop_wheels)

        rospy.loginfo(
            "[hsr_base_drive_controller] cmd=%s joints=(%s,%s) r=%.3f sep=%.3f kp=%.2f kff=%.2f",
            self.cmd_vel_topic,
            self.left_joint,
            self.right_joint,
            self.wheel_radius,
            self.wheel_separation,
            self.kp,
            self.kff,
        )

    def _on_joint_states(self, msg: JointState) -> None:
        for idx, name in enumerate(msg.name):
            if idx < len(msg.velocity):
                self._joint_velocity[name] = float(msg.velocity[idx])

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._last_cmd = msg
        self._last_cmd_stamp = rospy.Time.now()

    def _target_wheel_velocity(self) -> Dict[str, float]:
        now = rospy.Time.now()
        if self._last_cmd_stamp == rospy.Time(0) or (now - self._last_cmd_stamp).to_sec() > self.cmd_timeout:
            linear_x = 0.0
            angular_z = 0.0
        else:
            linear_x = float(self._last_cmd.linear.x)
            angular_z = float(self._last_cmd.angular.z)

        half_sep = 0.5 * self.wheel_separation
        left_linear = linear_x - angular_z * half_sep
        right_linear = linear_x + angular_z * half_sep
        return {
            self.left_joint: left_linear / self.wheel_radius,
            self.right_joint: right_linear / self.wheel_radius,
        }

    def _apply_joint_effort(self, joint_name: str, target_velocity: float, duration: rospy.Duration) -> None:
        current_velocity = float(self._joint_velocity.get(joint_name, 0.0))
        error = target_velocity - current_velocity
        effort = self.kp * error + self.kff * target_velocity
        effort = max(-self.max_effort, min(self.max_effort, effort))

        clear_req = JointRequestRequest()
        clear_req.joint_name = joint_name
        try:
            self._clear_effort(clear_req)
        except rospy.ServiceException:
            pass

        req = ApplyJointEffortRequest()
        req.joint_name = joint_name
        req.effort = effort
        req.start_time = rospy.Time(0)
        req.duration = duration
        try:
            self._apply_effort(req)
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(2.0, "[hsr_base_drive_controller] effort apply failed for %s: %s", joint_name, exc)

    def _stop_wheels(self) -> None:
        for joint_name in (self.left_joint, self.right_joint):
            clear_req = JointRequestRequest()
            clear_req.joint_name = joint_name
            try:
                self._clear_effort(clear_req)
            except rospy.ServiceException:
                pass

    def spin(self) -> None:
        rate_hz = max(self.control_rate_hz, 2.0)
        rate = rospy.Rate(rate_hz)
        duration = rospy.Duration(1.5 / rate_hz)
        while not rospy.is_shutdown():
            targets = self._target_wheel_velocity()
            for joint_name, target_velocity in targets.items():
                self._apply_joint_effort(joint_name, target_velocity, duration)
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break


def main() -> None:
    rospy.init_node("hsr_base_drive_controller")
    HsrBaseDriveController().spin()


if __name__ == "__main__":
    main()
