#!/usr/bin/env python3
"""Publish a JSON semantic room context as a latched ROS topic."""

from __future__ import annotations

from pathlib import Path

import rospy
from std_msgs.msg import String


def main() -> None:
    rospy.init_node("publish_room_context", anonymous=False)
    context_path = rospy.get_param("~context_path", "")
    topic = rospy.get_param("~topic", "/vlmap/room_context")
    if not context_path:
        raise rospy.ROSInitException("~context_path is required")
    payload = Path(context_path).read_text(encoding="utf-8")
    pub = rospy.Publisher(topic, String, queue_size=1, latch=True)
    rospy.sleep(0.25)
    pub.publish(String(data=payload))
    rospy.loginfo("[publish_room_context] published %s to %s", context_path, topic)
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
