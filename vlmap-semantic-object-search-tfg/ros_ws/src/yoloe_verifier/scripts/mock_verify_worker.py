#!/usr/bin/env python3
"""Simple mock worker that answers bridged verification requests."""

from __future__ import annotations

import json

import rospy
from std_msgs.msg import String

from yoloe_verifier import make_verification_response


def main() -> None:
    rospy.init_node("mock_verify_worker", anonymous=False)
    request_topic = rospy.get_param("~request_topic", "/vlmap/verification/request")
    result_topic = rospy.get_param("~result_topic", "/vlmap/verification/result")
    known = {
        token.strip().lower()
        for token in str(rospy.get_param("~known_objects", "bottle,mug,laptop,book")).split(",")
        if token.strip()
    }
    pub = rospy.Publisher(result_topic, String, queue_size=10)

    def _on_request(msg: String) -> None:
        payload = json.loads(msg.data)
        object_class = str(payload.get("object_class") or "").strip().lower()
        request_id = str(payload.get("request_id") or "").strip()
        found = object_class in known
        response = make_verification_response(
            request_id=request_id,
            success=True,
            found=found,
            message="mock verification complete",
            metadata={"worker": "mock_verify_worker", "known_object": found},
        )
        pub.publish(String(data=json.dumps(response, sort_keys=True)))

    rospy.Subscriber(request_topic, String, _on_request, queue_size=10)
    rospy.loginfo("[mock_verify_worker] ready request=%s result=%s", request_topic, result_topic)
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
