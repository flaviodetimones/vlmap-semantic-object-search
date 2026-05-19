#!/usr/bin/env python3
"""One-shot smoke test for the ROS semantic/navigation bridge."""

from __future__ import annotations

import json
import uuid
from threading import Event

import rospy
from std_msgs.msg import String

from vlmap_msgs.srv import QueryRoom, ResolveRoom, VerifyObject


def main() -> None:
    rospy.init_node("ros_contract_smoke_test", anonymous=False)

    target = rospy.get_param("~target", "laptop")
    explicit_room = rospy.get_param("~room", "")
    instruction = rospy.get_param(
        "~instruction",
        f"find the {target} in the {explicit_room}" if explicit_room else f"find {target}",
    )
    timeout_s = float(rospy.get_param("~timeout_s", 8.0))

    rospy.loginfo("[ros_contract_smoke_test] waiting for services")
    rospy.wait_for_service("/vlmap/query_room", timeout=timeout_s)
    rospy.wait_for_service("/vlmap/resolve_room", timeout=timeout_s)
    rospy.wait_for_service("/vlmap/verify_object", timeout=timeout_s)

    query_room = rospy.ServiceProxy("/vlmap/query_room", QueryRoom)
    resolve_room = rospy.ServiceProxy("/vlmap/resolve_room", ResolveRoom)
    verify_object = rospy.ServiceProxy("/vlmap/verify_object", VerifyObject)
    instruction_pub = rospy.Publisher("/vlmap/instruction", String, queue_size=1)

    ranking = query_room(target)
    if not ranking.rooms:
        raise RuntimeError(f"no ranked rooms returned for target {target!r}")

    selected_room = explicit_room or ranking.rooms[0]
    selected_score = float(ranking.scores[0]) if ranking.scores else None
    room_resp = resolve_room(selected_room)
    if not room_resp.found:
        raise RuntimeError(f"room {selected_room!r} could not be resolved")

    verify_resp = verify_object(
        object_class=target,
        request_id=uuid.uuid4().hex,
        rgb_topic="",
        depth_topic="",
        camera_info_topic="",
        point_cloud_topic="",
        metadata=json.dumps({"source": "ros_contract_smoke_test"}, sort_keys=True),
        timeout_s=timeout_s,
    )

    rospy.loginfo(
        "[ros_contract_smoke_test] verify_object target=%s found=%s message=%s",
        target,
        verify_resp.found,
        verify_resp.message,
    )

    navigation_payload = {"value": None}
    navigation_event = Event()

    def _on_navigation_result(msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if payload.get("object_class") != target:
            return
        navigation_payload["value"] = payload
        navigation_event.set()

    rospy.Subscriber("/vlmap/navigation_result", String, _on_navigation_result, queue_size=10)
    rospy.sleep(0.1)
    instruction_pub.publish(String(data=instruction))

    if not navigation_event.wait(timeout_s):
        raise RuntimeError("timed out waiting for /vlmap/navigation_result")

    payload = navigation_payload["value"]
    if payload is None:
        raise RuntimeError("received navigation event without payload")

    print("")
    print("ROS contract smoke test")
    print(f"  target:         {target}")
    print(f"  instruction:    {instruction}")
    print(f"  ranked room[0]: {ranking.rooms[0]}")
    if selected_score is not None:
        print(f"  ranked score:   {selected_score:.3f}")
    print(f"  resolved room:  {room_resp.room_id} @ ({room_resp.x:.2f}, {room_resp.y:.2f})")
    print(f"  verify found:   {verify_resp.found}")
    print(f"  nav success:    {payload.get('success')}")
    print(f"  nav room:       {payload.get('actual_room')}")
    print(f"  nav message:    {payload.get('message')}")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
