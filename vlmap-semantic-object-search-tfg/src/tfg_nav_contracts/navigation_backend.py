"""Backend abstraction for navigation execution.

The contract stays intentionally small: ``submit_goal`` to fire-and-forget,
``wait_result`` to block on completion, and ``cancel`` for preemption. That
keeps the strategic policy independent of whether execution happens inside
Habitat-Sim or through a ROS navigation stack.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import time
from typing import Any, Callable, Dict, Optional
import uuid

from .semantic_goal import SemanticGoal


@dataclass
class NavigationResult:
    """Outcome of executing a SemanticGoal."""

    success: bool
    actual_room: Optional[str] = None
    found: bool = False
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class NavigationBackend(ABC):
    """Common interface for Habitat and ROS execution backends."""

    @abstractmethod
    def submit_goal(self, goal: SemanticGoal) -> str:
        """Send the goal to the backend. Returns an opaque token used by
        :meth:`wait_result` and :meth:`cancel`.
        """

    @abstractmethod
    def wait_result(self, token: str, timeout_s: Optional[float] = None) -> NavigationResult:
        """Block until the backend reports a result, or until *timeout_s*."""

    @abstractmethod
    def cancel(self, token: str) -> None:
        """Cancel an in-flight goal."""


class HabitatFollowerBackend(NavigationBackend):
    """Wraps the existing Habitat path follower.

    Sprint 2 will plug in:
      - ``application.interactive_object_nav.execute_nav_replay``
      - ``application.interactive_object_nav.face_toward_pos``
      - ``application.interactive_object_nav.scan_local_and_verify``
    """

    def submit_goal(self, goal: SemanticGoal) -> str:
        raise NotImplementedError(
            "HabitatFollowerBackend.submit_goal is a Sprint 0 stub. "
            "TODO Sprint 2: wire to interactive_object_nav.execute_nav_replay."
        )

    def wait_result(self, token: str, timeout_s: Optional[float] = None) -> NavigationResult:
        raise NotImplementedError(
            "HabitatFollowerBackend.wait_result is a Sprint 0 stub. "
            "TODO Sprint 2: synchronise with the Habitat sim main loop."
        )

    def cancel(self, token: str) -> None:
        raise NotImplementedError(
            "HabitatFollowerBackend.cancel is a Sprint 0 stub. "
            "TODO Sprint 2: implement preemption (Habitat is currently single-threaded)."
        )


class RosNavigationBackend(NavigationBackend):
    """Pushes goals to the ROS navigation stack through rosbridge.

    Design constraints:
      - no hard dependency on ``rospy`` or a ROS runtime in the Python process
      - same contract can be used from ``tfg-sim`` or a standalone harness
      - sequential goals by default, matching the current strategic policy

    The backend publishes two topics:
      - a semantic goal mirror on ``/vlmap/semantic_goal``
      - a plain 2D goal on ``/move_base_simple/goal``

    Results can arrive either as:
      - a JSON string on a custom topic (recommended while the stack is still
        being wired), or
      - ``move_base_msgs/MoveBaseActionResult`` on ``/move_base/result``
    """

    def __init__(
        self,
        *,
        host: str = "tfg-ros",
        port: int = 9090,
        semantic_goal_topic: str = "/vlmap/semantic_goal",
        semantic_goal_topic_type: str = "vlmap_msgs/SemanticGoal",
        nav_goal_topic: str = "/move_base_simple/goal",
        nav_goal_topic_type: str = "geometry_msgs/PoseStamped",
        cancel_topic: str = "/move_base/cancel",
        cancel_topic_type: str = "actionlib_msgs/GoalID",
        result_topic: str = "/vlmap/navigation_result",
        result_topic_type: str = "std_msgs/String",
        goal_frame: str = "map",
        poll_interval_s: float = 0.05,
        publish_settle_s: float = 0.1,
        auto_connect: bool = True,
        client: Any = None,
        topic_factory: Optional[Callable[[Any, str, str], Any]] = None,
        message_factory: Optional[Callable[[Dict[str, Any]], Any]] = None,
        uuid_factory: Optional[Callable[[], str]] = None,
        clock: Optional[Callable[[], float]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.semantic_goal_topic = semantic_goal_topic
        self.semantic_goal_topic_type = semantic_goal_topic_type
        self.nav_goal_topic = nav_goal_topic
        self.nav_goal_topic_type = nav_goal_topic_type
        self.cancel_topic = cancel_topic
        self.cancel_topic_type = cancel_topic_type
        self.result_topic = result_topic
        self.result_topic_type = result_topic_type
        self.goal_frame = goal_frame
        self.poll_interval_s = float(poll_interval_s)
        self.publish_settle_s = float(publish_settle_s)
        self._clock = clock or time.monotonic
        self._sleep = sleep_fn or time.sleep
        self._uuid_factory = uuid_factory or (lambda: uuid.uuid4().hex)
        self._results: Dict[str, NavigationResult] = {}
        self._pending_goals: Dict[str, SemanticGoal] = {}
        self._pending_order: list[str] = []

        if client is None or topic_factory is None or message_factory is None:
            roslibpy = self._import_roslibpy()
            self._client = client or roslibpy.Ros(host=self.host, port=self.port)
            self._topic_factory = topic_factory or (
                lambda ros_client, name, type_name: roslibpy.Topic(ros_client, name, type_name)
            )
            self._message_factory = message_factory or roslibpy.Message
        else:
            self._client = client
            self._topic_factory = topic_factory
            self._message_factory = message_factory

        self._semantic_pub = None
        self._nav_pub = None
        self._cancel_pub = None
        self._result_sub = None
        if auto_connect:
            self._ensure_connected()
        self._bind_topics()

    @staticmethod
    def _import_roslibpy():
        try:
            import roslibpy  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "RosNavigationBackend requires roslibpy when no fake client is injected."
            ) from exc
        return roslibpy

    def _ensure_connected(self) -> None:
        if not getattr(self._client, "is_connected", False):
            self._client.run()

    def _bind_topics(self) -> None:
        if self._semantic_pub is not None:
            return
        self._semantic_pub = self._topic_factory(
            self._client, self.semantic_goal_topic, self.semantic_goal_topic_type
        )
        self._nav_pub = self._topic_factory(self._client, self.nav_goal_topic, self.nav_goal_topic_type)
        self._cancel_pub = self._topic_factory(self._client, self.cancel_topic, self.cancel_topic_type)
        self._result_sub = self._topic_factory(self._client, self.result_topic, self.result_topic_type)
        self._result_sub.subscribe(self._handle_result_message)

    def _publish(self, topic: Any, payload: Dict[str, Any]) -> None:
        topic.publish(self._message_factory(payload))

    def _make_pose_stamped(self, x: float, y: float) -> Dict[str, Any]:
        return {
            "header": {"frame_id": self.goal_frame},
            "pose": {
                "position": {"x": float(x), "y": float(y), "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        }

    def _semantic_goal_payload(self, goal: SemanticGoal, token: str) -> Dict[str, Any]:
        metadata = dict(goal.metadata or {})
        metadata.setdefault("navigation_token", token)
        metadata.setdefault("backend", "ros")
        metadata.setdefault("goal_type", goal.type.value)
        x, y = goal.map_pose
        return {
            "type": goal.type.value,
            "map_pose": self._make_pose_stamped(x, y),
            "room_id": goal.room_id or "",
            "object_class": goal.object_class or "",
            "metadata": json.dumps(metadata, sort_keys=True),
        }

    def _nav_goal_payload(self, goal: SemanticGoal) -> Dict[str, Any]:
        x, y = goal.map_pose
        return self._make_pose_stamped(x, y)

    def submit_goal(self, goal: SemanticGoal) -> str:
        self._ensure_connected()
        self._bind_topics()
        token = self._uuid_factory()
        self._pending_goals[token] = goal
        self._pending_order.append(token)
        self._publish(self._semantic_pub, self._semantic_goal_payload(goal, token))
        self._publish(self._nav_pub, self._nav_goal_payload(goal))
        # roslibpy writes over a websocket; a tiny settle interval makes short-lived
        # helper scripts much less likely to exit before the frames are flushed.
        if self.publish_settle_s > 0.0:
            self._sleep(self.publish_settle_s)
        return token

    def _pop_oldest_pending_token(self) -> Optional[str]:
        while self._pending_order:
            token = self._pending_order.pop(0)
            if token in self._pending_goals:
                return token
        return None

    def _drop_pending_token(self, token: str) -> None:
        self._pending_goals.pop(token, None)
        try:
            self._pending_order.remove(token)
        except ValueError:
            pass

    def _handle_result_message(self, message: Dict[str, Any]) -> None:
        parsed = self._parse_result_message(message)
        if not parsed:
            return
        token, result = parsed
        self._results[token] = result
        self._drop_pending_token(token)

    def _parse_result_message(self, message: Dict[str, Any]) -> Optional[tuple[str, NavigationResult]]:
        if not isinstance(message, dict):
            return None

        # Preferred during migration: std_msgs/String carrying a JSON payload.
        data = message.get("data")
        if isinstance(data, str):
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                token = str(payload.get("token") or payload.get("navigation_token") or "").strip()
                if token:
                    return token, NavigationResult(
                        success=bool(payload.get("success", False)),
                        actual_room=payload.get("actual_room"),
                        found=bool(payload.get("found", payload.get("success", False))),
                        message=str(payload.get("message", "")),
                        metadata=dict(payload.get("metadata") or {}),
                    )

        # Compatibility path for move_base action results. The backend assumes
        # sequential goals, so an empty action goal id maps to the oldest pending
        # token.
        status = message.get("status")
        if isinstance(status, dict):
            status_code = int(status.get("status", -1))
            text = str(status.get("text", ""))
            goal_id = status.get("goal_id") or {}
            token = str(goal_id.get("id", "")).strip() or self._pop_oldest_pending_token()
            if token:
                success = status_code == 3
                metadata = {
                    "source": self.result_topic,
                    "status_code": status_code,
                }
                return token, NavigationResult(
                    success=success,
                    actual_room=None,
                    found=success,
                    message=text,
                    metadata=metadata,
                )
        return None

    def wait_result(self, token: str, timeout_s: Optional[float] = None) -> NavigationResult:
        self._ensure_connected()
        deadline = None if timeout_s is None else self._clock() + float(timeout_s)
        while token not in self._results:
            if deadline is not None and self._clock() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for ROS navigation result for token {token!r}"
                )
            self._sleep(self.poll_interval_s)
        return self._results.pop(token)

    def cancel(self, token: str) -> None:
        self._ensure_connected()
        self._bind_topics()
        self._publish(
            self._cancel_pub,
            {"stamp": {"secs": 0, "nsecs": 0}, "id": token},
        )
        self._drop_pending_token(token)
        self._results[token] = NavigationResult(
            success=False,
            actual_room=None,
            found=False,
            message="Cancelled before completion.",
            metadata={"source": "client_cancel"},
        )
