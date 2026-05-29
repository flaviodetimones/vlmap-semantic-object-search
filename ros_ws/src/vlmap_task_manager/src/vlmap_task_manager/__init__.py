"""vlmap_task_manager — turn NL instructions into ROS goals.

Sprint 1: stub. Sprint 2 will wrap:
  - third_party/vlmaps/vlmaps/policy/strategic_policy.py  (room ranking + LLM)
  - third_party/vlmaps/vlmaps/policy/executor.py          (action vocabulary)
  - third_party/vlmaps/application/interactive_object_nav.compute_heatmap
And publish move_base_msgs/MoveBaseGoal once the strategic layer chooses one.
"""
from .parsing import ParsedInstruction, parse_instruction
from .result_bridge import (
    ActiveSemanticGoal,
    active_goal_from_fields,
    build_navigation_result_payload,
    decode_semantic_goal_metadata,
    move_base_status_to_success,
)

__all__ = [
    "ParsedInstruction",
    "parse_instruction",
    "ActiveSemanticGoal",
    "active_goal_from_fields",
    "build_navigation_result_payload",
    "decode_semantic_goal_metadata",
    "move_base_status_to_success",
]
