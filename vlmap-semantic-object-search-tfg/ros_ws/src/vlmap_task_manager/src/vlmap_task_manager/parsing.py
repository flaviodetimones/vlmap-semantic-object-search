"""Instruction parsing helpers for the ROS task manager.

The parser is deliberately lightweight: it should work inside the ROS container
without depending on the full Habitat/VLMaps runtime. It extracts:

- the target object category
- an optional explicit room hint from phrases such as "in the kitchen"
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional


@dataclass(frozen=True)
class ParsedInstruction:
    target: str
    explicit_room: Optional[str] = None
    raw_text: str = ""


_LEADING_PATTERNS = [
    r"^\s*find\s+",
    r"^\s*search\s+for\s+",
    r"^\s*look\s+for\s+",
    r"^\s*go\s+to\s+",
    r"^\s*go\s+find\s+",
    r"^\s*please\s+find\s+",
]


def parse_instruction(text: str) -> ParsedInstruction:
    raw = str(text or "").strip()
    normal = re.sub(r"\s+", " ", raw.lower()).strip()
    for pattern in _LEADING_PATTERNS:
        normal = re.sub(pattern, "", normal)

    explicit_room = None
    room_match = re.search(r"\bin\s+(?:the\s+)?([a-z][a-z0-9 _-]*)$", normal)
    if room_match:
        explicit_room = room_match.group(1).strip()
        target = normal[: room_match.start()].strip()
    else:
        target = normal

    target = re.sub(r"^(a|an|the)\s+", "", target).strip()
    return ParsedInstruction(
        target=target or normal,
        explicit_room=explicit_room,
        raw_text=raw,
    )
