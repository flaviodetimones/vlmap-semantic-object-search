"""Offline tests for the dynamic semantic room context JSON contract."""

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros_ws" / "src" / "vlmap_semantic_server" / "src"))

from vlmap_semantic_server import load_dynamic_room_context


def test_dynamic_context_supports_explicit_xy_and_aliases():
    ctx = load_dynamic_room_context(
        """
        {
          "rooms": [
            {"room_id": "kitchen", "x": 1.0, "y": 2.0, "aliases": ["cocina"]},
            {"room_id": "office", "x": -3.0, "y": 4.5}
          ],
          "priors": {
            "bottle": [{"room_id": "kitchen", "score": 0.9}, {"room_id": "office", "score": 0.1}]
          }
        }
        """
    )

    assert ctx.is_available() is True
    assert ctx.resolve_room_name("cocina") == "kitchen"
    assert ctx.resolve_room_pose("office") == (-3.0, 4.5)
    rooms, scores = ctx.rank_rooms_for_category("bottle")
    assert rooms == ["kitchen", "office"]
    assert scores == pytest.approx([0.9, 0.1])


def test_dynamic_context_supports_row_col_conversion_and_default_priors():
    ctx = load_dynamic_room_context(
        """
        {
          "rooms": [
            {"room_id": "bedroom", "row": 1, "col": 3}
          ]
        }
        """,
        resolution=0.5,
        origin_x=-5.0,
        origin_y=-5.0,
    )
    assert ctx.resolve_room_pose("bedroom") == pytest.approx((-3.25, -4.25))
    rooms, scores = ctx.rank_rooms_for_category("book")
    assert rooms == ["bedroom"]
    assert scores == [1.0]


def test_dynamic_context_rejects_missing_pose():
    with pytest.raises(ValueError):
        load_dynamic_room_context('{"rooms":[{"room_id":"kitchen"}]}')
