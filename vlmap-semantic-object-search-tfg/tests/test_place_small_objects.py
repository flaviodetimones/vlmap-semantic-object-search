"""Smoke tests for tools/place_small_objects.py pure helpers.

These tests cover the offline parts of the script (no Habitat, no real
HSSD assets needed): point-in-polygon, furniture alias normalisation,
template selection determinism and the selection of furniture
candidates from a synthetic scene.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "tools" / "place_small_objects.py"
    spec = importlib.util.spec_from_file_location("place_small_objects", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["place_small_objects"] = mod
    spec.loader.exec_module(mod)
    return mod


pso = _load_module()


# ---------------------------------------------------------------------------
# Point-in-polygon basics.
# ---------------------------------------------------------------------------


def test_point_in_polygon_inside_square():
    poly = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    assert pso._point_in_polygon(2.0, 2.0, poly) is True


def test_point_in_polygon_outside_square():
    poly = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    assert pso._point_in_polygon(5.0, 2.0, poly) is False


def test_point_in_polygon_degenerate_returns_false():
    assert pso._point_in_polygon(0.0, 0.0, [(0.0, 0.0)]) is False


# ---------------------------------------------------------------------------
# Furniture alias normalisation (sofa↔couch, shelf↔shelves).
# ---------------------------------------------------------------------------


def test_furniture_alias_sofa_to_couch():
    assert pso._normalize_furniture_label("sofa") == "couch"


def test_furniture_alias_shelf_to_shelves():
    assert pso._normalize_furniture_label("shelf") == "shelves"


def test_furniture_alias_passthrough_for_unknown():
    assert pso._normalize_furniture_label("counter") == "counter"
    assert pso._normalize_furniture_label("Table") == "table"


# ---------------------------------------------------------------------------
# Template selection.
# ---------------------------------------------------------------------------


def test_safe_template_for_category_prefers_readable_ids():
    catalog = {
        "book": [
            "deadbeef0123456789abcdef0123456789abcdef",  # hash-only
            "Book_5",                                     # human readable
            "Book_15",
            "abc1234567890abcdef1234567890abcdef12345",
        ],
    }
    chosen = pso._safe_template_for_category(catalog, "book")
    assert chosen == "Book_15"  # alphabetical first among readable ids


def test_safe_template_for_category_falls_back_to_hash():
    catalog = {
        "bottle": [
            "deadbeef0123456789abcdef0123456789abcdef",
            "abc1234567890abcdef1234567890abcdef12345",
        ],
    }
    chosen = pso._safe_template_for_category(catalog, "bottle")
    # Only hashes available; alphabetical first.
    assert chosen == "abc1234567890abcdef1234567890abcdef12345"


def test_safe_template_for_category_returns_none_when_missing():
    assert pso._safe_template_for_category({}, "dragon") is None


# ---------------------------------------------------------------------------
# Furniture candidate selection.
# ---------------------------------------------------------------------------


def test_select_furniture_filters_by_category_and_room():
    instances = [
        {"template_name": "AAA", "translation": [0.0, 0.0, 0.0]},  # counter @ kitchen
        {"template_name": "BBB", "translation": [10.0, 0.0, 10.0]},  # counter @ outside
        {"template_name": "CCC", "translation": [1.0, 0.0, 1.0]},  # table   @ kitchen
    ]
    template_to_cat = {"AAA": "counter", "BBB": "counter", "CCC": "table"}
    rooms = {
        "kitchen__0": [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)],
    }

    cands = pso._select_furniture_candidates(
        instances, template_to_cat, "counter", "kitchen", rooms
    )
    assert len(cands) == 1
    assert cands[0][0]["template_name"] == "AAA"
    assert cands[0][1] == "kitchen__0"


def test_select_furniture_alias_sofa_matches_couch():
    instances = [
        {"template_name": "X", "translation": [0.0, 0.0, 0.0]},
    ]
    template_to_cat = {"X": "couch"}
    rooms = {}
    cands = pso._select_furniture_candidates(
        instances, template_to_cat, "sofa", None, rooms
    )
    assert len(cands) == 1


def test_select_furniture_no_room_hint_accepts_any_room():
    instances = [
        {"template_name": "A", "translation": [0.0, 0.0, 0.0]},
        {"template_name": "B", "translation": [50.0, 0.0, 50.0]},
    ]
    template_to_cat = {"A": "table", "B": "table"}
    rooms = {}
    cands = pso._select_furniture_candidates(
        instances, template_to_cat, "table", None, rooms
    )
    assert len(cands) == 2
