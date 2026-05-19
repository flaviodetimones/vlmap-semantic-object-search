"""Offline tests for the ROS-side heatmap helper (tfg-ros / vlmap_semantic_server)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROS_SRC = (
    Path(__file__).resolve().parents[1]
    / "ros_ws"
    / "src"
    / "vlmap_semantic_server"
    / "src"
)
if str(ROS_SRC) not in sys.path:
    sys.path.insert(0, str(ROS_SRC))

from vlmap_semantic_server.heatmap import (  # noqa: E402
    extract_top_candidates,
    load_heatmap_array,
    normalize_to_occupancy,
)


def test_load_heatmap_array_returns_none_when_missing(tmp_path):
    assert load_heatmap_array(str(tmp_path), "mug") is None


def test_load_heatmap_array_returns_none_when_dir_empty():
    assert load_heatmap_array("", "mug") is None


def test_load_heatmap_array_reads_npy(tmp_path):
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np.save(tmp_path / "mug.npy", arr)
    out = load_heatmap_array(str(tmp_path), "mug")
    assert out is not None
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, arr)


def test_normalize_to_occupancy_min_max_to_int8():
    arr = np.array([[0.0, 50.0], [100.0, 25.0]], dtype=np.float32)
    out = normalize_to_occupancy(arr)
    assert out.dtype == np.int8
    assert out.min() == 0
    assert out.max() == 100
    assert out[1, 1] == 25


def test_normalize_to_occupancy_constant_array_returns_zeros():
    out = normalize_to_occupancy(np.full((3, 3), 7.0, dtype=np.float32))
    np.testing.assert_array_equal(out, np.zeros((3, 3), dtype=np.int8))


def test_extract_top_candidates_returns_empty_for_n_zero():
    arr = np.ones((4, 4), dtype=np.float32)
    assert extract_top_candidates(arr, 0, 0.05, 0.0, 0.0) == []


def test_extract_top_candidates_picks_global_max_first():
    arr = np.zeros((5, 5), dtype=np.float32)
    arr[2, 3] = 1.0
    out = extract_top_candidates(arr, 1, resolution=0.5, origin_x=10.0, origin_y=20.0)
    assert len(out) == 1
    x, y, score = out[0]
    assert score == pytest.approx(1.0)
    # cell (row=2, col=3) → x = 10 + (3 + 0.5)*0.5, y = 20 + (2 + 0.5)*0.5
    assert x == pytest.approx(10.0 + 3.5 * 0.5)
    assert y == pytest.approx(20.0 + 2.5 * 0.5)


def test_extract_top_candidates_applies_nms():
    arr = np.zeros((10, 10), dtype=np.float32)
    arr[5, 5] = 1.0
    arr[5, 6] = 0.9  # very close, should be suppressed by NMS
    arr[0, 0] = 0.8
    out = extract_top_candidates(
        arr, 3, resolution=1.0, origin_x=0.0, origin_y=0.0, min_distance_cells=3
    )
    scores = [s for _, _, s in out]
    assert pytest.approx(1.0) in scores
    assert pytest.approx(0.8) in scores
    assert pytest.approx(0.9) not in scores


def test_extract_top_candidates_caps_to_n():
    arr = np.random.RandomState(0).rand(20, 20).astype(np.float32)
    out = extract_top_candidates(arr, 4, resolution=0.05, origin_x=0.0, origin_y=0.0)
    assert len(out) == 4
