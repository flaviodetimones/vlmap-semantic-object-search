"""Offline tests for the file-backed heatmap dumper (tfg-sim side)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from vlmap_ros_export.heatmap_dumper import (  # noqa: E402
    dump_heatmap_npy,
    normalize_heatmap,
    pool_voxel_scores_to_2d,
)


def test_pool_voxel_scores_to_2d_keeps_max_per_cell():
    grid_pos = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],  # same (row, col), higher z, lower score → must NOT overwrite
            [1, 2, 0],
        ],
        dtype=np.int32,
    )
    scores = np.array([0.7, 0.1, 0.5], dtype=np.float32)
    out = pool_voxel_scores_to_2d(scores, grid_pos, gs=4)
    assert out.shape == (4, 4)
    assert out.dtype == np.float32
    assert out[0, 0] == pytest.approx(0.7)
    assert out[1, 2] == pytest.approx(0.5)
    assert out[2, 2] == 0.0  # untouched cell stays zero


def test_pool_voxel_scores_to_2d_rejects_length_mismatch():
    with pytest.raises(ValueError):
        pool_voxel_scores_to_2d(
            np.zeros(3, dtype=np.float32),
            np.zeros((2, 3), dtype=np.int32),
            gs=4,
        )


def test_dump_heatmap_npy_writes_loadable_file(tmp_path):
    arr = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    path = dump_heatmap_npy(arr, str(tmp_path), "mug")
    assert path == tmp_path / "mug.npy"
    loaded = np.load(path)
    np.testing.assert_array_equal(loaded, arr)


def test_dump_heatmap_npy_sanitizes_filename(tmp_path):
    arr = np.zeros((2, 2), dtype=np.float32)
    path = dump_heatmap_npy(arr, str(tmp_path), "coffee mug/large")
    assert path.name == "coffee_mug_large.npy"


def test_dump_heatmap_npy_rotate_matches_flipud_transpose(tmp_path):
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    path = dump_heatmap_npy(arr, str(tmp_path), "x", rotate_to_ros=True)
    loaded = np.load(path)
    np.testing.assert_array_equal(loaded, np.flipud(arr.T))


def test_dump_heatmap_npy_rejects_non_2d(tmp_path):
    with pytest.raises(ValueError):
        dump_heatmap_npy(np.zeros(4, dtype=np.float32), str(tmp_path), "mug")


def test_dump_heatmap_npy_rejects_empty_category(tmp_path):
    with pytest.raises(ValueError):
        dump_heatmap_npy(np.zeros((2, 2), dtype=np.float32), str(tmp_path), "   ")


def test_normalize_heatmap_min_max():
    arr = np.array([[0.0, 5.0], [10.0, 7.5]], dtype=np.float32)
    out = normalize_heatmap(arr)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)
    assert out[1, 1] == pytest.approx(0.75)


def test_normalize_heatmap_constant_returns_zeros():
    arr = np.full((3, 3), 4.2, dtype=np.float32)
    out = normalize_heatmap(arr)
    np.testing.assert_array_equal(out, np.zeros((3, 3), dtype=np.float32))
