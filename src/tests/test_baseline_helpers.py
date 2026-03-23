"""
Unit tests for the pure helper functions in run_baselines.py.

These tests only need numpy and can run on the host (no Docker / GPU needed):
    python -m pytest src/tests/test_baseline_helpers.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import map_dist, is_near_gt


# ─────────────────────────────────────────────────────────────────────────────
# map_dist
# ─────────────────────────────────────────────────────────────────────────────

class TestMapDist:
    def test_same_point(self):
        assert map_dist((100, 200), (100, 200), cs=0.05) == 0.0

    def test_horizontal(self):
        # 20 cells apart horizontally, cs=0.05 → 1.0 m
        d = map_dist((100, 100), (100, 120), cs=0.05)
        assert abs(d - 1.0) < 1e-6

    def test_vertical(self):
        d = map_dist((100, 100), (120, 100), cs=0.05)
        assert abs(d - 1.0) < 1e-6

    def test_diagonal(self):
        # 10 cells in each axis → sqrt(10^2+10^2)*0.05 = sqrt(200)*0.05
        d = map_dist((100, 100), (110, 110), cs=0.05)
        expected = np.sqrt(200) * 0.05
        assert abs(d - expected) < 1e-6

    def test_different_cs(self):
        d = map_dist((0, 0), (0, 10), cs=0.1)
        assert abs(d - 1.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# is_near_gt
# ─────────────────────────────────────────────────────────────────────────────

class TestIsNearGt:
    def test_exact_match(self):
        gt = [(100, 200)]
        assert is_near_gt((100, 200), gt, cs=0.05, radius_m=1.0)

    def test_within_radius(self):
        # 10 cells = 0.5m, radius 1.0m → True
        gt = [(100, 200)]
        assert is_near_gt((100, 210), gt, cs=0.05, radius_m=1.0)

    def test_outside_radius(self):
        # 30 cells = 1.5m, radius 1.0m → False
        gt = [(100, 200)]
        assert not is_near_gt((100, 230), gt, cs=0.05, radius_m=1.0)

    def test_on_boundary(self):
        # Exactly at radius: 20 cells * 0.05 = 1.0m, radius_m=1.0 → True (<=)
        gt = [(100, 200)]
        assert is_near_gt((100, 220), gt, cs=0.05, radius_m=1.0)

    def test_multiple_gt_objects(self):
        gt = [(0, 0), (500, 500)]
        # Far from (0,0) but close to (500,500)
        assert is_near_gt((500, 510), gt, cs=0.05, radius_m=1.0)

    def test_no_gt_objects(self):
        assert not is_near_gt((100, 200), [], cs=0.05, radius_m=1.0)

    def test_just_outside(self):
        # 21 cells * 0.05 = 1.05m > 1.0m → False
        gt = [(100, 200)]
        assert not is_near_gt((100, 221), gt, cs=0.05, radius_m=1.0)
