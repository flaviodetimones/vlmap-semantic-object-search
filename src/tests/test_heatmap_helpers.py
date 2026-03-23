"""
Tests for extract_regions from analyze_vlmap_heatmap.py (Step 6 helpers).

Uses synthetic masks and heatmaps — no VLMap or GPU needed.

Run on host:
    python -m pytest src/tests/test_heatmap_helpers.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="cv2 required for heatmap tests")
pytest.importorskip("scipy", reason="scipy required for heatmap tests")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyze_vlmap_heatmap import extract_regions


def _make_blob_mask(gs, center_rc, radius):
    """Create a binary mask with a filled circle."""
    mask = np.zeros((gs, gs), dtype=bool)
    cv2.circle(mask.astype(np.uint8), (center_rc[1], center_rc[0]), radius, 1, -1)
    return cv2.circle(
        np.zeros((gs, gs), dtype=np.uint8),
        (center_rc[1], center_rc[0]),
        radius, 1, -1,
    ).astype(bool)


def _make_heatmap(gs, mask, score=0.8):
    """Create a heatmap with uniform score inside mask."""
    h = np.zeros((gs, gs), dtype=np.float32)
    h[mask] = score
    return h


class TestExtractRegions:
    def test_single_blob(self):
        gs = 200
        mask = _make_blob_mask(gs, (100, 100), 20)
        heatmap = _make_heatmap(gs, mask, score=0.8)
        regions = extract_regions(mask, heatmap, cs=0.05)
        assert len(regions) >= 1
        assert regions[0]["mean_score"] >= 0.3
        assert regions[0]["rank"] == 1

    def test_empty_mask_returns_no_regions(self):
        gs = 200
        mask = np.zeros((gs, gs), dtype=bool)
        heatmap = np.zeros((gs, gs), dtype=np.float32)
        regions = extract_regions(mask, heatmap, cs=0.05)
        assert len(regions) == 0

    def test_low_score_filtered(self):
        gs = 200
        mask = _make_blob_mask(gs, (100, 100), 20)
        heatmap = _make_heatmap(gs, mask, score=0.1)  # below MIN_MEAN_SCORE=0.3
        regions = extract_regions(mask, heatmap, cs=0.05)
        assert len(regions) == 0

    def test_tiny_blob_filtered(self):
        gs = 200
        mask = _make_blob_mask(gs, (100, 100), 2)  # ~12 px area, below MIN_AREA_PX=50
        heatmap = _make_heatmap(gs, mask, score=0.8)
        regions = extract_regions(mask, heatmap, cs=0.05)
        assert len(regions) == 0

    def test_two_blobs_ranked(self):
        gs = 300
        mask1 = _make_blob_mask(gs, (80, 80), 20)
        mask2 = _make_blob_mask(gs, (200, 200), 30)
        mask = mask1 | mask2
        # Give blob2 higher score
        heatmap = np.zeros((gs, gs), dtype=np.float32)
        heatmap[mask1] = 0.5
        heatmap[mask2] = 0.9
        regions = extract_regions(mask, heatmap, cs=0.05)
        assert len(regions) >= 2
        # Rank 1 should have higher quality
        assert regions[0]["quality"] >= regions[1]["quality"]

    def test_area_m2_correct(self):
        gs = 200
        mask = _make_blob_mask(gs, (100, 100), 20)
        heatmap = _make_heatmap(gs, mask, score=0.8)
        cs = 0.05
        regions = extract_regions(mask, heatmap, cs=cs)
        if len(regions) > 0:
            r = regions[0]
            expected_area = r["area_px"] * cs * cs
            assert abs(r["area_m2"] - expected_area) < 0.01

    def test_density_within_bounds(self):
        gs = 200
        mask = _make_blob_mask(gs, (100, 100), 20)
        heatmap = _make_heatmap(gs, mask, score=0.8)
        regions = extract_regions(mask, heatmap, cs=0.05)
        for r in regions:
            assert 0 < r["density"] <= 1.0
