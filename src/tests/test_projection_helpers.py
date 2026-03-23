"""
Tests for backproject_bbox_center, cam_to_world, and cluster_detections
from utils.py (Step 5 helpers).

Run on host (no Docker/GPU needed):
    python -m pytest src/tests/test_projection_helpers.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    backproject_bbox_center,
    cam_to_world,
    cluster_detections,
    CAM_CALIB,
    BASE_TRANSFORM,
    BASE2CAM_TF,
)


# ═══════════════════════════════════════════════════════════════════════════════
# backproject_bbox_center
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackprojectBboxCenter:
    def _make_depth(self, val, h=720, w=1080):
        return np.full((h, w), val, dtype=np.float32)

    def test_center_pixel_at_known_depth(self):
        """Image center should project to (0, 0, d) in camera frame."""
        depth = self._make_depth(3.0)
        bbox = [490, 310, 590, 410]  # center at (540, 360) = principal point
        p = backproject_bbox_center(bbox, depth, patch_radius=5)
        assert p is not None
        # At principal point: x ≈ 0, y ≈ 0, z ≈ depth
        assert abs(p[2] - 3.0) < 0.05, f"z={p[2]}"
        assert abs(p[0]) < 0.05, f"x={p[0]}"
        assert abs(p[1]) < 0.05, f"y={p[1]}"

    def test_returns_none_for_zero_depth(self):
        depth = self._make_depth(0.0)
        bbox = [100, 100, 200, 200]
        assert backproject_bbox_center(bbox, depth) is None

    def test_returns_none_for_depth_too_far(self):
        depth = self._make_depth(15.0)  # > 10 m
        bbox = [100, 100, 200, 200]
        assert backproject_bbox_center(bbox, depth) is None

    def test_valid_depth_range(self):
        depth = self._make_depth(1.5)  # within [0.1, 10)
        bbox = [100, 100, 200, 200]
        p = backproject_bbox_center(bbox, depth)
        assert p is not None
        assert len(p) == 3

    def test_off_center_projects_correctly(self):
        """A bbox at the right edge should have positive x in camera frame."""
        depth = self._make_depth(2.0)
        bbox = [900, 310, 1000, 410]  # center ~(950, 360), right of principal
        p = backproject_bbox_center(bbox, depth)
        assert p is not None
        # u > cx=540 → x > 0 in camera frame (for standard pinhole)
        assert p[0] > 0, f"Expected positive x for right-of-center, got {p[0]}"

    def test_bbox_clamped_to_image(self):
        """Bbox partially outside image should still work."""
        depth = self._make_depth(2.0, h=100, w=100)
        bbox = [80, 80, 120, 120]  # center at (100, 100), clamped to (99, 99)
        p = backproject_bbox_center(bbox, depth)
        assert p is not None

    def test_custom_cam_calib(self):
        K = np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1]], dtype=np.float64)
        depth = np.full((480, 640), 2.0, dtype=np.float32)
        bbox = [270, 190, 370, 290]  # center ~ (320, 240) = principal point
        p = backproject_bbox_center(bbox, depth, cam_calib=K)
        assert p is not None
        assert abs(p[2] - 2.0) < 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# cam_to_world
# ═══════════════════════════════════════════════════════════════════════════════

class TestCamToWorld:
    def test_identity_pose(self):
        """With identity pose and init_base_tf, output should be deterministic."""
        pose_tf = np.eye(4)
        init_base_tf = BASE_TRANSFORM @ pose_tf @ np.linalg.inv(BASE_TRANSFORM)
        init_base_tf_inv = np.linalg.inv(init_base_tf)
        p_cam = np.array([0.0, 0.0, 2.0])
        p_world = cam_to_world(p_cam, pose_tf, init_base_tf_inv)
        assert len(p_world) == 3
        # Should be finite
        assert np.all(np.isfinite(p_world))

    def test_output_shape(self):
        pose_tf = np.eye(4)
        init_base_tf_inv = np.eye(4)
        p = cam_to_world(np.array([1.0, 2.0, 3.0]), pose_tf, init_base_tf_inv)
        assert p.shape == (3,)

    def test_different_poses_give_different_results(self):
        init_inv = np.eye(4)
        p_cam = np.array([0.0, 0.0, 2.0])

        pose1 = np.eye(4)
        pose2 = np.eye(4)
        pose2[0, 3] = 5.0  # translate 5m along x

        w1 = cam_to_world(p_cam, pose1, init_inv)
        w2 = cam_to_world(p_cam, pose2, init_inv)
        assert not np.allclose(w1, w2), "Different poses should give different world points"


# ═══════════════════════════════════════════════════════════════════════════════
# cluster_detections
# ═══════════════════════════════════════════════════════════════════════════════

class TestClusterDetections:
    def test_empty_input(self):
        assert cluster_detections([]) == []

    def test_single_detection(self):
        dets = [{"class_name": "chair", "confidence": 0.9, "pos_3d_world": [1, 2, 0]}]
        result = cluster_detections(dets, merge_radius_m=0.5)
        assert len(result) == 1

    def test_merges_nearby_same_class(self):
        dets = [
            {"class_name": "chair", "confidence": 0.9, "pos_3d_world": [1.0, 2.0, 0]},
            {"class_name": "chair", "confidence": 0.8, "pos_3d_world": [1.1, 2.1, 0]},
        ]
        result = cluster_detections(dets, merge_radius_m=0.5)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9  # keeps highest confidence

    def test_does_not_merge_far_same_class(self):
        dets = [
            {"class_name": "chair", "confidence": 0.9, "pos_3d_world": [1, 2, 0]},
            {"class_name": "chair", "confidence": 0.8, "pos_3d_world": [5, 6, 0]},
        ]
        result = cluster_detections(dets, merge_radius_m=0.5)
        assert len(result) == 2

    def test_does_not_merge_different_class(self):
        dets = [
            {"class_name": "chair", "confidence": 0.9, "pos_3d_world": [1, 2, 0]},
            {"class_name": "table", "confidence": 0.8, "pos_3d_world": [1, 2, 0]},
        ]
        result = cluster_detections(dets, merge_radius_m=0.5)
        assert len(result) == 2

    def test_three_nearby_merged_to_one(self):
        dets = [
            {"class_name": "chair", "confidence": 0.7, "pos_3d_world": [1.0, 1.0, 0]},
            {"class_name": "chair", "confidence": 0.9, "pos_3d_world": [1.1, 1.0, 0]},
            {"class_name": "chair", "confidence": 0.8, "pos_3d_world": [1.2, 1.0, 0]},
        ]
        result = cluster_detections(dets, merge_radius_m=0.5)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9
