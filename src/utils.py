"""
Shared pure-Python / numpy utility functions used by baselines and other scripts.
No simulator or GPU dependencies — safe to import anywhere.
"""

import numpy as np


# ── Camera intrinsics (from vlmaps config: cam_calib_mat) ────────────────────
CAM_CALIB = np.array([[540, 0, 540],
                       [0, 540, 360],
                       [0,   0,   1]], dtype=np.float64)

# ── Base / camera transforms (from vlmap_builder config) ─────────────────────
BASE_TRANSFORM = np.array([
    [0,  0, -1, 0],
    [-1, 0,  0, 0],
    [0,  1,  0, 0],
    [0,  0,  0, 1],
], dtype=np.float64)

BASE2CAM_TF = np.array([
    [1,  0,  0,  0],
    [0, -1,  0,  1.5],   # camera_height = 1.5
    [0,  0, -1,  0],
    [0,  0,  0,  1],
], dtype=np.float64)


def map_dist(a, b, cs):
    """Euclidean distance in metres between two (row, col) map cells."""
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) * cs)


def is_near_gt(robot_pos, gt_positions, cs, radius_m=1.0):
    """Check if robot_pos is within radius_m of any GT position."""
    for gt in gt_positions:
        if map_dist(robot_pos, gt, cs) <= radius_m:
            return True
    return False


def backproject_bbox_center(bbox_xyxy, depth_map, patch_radius=5, cam_calib=None):
    """
    Back-project the center of a 2D bounding box to a 3D camera-frame point.
    Returns (X, Y, Z) in camera frame or None if depth is invalid.
    """
    if cam_calib is None:
        cam_calib = CAM_CALIB

    x1, y1, x2, y2 = bbox_xyxy
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    h, w = depth_map.shape
    cx = max(0, min(cx, w - 1))
    cy = max(0, min(cy, h - 1))

    r = patch_radius
    patch = depth_map[max(0, cy-r):min(h, cy+r+1),
                      max(0, cx-r):min(w, cx+r+1)]
    valid = patch[(patch > 0.1) & (patch < 10.0)]
    if len(valid) == 0:
        return None

    d = float(np.median(valid))

    K_inv = np.linalg.inv(cam_calib)
    uv1 = np.array([cx + 0.5, cy + 0.5, 1.0])
    p_cam = K_inv @ uv1 * d
    return p_cam


def cam_to_world(p_cam, pose_tf, init_base_tf_inv):
    """
    Transform a point from camera frame to world (init-base) frame.
    """
    base_pose = BASE_TRANSFORM @ pose_tf @ np.linalg.inv(BASE_TRANSFORM)
    tf = init_base_tf_inv @ base_pose
    pc_transform = tf @ BASE_TRANSFORM @ BASE2CAM_TF
    p_hom = np.append(p_cam, 1.0)
    p_world = pc_transform @ p_hom
    return p_world[:3]


def cluster_detections(detections_3d, merge_radius_m=0.5):
    """
    Merge nearby detections of the same class.
    Keep the highest-confidence one per cluster.
    """
    if not detections_3d:
        return detections_3d

    by_class = {}
    for d in detections_3d:
        by_class.setdefault(d["class_name"], []).append(d)

    merged = []
    for cls, dets in by_class.items():
        dets.sort(key=lambda x: -x["confidence"])
        used = [False] * len(dets)
        for i, d in enumerate(dets):
            if used[i]:
                continue
            used[i] = True
            pi = np.array(d["pos_3d_world"])
            for j in range(i + 1, len(dets)):
                if used[j]:
                    continue
                pj = np.array(dets[j]["pos_3d_world"])
                if np.linalg.norm(pi - pj) < merge_radius_m:
                    used[j] = True
            merged.append(d)
    return merged
