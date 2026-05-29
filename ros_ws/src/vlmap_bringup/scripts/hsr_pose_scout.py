#!/usr/bin/env python3
"""Score candidate HSR poses in Gazebo by stability and RGB scene richness."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image


@dataclass
class PoseScore:
    x: float
    y: float
    yaw: float
    pose_error: float
    drift_xy: float
    drift_yaw: float
    intensity_std: float
    edge_std: float
    center_std: float
    score: float


def _yaw_to_quat(yaw: float) -> Quaternion:
    half = yaw * 0.5
    return Quaternion(0.0, 0.0, math.sin(half), math.cos(half))


def _quat_to_yaw(q: Quaternion) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _parse_candidate(text: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"candidate '{text}' must have x,y,yaw")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _image_to_gray(msg: Image) -> np.ndarray:
    rgb = _image_to_rgb(msg)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return gray


def _image_to_rgb(msg: Image) -> np.ndarray:
    if msg.encoding not in {"rgb8", "bgr8", "rgba8", "bgra8"}:
        raise ValueError(f"unsupported image encoding: {msg.encoding}")

    channels = 4 if "a8" in msg.encoding else 3
    frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
    if msg.encoding.startswith("bgr"):
        frame = frame[..., ::-1]
    return frame[..., :3].astype(np.uint8)


def _edge_std(gray: np.ndarray) -> float:
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    cropped = np.concatenate([gx[:, :-1].reshape(-1), gy[:-1, :].reshape(-1)])
    return float(np.std(cropped)) if cropped.size else 0.0


def _center_crop(gray: np.ndarray, frac: float = 0.4) -> np.ndarray:
    h, w = gray.shape
    hh = max(1, int(h * frac))
    ww = max(1, int(w * frac))
    y0 = max(0, (h - hh) // 2)
    x0 = max(0, (w - ww) // 2)
    return gray[y0 : y0 + hh, x0 : x0 + ww]


def _wait_msg(topic: str, msg_type, timeout: float):
    return rospy.wait_for_message(topic, msg_type, timeout=timeout)


def _capture_metrics(image_topic: str, timeout: float) -> Tuple[float, float, float, np.ndarray]:
    rgb = _image_to_rgb(_wait_msg(image_topic, Image, timeout))
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return float(np.std(gray)), _edge_std(gray), float(np.std(_center_crop(gray))), rgb


def _write_ppm(path: str, rgb: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode("ascii"))
        fh.write(rgb.tobytes())


def _odom_xy_yaw(msg: Odometry) -> Tuple[float, float, float]:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return float(p.x), float(p.y), float(_quat_to_yaw(q))


def _angle_diff(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def _set_pose(set_state, model_name: str, x: float, y: float, yaw: float, z: float) -> None:
    state = ModelState()
    state.model_name = model_name
    state.pose.position.x = x
    state.pose.position.y = y
    state.pose.position.z = z
    state.pose.orientation = _yaw_to_quat(yaw)
    state.reference_frame = "world"
    set_state(state)


def _score_pose(
    set_state,
    get_state,
    model_name: str,
    image_topic: str,
    odom_topic: str,
    x: float,
    y: float,
    yaw: float,
    *,
    settle_sec: float,
    z: float,
    timeout: float,
) -> Tuple[PoseScore, np.ndarray]:
    _set_pose(set_state, model_name, x, y, yaw, z)
    rospy.sleep(settle_sec)
    model_state = get_state(model_name, "world")
    actual = model_state.pose.position
    pose_error = math.hypot(actual.x - x, actual.y - y)
    actual_yaw = _quat_to_yaw(model_state.pose.orientation)
    odom1 = _wait_msg(odom_topic, Odometry, timeout)
    rospy.sleep(max(settle_sec * 0.75, 0.5))
    odom2 = _wait_msg(odom_topic, Odometry, timeout)
    x1, y1, yaw1 = _odom_xy_yaw(odom1)
    x2, y2, yaw2 = _odom_xy_yaw(odom2)
    drift_xy = math.hypot(x2 - x1, y2 - y1)
    drift_yaw = abs(_angle_diff(yaw2, yaw1))
    intensity_std, edge_std, center_std, rgb = _capture_metrics(image_topic, timeout)
    score = (
        intensity_std
        + 1.35 * edge_std
        + 0.75 * center_std
        - 350.0 * pose_error
        - 250.0 * drift_xy
        - 40.0 * drift_yaw
        - 10.0 * abs(_angle_diff(actual_yaw, yaw))
    )
    return PoseScore(x, y, yaw, pose_error, drift_xy, drift_yaw, intensity_std, edge_std, center_std, score), rgb


def _default_candidates(grid_values: Sequence[float], yaw_values: Sequence[float]) -> Sequence[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    for x in grid_values:
        for y in grid_values:
            for yaw in yaw_values:
                out.append((x, y, yaw))
    return out


def _iter_candidates(args) -> Iterable[Tuple[float, float, float]]:
    if args.candidate:
        return [_parse_candidate(text) for text in args.candidate]
    return _default_candidates(args.grid_values, args.yaw_values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="hsrb")
    parser.add_argument("--image-topic", default="/hsrb/head_rgbd_sensor/rgb/image_rect_color")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--settle-sec", type=float, default=1.5)
    parser.add_argument("--robot-z", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--grid-values", type=float, nargs="+", default=[-8.0, -4.0, 0.0, 4.0, 8.0])
    parser.add_argument("--yaw-values", type=float, nargs="+", default=[0.0, math.pi / 2.0, math.pi, -math.pi / 2.0])
    parser.add_argument("--save-best-image-path")
    parser.add_argument(
        "--candidate",
        action="append",
        help="candidate pose as x,y,yaw radians; can be repeated",
    )
    args = parser.parse_args(rospy.myargv(sys.argv)[1:])

    rospy.init_node("hsr_pose_scout", anonymous=True)
    rospy.wait_for_service("/gazebo/set_model_state", timeout=args.timeout)
    rospy.wait_for_service("/gazebo/get_model_state", timeout=args.timeout)
    set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

    scores: List[PoseScore] = []
    best_rgb: np.ndarray | None = None
    best_score = float("-inf")
    for x, y, yaw in _iter_candidates(args):
        try:
            score, rgb = _score_pose(
                set_state,
                get_state,
                args.model_name,
                args.image_topic,
                args.odom_topic,
                x,
                y,
                yaw,
                settle_sec=args.settle_sec,
                z=args.robot_z,
                timeout=args.timeout,
            )
        except Exception as exc:  # pragma: no cover - runtime utility
            rospy.logwarn("candidate x=%.2f y=%.2f yaw=%.2f failed: %s", x, y, yaw, exc)
            continue
        scores.append(score)
        if score.score > best_score:
            best_rgb = rgb
            best_score = score.score
        line = (
            f"candidate x={score.x:.2f} y={score.y:.2f} yaw={score.yaw:.2f} -> "
            f"score={score.score:.2f} pose_err={score.pose_error:.4f} drift={score.drift_xy:.4f} yaw_drift={score.drift_yaw:.4f} "
            f"std={score.intensity_std:.2f} edge={score.edge_std:.2f} center={score.center_std:.2f}"
        )
        rospy.loginfo(line)
        print(line, flush=True)

    if not scores:
        rospy.logerr("no valid candidates scored")
        return 1

    ranked = sorted(scores, key=lambda item: item.score, reverse=True)
    if args.save_best_image_path and best_rgb is not None:
        _write_ppm(args.save_best_image_path, best_rgb)
        print(f"Saved best RGB frame to {args.save_best_image_path}", flush=True)
    print("Top candidate poses:")
    for idx, item in enumerate(ranked[: max(args.top_k, 1)], start=1):
        print(
            f"{idx:02d}. x={item.x:.2f} y={item.y:.2f} yaw={item.yaw:.2f} "
            f"score={item.score:.2f} pose_err={item.pose_error:.4f} drift_xy={item.drift_xy:.4f} drift_yaw={item.drift_yaw:.4f} "
            f"std={item.intensity_std:.2f} edge={item.edge_std:.2f} center={item.center_std:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
