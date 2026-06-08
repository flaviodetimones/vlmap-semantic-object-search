#!/usr/bin/env python3
"""Capture RGB-only Habitat/HSSD background frames for synthetic detector data.

This is intentionally separate from VLMap capture/building. It samples
navigable camera poses in an HSSD scene and saves only RGB images plus a small
metadata JSONL file. The frames are later used as backgrounds for automatic
object compositing and YOLO label generation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable

import cv2
import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis


DEFAULT_SCENE_DATASET_CONFIG = (
    "/workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json"
)


def _make_cfg(
    scene_dataset_config: str,
    scene_id: str,
    width: int,
    height: int,
    sensor_height: float,
    pitch_deg: float,
) -> habitat_sim.Configuration:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = scene_dataset_config
    sim_cfg.scene_id = scene_id
    sim_cfg.enable_physics = False

    color_spec = habitat_sim.CameraSensorSpec()
    color_spec.uuid = "color_sensor"
    color_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_spec.resolution = [height, width]
    color_spec.position = [0.0, sensor_height, 0.0]
    color_spec.orientation = [math.radians(pitch_deg), 0.0, 0.0]
    color_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [color_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    }
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def _ensure_navmesh(sim: habitat_sim.Simulator, agent_radius: float, agent_height: float) -> None:
    nav_settings = habitat_sim.NavMeshSettings()
    nav_settings.set_defaults()
    nav_settings.agent_radius = agent_radius
    nav_settings.agent_height = agent_height
    try:
        nav_settings.include_static_objects = True
    except AttributeError:
        pass
    try:
        sim.recompute_navmesh(sim.pathfinder, nav_settings, include_static_objects=True)
    except TypeError:
        sim.recompute_navmesh(sim.pathfinder, nav_settings)
    if not sim.pathfinder.is_loaded:
        raise RuntimeError("Habitat navmesh failed to load")


def _yaw_values(count: int, rng: random.Random, mode: str) -> Iterable[float]:
    if mode == "sweep4":
        base = rng.uniform(-math.pi, math.pi)
        for k in range(4):
            yield base + k * (math.pi / 2.0)
        return
    if mode == "sweep8":
        base = rng.uniform(-math.pi, math.pi)
        for k in range(8):
            yield base + k * (math.pi / 4.0)
        return
    for _ in range(count):
        yield rng.uniform(-math.pi, math.pi)


def _set_agent_pose(agent, position: np.ndarray, yaw_rad: float) -> None:
    state = habitat_sim.AgentState()
    state.position = position
    state.rotation = quat_from_angle_axis(yaw_rad, np.array([0.0, 1.0, 0.0]))
    agent.set_state(state)


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"failed to write image: {path}")


def _angle_delta_rad(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _pose_changed_enough(
    pos: np.ndarray,
    yaw_rad: float,
    last_pos: np.ndarray | None,
    last_yaw_rad: float | None,
    min_distance_m: float,
    min_yaw_deg: float,
) -> bool:
    if last_pos is None or last_yaw_rad is None:
        return True
    dist = float(np.linalg.norm(np.asarray(pos)[[0, 2]] - np.asarray(last_pos)[[0, 2]]))
    yaw_delta = math.degrees(_angle_delta_rad(yaw_rad, last_yaw_rad))
    return dist >= min_distance_m or yaw_delta >= min_yaw_deg


def _save_frame(
    image_dir: Path,
    meta_f,
    frame_id: int,
    rgb: np.ndarray,
    scene_id: str,
    position,
    yaw_rad: float,
    pitch_deg: float,
    sensor_height: float,
) -> None:
    filename = f"{frame_id:06d}.png"
    _write_rgb(image_dir / filename, rgb)
    record = {
        "file": f"images/{filename}",
        "scene_id": scene_id,
        "position": [float(v) for v in position],
        "yaw_rad": float(yaw_rad),
        "pitch_deg": float(pitch_deg),
        "sensor_height": float(sensor_height),
    }
    meta_f.write(json.dumps(record, sort_keys=True) + "\n")
    meta_f.flush()


def _passes_image_filters(rgb: np.ndarray, min_mean: float, min_std: float) -> bool:
    gray = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2GRAY)
    return float(np.mean(gray)) >= min_mean and float(np.std(gray)) >= min_std


def _run_auto_capture(sim, agent, image_dir: Path, meta_f, args) -> int:
    rng = random.Random(args.seed)
    saved = 0
    attempts = 0
    max_attempts = max(args.frames * 20, 200)
    start = time.time()
    while saved < args.frames and attempts < max_attempts:
        attempts += 1
        pos = sim.pathfinder.get_random_navigable_point()
        yaw_list = list(_yaw_values(1, rng, args.yaw_mode))
        for yaw_rad in yaw_list:
            if saved >= args.frames:
                break
            _set_agent_pose(agent, pos, yaw_rad)
            obs = sim.get_sensor_observations()
            rgb = obs["color_sensor"][:, :, :3]
            if not _passes_image_filters(rgb, args.min_mean_brightness, args.min_std_brightness):
                continue
            state = agent.get_state()
            _save_frame(
                image_dir,
                meta_f,
                saved,
                rgb,
                args.scene_id,
                state.position,
                yaw_rad,
                args.pitch_deg,
                args.sensor_height,
            )
            saved += 1
            if saved % 100 == 0:
                elapsed = time.time() - start
                print(f"[backgrounds] saved={saved}/{args.frames} elapsed={elapsed:.1f}s")

    if saved < args.frames:
        raise RuntimeError(f"saved only {saved}/{args.frames} frames after {attempts} attempts")
    return saved


def _run_manual_capture(sim, agent, image_dir: Path, meta_f, args) -> int:
    state = habitat_sim.AgentState()
    if args.start_x is not None and args.start_z is not None:
        state.position = sim.pathfinder.snap_point(
            np.array([args.start_x, 0.0, args.start_z], dtype=np.float32)
        )
    else:
        state.position = sim.pathfinder.get_random_navigable_point()
    current_yaw = math.radians(args.start_yaw_deg)
    state.rotation = quat_from_angle_axis(current_yaw, np.array([0.0, 1.0, 0.0]))
    agent.set_state(state)

    saved = 0
    last_saved_pos = None
    last_saved_yaw = None

    print("")
    print("[backgrounds/manual] Controls:")
    print("  w = move forward")
    print("  a = turn left")
    print("  d = turn right")
    print("  s = force-save current frame")
    print("  q = quit")
    print(
        f"  auto-save threshold: {args.min_distance_m:.2f} m or "
        f"{args.min_yaw_deg:.1f} deg"
    )
    print("")

    while True:
        obs = sim.get_sensor_observations()
        rgb = obs["color_sensor"][:, :, :3]
        display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hud = (
            f"saved={saved}  threshold={args.min_distance_m:.1f}m/"
            f"{args.min_yaw_deg:.0f}deg  w/a/d move  s save  q quit"
        )
        cv2.putText(display, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow("HSSD background capture", display)
        key = cv2.waitKey(30) & 0xFF

        moved = False
        forced = False
        if key == ord("w"):
            agent.act("move_forward")
            moved = True
        elif key == ord("a"):
            agent.act("turn_left")
            current_yaw += math.radians(10.0)
            moved = True
        elif key == ord("d"):
            agent.act("turn_right")
            current_yaw -= math.radians(10.0)
            moved = True
        elif key == ord("s"):
            forced = True
        elif key == ord("q"):
            break

        if not moved and not forced:
            continue

        obs = sim.get_sensor_observations()
        rgb = obs["color_sensor"][:, :, :3]
        state = agent.get_state()
        if not forced and not _passes_image_filters(
            rgb, args.min_mean_brightness, args.min_std_brightness
        ):
            continue
        should_save = forced or _pose_changed_enough(
            state.position,
            current_yaw,
            last_saved_pos,
            last_saved_yaw,
            args.min_distance_m,
            args.min_yaw_deg,
        )
        if should_save:
            _save_frame(
                image_dir,
                meta_f,
                saved,
                rgb,
                args.scene_id,
                state.position,
                current_yaw,
                args.pitch_deg,
                args.sensor_height,
            )
            last_saved_pos = np.array(state.position, dtype=np.float32)
            last_saved_yaw = current_yaw
            saved += 1
            print(f"[backgrounds/manual] saved frame {saved:06d}")
            if saved >= args.frames:
                print("[backgrounds/manual] frame target reached")
                break

    cv2.destroyAllWindows()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dataset-config", default=DEFAULT_SCENE_DATASET_CONFIG)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["auto", "manual"], default="auto")
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--sensor-height", type=float, default=1.35)
    parser.add_argument("--pitch-deg", type=float, default=-15.0)
    parser.add_argument("--yaw-mode", choices=["random", "sweep4", "sweep8"], default="sweep4")
    parser.add_argument("--agent-radius", type=float, default=0.18)
    parser.add_argument("--agent-height", type=float, default=1.5)
    parser.add_argument("--min-mean-brightness", type=float, default=8.0)
    parser.add_argument(
        "--min-std-brightness",
        type=float,
        default=18.0,
        help="Reject near-flat wall/ceiling frames with little visual texture.",
    )
    parser.add_argument("--min-distance-m", type=float, default=1.0)
    parser.add_argument("--min-yaw-deg", type=float, default=30.0)
    parser.add_argument("--start-x", type=float, default=None)
    parser.add_argument("--start-z", type=float, default=None)
    parser.add_argument("--start-yaw-deg", type=float, default=0.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"

    cfg = _make_cfg(
        args.scene_dataset_config,
        args.scene_id,
        args.width,
        args.height,
        args.sensor_height,
        args.pitch_deg,
    )

    sim = habitat_sim.Simulator(cfg)
    try:
        _ensure_navmesh(sim, args.agent_radius, args.agent_height)
        agent = sim.initialize_agent(0)

        with meta_path.open("w", encoding="utf-8") as meta_f:
            if args.mode == "manual":
                saved = _run_manual_capture(sim, agent, image_dir, meta_f, args)
            else:
                saved = _run_auto_capture(sim, agent, image_dir, meta_f, args)
        print(f"[backgrounds] wrote {saved} RGB backgrounds to {out_dir}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
