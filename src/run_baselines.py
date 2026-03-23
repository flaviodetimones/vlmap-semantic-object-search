"""
run_baselines.py
================
Implements and evaluates two object-search baselines on a VLMap scene:

  M1 — Random Search:   visit VLMap candidate positions in random order.
  M2 — Nearest First:   always visit the nearest remaining candidate.

Both baselines:
  1. Query the VLMap for the target object → extract candidate centroids.
  2. Retrieve ground-truth object positions from the Habitat semantic scene.
  3. Navigate to candidates according to the policy.
  4. Declare SUCCESS when arriving within SUCCESS_RADIUS_M of a GT object.

Metrics logged per episode (JSON):
  success, steps, path_length_m, candidates_visited,
  correct_first_goal, wrong_visits

Usage (inside Docker):
  python src/run_baselines.py scene_id=0 +query=chair +method=random  +episodes=10
  python src/run_baselines.py scene_id=0 +query=chair +method=nearest +episodes=10
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import distance_transform_edt

# ── VLMaps imports ───────────────────────────────────────────────────────────
from vlmaps.robot.habitat_lang_robot import HabitatLanguageRobot
from vlmaps.utils.mapping_utils import cvt_pose_vec2tf, base_pos2grid_id_3d
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.visualize_utils import pool_3d_label_to_2d
from vlmaps.utils.habitat_utils import agent_state2tf

# ── Constants ────────────────────────────────────────────────────────────────
from utils import map_dist, is_near_gt

SUCCESS_RADIUS_M = 1.0      # robot must be within 1 m of a GT object
MAX_STEPS        = 2000     # action budget per episode
MIN_CANDIDATES   = 1        # skip episode if VLMap finds fewer candidates


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_gt_object_positions(sim, query: str, gs: int, cs: float):
    """
    Return a list of (row, col) map positions for all Habitat semantic-scene
    objects whose category name contains *query* (case-insensitive).
    """
    positions = []
    for obj in sim.semantic_scene.objects:
        if obj is None or obj.category is None:
            continue
        if query.lower() in obj.category.name().lower():
            center = obj.aabb.center  # (x_hab, y_hab, z_hab)
            row, col, _ = base_pos2grid_id_3d(gs, cs, center[0], center[1], center[2])
            positions.append((row, col))
    return positions


def _snap_to_navigable(row, col, robot):
    """
    Snap (row, col) in full-map coords to the nearest free cell on the
    eroded obstacle map (the same map the visgraph plans on).
    Returns (row, col) in full-map coords, or None if impossible.
    """
    rmin = robot.map.rmin
    cmin = robot.map.cmin
    eroded = robot.nav_eroded_map
    ch, cw = eroded.shape

    # Convert to cropped coords
    cr = int(round(row - rmin))
    cc = int(round(col - cmin))

    # If already inside cropped bounds and on free (eroded) space, keep it
    if 0 <= cr < ch and 0 <= cc < cw and eroded[cr, cc]:
        return (float(rmin + cr), float(cmin + cc))

    # Find nearest free cell on the eroded map
    free_rows, free_cols = np.where(eroded)
    if len(free_rows) == 0:
        return None
    dist_sq = (free_rows - cr) ** 2 + (free_cols - cc) ** 2
    best = np.argmin(dist_sq)
    return (float(rmin + free_rows[best]), float(cmin + free_cols[best]))


def find_random_navigable_start(robot):
    """
    Pick a random trajectory pose that sits on free space, inside the
    cropped map bounds, and far from obstacles.  Returns a 4×4 transform.
    """
    cropped = robot.nav_eroded_map
    dist_map = distance_transform_edt(cropped)
    rmin, cmin = robot.map.rmin, robot.map.cmin
    ch, cw = cropped.shape

    poses = robot.vlmaps_dataloader.base_poses
    n = len(poses)
    gs = robot.map.obstacles_map.shape[0]

    # Score every pose by its distance to obstacles on the cropped map
    scores = np.zeros(n)
    for i in range(n):
        tf = cvt_pose_vec2tf(poses[i])
        pos = tf[:3, 3]
        row, col, _ = base_pos2grid_id_3d(gs, robot.cs, pos[0], pos[1], pos[2])
        cr = int(round(row - rmin))
        cc = int(round(col - cmin))
        if 0 <= cr < ch and 0 <= cc < cw:
            scores[i] = dist_map[cr, cc]
        # else stays 0 — outside cropped map, never selected

    # Shuffle indices, but prefer poses deep inside free space (dist > 5)
    indices = np.random.permutation(n)
    for idx in indices:
        if scores[idx] > 5:
            tf = cvt_pose_vec2tf(poses[idx])
            robot.set_agent_state(tf)
            robot._set_nav_curr_pose()
            return tf

    # Fallback: use the pose with the largest distance to obstacles
    best_idx = int(np.argmax(scores))
    tf = cvt_pose_vec2tf(poses[best_idx])
    robot.set_agent_state(tf)
    robot._set_nav_curr_pose()
    return tf


def get_vlmap_candidates(robot, query: str):
    """
    Query the VLMap for *query* and return a list of candidate (row, col)
    centroids on the full map, snapped to navigable space.
    """
    contours, centers, bbox_list = robot.map.get_pos(query)
    if not centers:
        return []
    # Filter small blobs
    ids = robot.map.filter_small_objects(bbox_list)
    if ids:
        centers = [centers[i] for i in ids]

    candidates = []
    for c in centers:
        snapped = _snap_to_navigable(float(c[0]), float(c[1]), robot)
        if snapped is not None:
            candidates.append(snapped)
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def _show_result_overlay(robot, query, found):
    """Show a 2-second overlay on the robot's RGB view indicating HIT or MISS."""
    obs = robot.sim.get_sensor_observations(0)
    rgba = obs["color_sensor"]
    img = cv2.cvtColor(np.array(rgba), cv2.COLOR_RGBA2BGR)
    h, w = img.shape[:2]

    label = query.capitalize()
    if found:
        text = f"{label} found!"
        color = (0, 200, 0)       # green
    else:
        text = f"{label} not found"
        color = (0, 0, 220)       # red

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.5
    thickness = 3
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    tx = (w - tw) // 2
    ty = (h + th) // 2

    # Dark semi-transparent banner behind text
    overlay = img.copy()
    cv2.rectangle(overlay, (tx - 20, ty - th - 20), (tx + tw + 20, ty + baseline + 20), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)

    cv2.imshow("observations", img)
    cv2.waitKey(2000)


def run_episode(robot, query, method, gt_positions, episode_id, cs):
    """
    Run a single object-search episode.

    Returns a dict of metrics.
    """
    # 1. Get VLMap candidates
    candidates = get_vlmap_candidates(robot, query)
    if len(candidates) < MIN_CANDIDATES:
        print(f"  [ep {episode_id}] Only {len(candidates)} candidates found — skipping.")
        return None

    # 2. Random start
    start_tf = find_random_navigable_start(robot)
    robot.set_agent_state(start_tf)
    robot._set_nav_curr_pose()
    start_pos = tuple(robot.curr_pos_on_map)

    # 3. Order candidates
    remaining = list(candidates)
    if method == "random":
        np.random.shuffle(remaining)
    # For "nearest", we re-sort before each navigation

    total_steps = 0
    candidates_visited = 0
    wrong_visits = 0
    path_length_m = 0.0
    success = False
    correct_first = False

    print(f"  [ep {episode_id}] start=({int(start_pos[0])},{int(start_pos[1])})  "
          f"candidates={len(remaining)}  gt_objects={len(gt_positions)}  method={method}")

    while remaining and total_steps < MAX_STEPS:
        # For nearest: always re-sort by distance from current position
        if method == "nearest":
            cur = tuple(robot.curr_pos_on_map)
            remaining.sort(key=lambda c: map_dist(cur, c, cs))

        target = remaining.pop(0)

        # Navigate
        robot.empty_recorded_actions()
        try:
            actions = robot.move_to(list(target))
        except Exception as e:
            print(f"    Navigation error: {e}")
            continue
        n_actions = len([a for a in actions if a != "stop"])
        total_steps += n_actions
        path_length_m += n_actions * robot.forward_dist
        candidates_visited += 1

        # Check success
        robot._set_nav_curr_pose()
        cur = tuple(robot.curr_pos_on_map)
        hit = is_near_gt(cur, gt_positions, cs)

        # Show result overlay on the robot's view
        _show_result_overlay(robot, query, hit)

        if hit:
            success = True
            correct_first = (candidates_visited == 1)
            break
        else:
            wrong_visits += 1

    # End-of-episode overlay if budget exhausted without success
    if not success and not remaining:
        _show_result_overlay(robot, query, False)

    result = {
        "episode": episode_id,
        "method": method,
        "query": query,
        "success": success,
        "steps": total_steps,
        "path_length_m": round(path_length_m, 2),
        "candidates_visited": candidates_visited,
        "correct_first_goal": correct_first,
        "wrong_visits": wrong_visits,
        "total_candidates": len(candidates),
        "total_gt_objects": len(gt_positions),
    }
    status = "SUCCESS" if success else "FAIL"
    print(f"    {status}  steps={total_steps}  visited={candidates_visited}  "
          f"wrong={wrong_visits}  path={path_length_m:.1f}m")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(
    version_base=None,
    config_path="../third_party/vlmaps/config",
    config_name="object_goal_navigation_cfg.yaml",
)
def main(config: DictConfig) -> None:
    # ── Parse extra CLI args ─────────────────────────────────────────────
    query    = config.get("query", "chair")
    method   = config.get("method", "random")
    episodes = config.get("episodes", 5)

    assert method in ("random", "nearest"), f"Unknown method: {method}"
    print(f"\n{'='*60}")
    print(f"  Baseline benchmark: M{'1' if method == 'random' else '2'} — {method}")
    print(f"  Query: {query}   Episodes: {episodes}   Scene: {config.scene_id}")
    print(f"{'='*60}\n")

    # ── Setup robot + scene ──────────────────────────────────────────────
    robot = HabitatLanguageRobot(config)
    robot.setup_scene(config.scene_id)
    robot.map.init_categories(mp3dcat.copy())
    print("VLMap loaded and categories initialised.\n")

    # Close obstacle-map windows from setup (keep only navigation views)
    for win in ["obs", "new obstacles_cropped"]:
        cv2.destroyWindow(win)
    cv2.waitKey(1)

    gs = robot.gs
    cs = robot.cs

    # ── Get ground-truth object positions ────────────────────────────────
    gt_positions = get_gt_object_positions(robot.sim, query, gs, cs)
    if not gt_positions:
        print(f"ERROR: No ground-truth objects matching '{query}' in this scene.")
        sys.exit(1)
    print(f"Found {len(gt_positions)} ground-truth '{query}' objects.\n")

    # ── Run episodes ─────────────────────────────────────────────────────
    results = []
    for ep in range(episodes):
        np.random.seed(ep * 42 + config.scene_id)
        r = run_episode(robot, query, method, gt_positions, ep, cs)
        if r is not None:
            results.append(r)

    # ── Aggregate metrics ────────────────────────────────────────────────
    if not results:
        print("\nNo valid episodes completed.")
        return

    successes  = [r for r in results if r["success"]]
    n_total    = len(results)
    n_success  = len(successes)
    sr         = n_success / n_total * 100
    avg_steps  = np.mean([r["steps"] for r in results])
    avg_path   = np.mean([r["path_length_m"] for r in results])
    avg_wrong  = np.mean([r["wrong_visits"] for r in results])
    cfg_rate   = (sum(r["correct_first_goal"] for r in results) / n_total * 100
                  if n_total else 0)

    summary = {
        "method": method,
        "method_name": f"M{'1_random' if method == 'random' else '2_nearest'}",
        "query": query,
        "scene_id": config.scene_id,
        "episodes_total": n_total,
        "episodes_success": n_success,
        "success_rate_pct": round(sr, 1),
        "correct_first_goal_pct": round(cfg_rate, 1),
        "avg_steps": round(float(avg_steps), 1),
        "avg_path_length_m": round(float(avg_path), 2),
        "avg_wrong_visits": round(float(avg_wrong), 2),
    }

    print(f"\n{'='*60}")
    print(f"  Results: {method.upper()}  |  {query}  |  scene {config.scene_id}")
    print(f"{'='*60}")
    print(f"  Success rate:       {sr:.1f}%  ({n_success}/{n_total})")
    print(f"  Correct first goal: {cfg_rate:.1f}%")
    print(f"  Avg steps:          {avg_steps:.1f}")
    print(f"  Avg path length:    {avg_path:.1f} m")
    print(f"  Avg wrong visits:   {avg_wrong:.1f}")
    print(f"{'='*60}\n")

    # ── Save results ─────────────────────────────────────────────────────
    scene_dir = robot.vlmaps_data_save_dirs[config.scene_id]
    out_dir = scene_dir / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{method}_{query}.json"
    payload = {"summary": summary, "episodes": results}
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
