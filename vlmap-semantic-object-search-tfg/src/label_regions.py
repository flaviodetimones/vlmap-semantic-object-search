"""
label_regions.py
================
Get MP3D ground-truth room regions from Habitat-Sim, place the centroid
of each region on the navigation grid, and display the result.

Nothing more — no voting, no propagation, no smoothing.

Usage
-----
    python src/label_regions.py --scene_id 9 --data_dir /workspace/data --mp3d_dir /workspace/data/mp3d
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2


MP3D_REGION_LABELS = {
    "a": "bathroom",    "b": "bedroom",     "c": "closet",
    "d": "dining_room", "e": "entryway",    "f": "family_room",
    "g": "garage",      "h": "hallway",     "i": "library",
    "j": "laundry_room","k": "kitchen",     "l": "living_room",
    "m": "meeting_room","n": "lounge",      "o": "office",
    "p": "porch",       "r": "rec_room",    "s": "stairs",
    "t": "toilet",      "v": "tv_room",     "w": "workout_room",
    "x": "outdoor",     "y": "balcony",     "z": "other",
    "B": "bar",         "C": "classroom",   "D": "dining_booth",
    "S": "spa",         "Z": "junk",        "-": "unlabeled",
}

_SKIP = {"junk", "outdoor", "other", "unlabeled", ""}

_COLORS = plt.get_cmap("tab10").colors


def find_vlmap_dir(scene_id, data_dir):
    dataset_dir = Path(data_dir) / "vlmaps_dataset"
    dirs = sorted(d for d in dataset_dir.iterdir() if d.is_dir())
    scene_dir = None
    try:
        scene_dir = dirs[int(scene_id)]
    except (ValueError, IndexError):
        for d in dirs:
            if d.name == scene_id or d.name.startswith(scene_id + "_"):
                scene_dir = d
                break
    if scene_dir is None:
        print(f"Scene '{scene_id}' not found in {dataset_dir}", file=sys.stderr)
        sys.exit(1)
    for sub in sorted(scene_dir.iterdir()):
        if sub.is_dir() and sub.name.startswith("vlmap"):
            return sub, scene_dir
    return scene_dir, scene_dir


def load_grid(scene_id, data_dir):
    vlmap_dir, scene_dir = find_vlmap_dir(scene_id, data_dir)

    # obstacle_map
    for p in [vlmap_dir / "obstacle_map.npy", scene_dir / "obstacle_map.npy"]:
        if p.exists():
            obstacle_map = np.load(str(p)).astype(bool)
            print(f"Obstacle map : {p}  shape={obstacle_map.shape}")
            break
    else:
        print("obstacle_map.npy not found", file=sys.stderr)
        sys.exit(1)

    # grid_pos
    for p in [vlmap_dir / "grid_pos.npy", scene_dir / "grid_pos.npy"]:
        if p.exists():
            grid_pos = np.load(str(p))
            print(f"Grid pos     : {p}")
            return obstacle_map, grid_pos, vlmap_dir

    # compute from poses.txt
    for p in [scene_dir / "poses.txt", vlmap_dir / "poses.txt"]:
        if p.exists():
            print(f"Grid pos     : computing from {p}")
            grid_pos = compute_grid_pos(obstacle_map, p)
            return obstacle_map, grid_pos, vlmap_dir

    print("grid_pos.npy and poses.txt not found", file=sys.stderr)
    sys.exit(1)


def compute_grid_pos(obstacle_map, poses_path, cell_size=0.05):
    from scipy.spatial.transform import Rotation
    gs = obstacle_map.shape[0]
    poses = np.loadtxt(str(poses_path))
    row = poses if poses.ndim == 1 else poses[0]
    init_tf = np.eye(4)
    init_tf[:3, 3] = row[:3]
    init_tf[:3, :3] = Rotation.from_quat(row[3:]).as_matrix()
    BASE_ROT = np.array([[0,0,-1],[-1,0,0],[0,1,0]], dtype=np.float64)
    inv_base = np.eye(4)
    inv_base[:3, :3] = BASE_ROT.T
    b2w = init_tf @ inv_base
    ri, ci = np.arange(gs, dtype=np.float64), np.arange(gs, dtype=np.float64)
    R, C = np.meshgrid(ri, ci, indexing="ij")
    bx = (gs/2.0 - R) * cell_size
    by = (gs/2.0 - C) * cell_size
    pts = np.stack([bx.ravel(), by.ravel(), np.zeros(gs*gs), np.ones(gs*gs)])
    pw = b2w @ pts
    return np.stack([pw[0].reshape(gs,gs), pw[1].reshape(gs,gs), pw[2].reshape(gs,gs)], axis=-1).astype(np.float32)


def get_regions(scene_name, mp3d_dir):
    try:
        import habitat_sim
    except ImportError:
        print("habitat_sim not installed", file=sys.stderr)
        sys.exit(1)

    scene_path = f"{mp3d_dir}/{scene_name}/{scene_name}.glb"
    if not Path(scene_path).exists():
        print(f"Scene not found: {scene_path}", file=sys.stderr)
        sys.exit(1)

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    backend_cfg.enable_physics = False
    backend_cfg.load_semantic_mesh = True
    cfg = habitat_sim.Configuration(backend_cfg, [habitat_sim.agent.AgentConfiguration()])
    sim = habitat_sim.Simulator(cfg)

    regions = []
    for region in sim.semantic_scene.regions:
        raw = region.category.name().strip()
        # Map single-letter codes; for full names take only the first alias
        name = MP3D_REGION_LABELS.get(raw, raw).lower()
        name = name.split("/")[0].strip()  # drop e.g. "terrace/deck" from "porch/terrace/deck"
        if name in _SKIP:
            continue
        center = np.array(region.aabb.center)  # SemanticScene coords
        # SemanticScene X = Habitat Z, SemanticScene Z = Habitat X
        hab_x = float(center[2])
        hab_y = float(center[1])
        hab_z = float(center[0])
        regions.append({"name": name, "hab_x": hab_x, "hab_y": hab_y, "hab_z": hab_z})

    sim.close()
    print(f"Regions found: {len(regions)}")
    return regions


def world_to_nearest_grid(hab_x, hab_z, grid_pos, obstacle_map):
    """Find the navigable grid cell closest (XZ) to a Habitat world position."""
    nav_rows, nav_cols = np.where(obstacle_map)
    nav_world = grid_pos[nav_rows, nav_cols]  # (N, 3)
    diff = nav_world[:, [0, 2]] - np.array([hab_x, hab_z])
    dists = (diff ** 2).sum(axis=1)
    idx = dists.argmin()
    return int(nav_rows[idx]), int(nav_cols[idx])


def main():
    parser = argparse.ArgumentParser(description="Label MP3D region centroids on the nav grid")
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--data_dir", default="/workspace/data")
    parser.add_argument("--mp3d_dir", default="/workspace/data/mp3d")
    parser.add_argument("--out", default="", help="Output PNG path (auto if empty)")
    args = parser.parse_args()

    # ── Load grid data ───────────────────────────────────────────────────
    obstacle_map, grid_pos, vlmap_dir = load_grid(args.scene_id, args.data_dir)
    H, W = obstacle_map.shape

    # ── Resolve scene name ───────────────────────────────────────────────
    scene_dir = vlmap_dir.parent if vlmap_dir.name.startswith("vlmap") else vlmap_dir
    scene_name = scene_dir.name.split("_")[0]
    print(f"Scene name   : {scene_name}")

    # ── Get Habitat regions ──────────────────────────────────────────────
    regions = get_regions(scene_name, args.mp3d_dir)

    # ── Filter to ground floor (match robot's Y level) ───────────────────
    nav_rows_all, nav_cols_all = np.where(obstacle_map)
    nav_ys = grid_pos[nav_rows_all, nav_cols_all][:, 1]
    floor_y = float(np.median(nav_ys))
    print(f"Floor Y (median): {floor_y:.2f}")
    print(f"All region Y values:")
    for r in regions:
        print(f"  {r['name']:25s}  Y={r['hab_y']:.2f}  diff={abs(r['hab_y']-floor_y):.2f}")

    floor_regions = [r for r in regions if abs(r["hab_y"] - floor_y) < 2.0]
    print(f"Floor regions: {len(floor_regions)} (of {len(regions)} total)")

    # ── Map each centroid to nearest navigable grid cell ─────────────────
    labeled = []
    for r in floor_regions:
        row, col = world_to_nearest_grid(r["hab_x"], r["hab_z"], grid_pos, obstacle_map)
        labeled.append({"name": r["name"], "row": row, "col": col})
        print(f"  {r['name']:20s}  -> grid ({row}, {col})")

    # ── Crop to building footprint ───────────────────────────────────────
    nav_r, nav_c = np.where(obstacle_map)
    pad = 30
    rmin = max(0, nav_r.min() - pad)
    rmax = min(H - 1, nav_r.max() + pad)
    cmin = max(0, nav_c.min() - pad)
    cmax = min(W - 1, nav_c.max() + pad)

    obs_crop = obstacle_map[rmin:rmax+1, cmin:cmax+1]

    # ── Build image ──────────────────────────────────────────────────────
    img = np.ones((*obs_crop.shape, 3), dtype=np.float32) * 0.85  # free = light gray
    img[~obs_crop] = (0.1, 0.1, 0.1)                               # obstacle = dark

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 12), dpi=120)
    ax.imshow(img, origin="upper", interpolation="nearest")
    ax.axis("off")

    unique_names = sorted(set(r["name"] for r in labeled))
    name_to_color = {n: _COLORS[i % len(_COLORS)] for i, n in enumerate(unique_names)}

    for entry in labeled:
        r = entry["row"] - rmin
        c = entry["col"] - cmin
        color = name_to_color[entry["name"]]

        # Dot at centroid
        ax.plot(c, r, "o", color=color, markersize=10, markeredgecolor="white",
                markeredgewidth=1.5, zorder=3)

        # Label
        ax.text(c + 5, r, entry["name"], fontsize=8, color="white",
                fontweight="bold", va="center",
                bbox=dict(facecolor=color, alpha=0.8, pad=2, linewidth=0),
                zorder=4)

    # Legend
    handles = [mpatches.Patch(facecolor=name_to_color[n], label=n) for n in unique_names]
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              framealpha=0.9, borderpad=0.8)

    ax.set_title(f"MP3D region centroids — scene {args.scene_id} ({scene_name})", fontsize=10)
    plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = Path(args.out) if args.out else vlmap_dir / "region_centroids.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved -> {out_path.resolve()}")
    print(f"On host: ~/tfg/data/vlmaps_dataset/{scene_dir.name}/{vlmap_dir.name}/region_centroids.png")

    # Open with OpenCV (press any key to close)
    img_bgr = cv2.imread(str(out_path))
    if img_bgr is not None:
        # Resize to fit screen if too large
        h, w = img_bgr.shape[:2]
        max_side = 900
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        cv2.imshow("Region centroids (press any key to close)", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
