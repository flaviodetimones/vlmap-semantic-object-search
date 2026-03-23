"""
analyze_vlmap_heatmap.py  (Step 6)
====================================
Query the VLMap for an object, extract connected-component regions from the
heatmap, and rank them by quality (area, density, mean score).

Reads:  <scene_dir>/vlmap/vlmaps.h5df
Writes: <scene_dir>/heatmap_regions_<query>.json
        <scene_dir>/heatmap_regions_<query>.png   (visualization)

Usage (inside Docker):
  python src/analyze_vlmap_heatmap.py data_paths=docker scene_id=0 +query=chair
"""

import json
import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.ndimage import binary_closing, gaussian_filter

from vlmaps.map.map import Map
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.index_utils import find_similar_category_id
from vlmaps.utils.visualize_utils import pool_3d_label_to_2d

# ── Constants ────────────────────────────────────────────────────────────────
MIN_AREA_PX     = 50    # minimum component area in grid cells
MIN_MEAN_SCORE  = 0.3   # minimum mean CLIP score to keep
MIN_DENSITY     = 0.4   # minimum bbox fill ratio


def extract_heatmap_2d(vlmap, query, categories):
    """
    Get a continuous 2D heatmap for *query* from the VLMap.

    Returns (heatmap_2d, mask_2d):
      heatmap_2d: (gs, gs) float32 in [0, 1]
      mask_2d:    (gs, gs) bool — binary mask after argmax
    """
    gs = vlmap.gs
    cat_id = find_similar_category_id(query, categories)
    print(f"  Matched query '{query}' → category '{categories[cat_id]}' (id={cat_id})")

    # Per-voxel scores for the target category
    voxel_scores = vlmap.scores_mat[:, cat_id].copy()

    # Binary mask: voxels where this category wins the argmax
    argmax_ids = np.argmax(vlmap.scores_mat, axis=1)
    mask_3d = (argmax_ids == cat_id)

    # Project to 2D
    mask_2d = pool_3d_label_to_2d(mask_3d, vlmap.grid_pos, gs)

    # Continuous heatmap: max score per (row, col) cell
    heatmap_2d = np.zeros((gs, gs), dtype=np.float32)
    for i in range(len(voxel_scores)):
        r, c = int(vlmap.grid_pos[i, 0]), int(vlmap.grid_pos[i, 1])
        if 0 <= r < gs and 0 <= c < gs:
            heatmap_2d[r, c] = max(heatmap_2d[r, c], voxel_scores[i])

    return heatmap_2d, mask_2d


def extract_regions(mask_2d, heatmap_2d, cs, obstacle_map=None):
    """
    Extract connected components from the binary mask and compute
    quality metrics for each region.

    Returns a list of region dicts sorted by quality (descending).
    """
    # Morphological cleanup
    cleaned = binary_closing(mask_2d, iterations=2).astype(np.uint8)
    cleaned = (gaussian_filter(cleaned.astype(np.float32), sigma=1.0) > 0.5).astype(np.uint8)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    regions = []
    for label_id in range(1, n_labels):  # skip background (0)
        area_px = int(stats[label_id, cv2.CC_STAT_AREA])
        if area_px < MIN_AREA_PX:
            continue

        # Bounding box
        bx = int(stats[label_id, cv2.CC_STAT_LEFT])
        by = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        bbox_area = bw * bh

        # Density = filled fraction of bounding box
        density = area_px / max(bbox_area, 1)
        if density < MIN_DENSITY:
            continue

        # Centroid
        cy, cx = centroids[label_id]  # note: cv2 returns (x, y)
        centroid_rc = (int(round(cx)), int(round(cy)))

        # Scores within this region
        region_mask = (labels == label_id)
        scores_in_region = heatmap_2d[region_mask]
        mean_score = float(np.mean(scores_in_region)) if len(scores_in_region) > 0 else 0.0
        peak_score = float(np.max(scores_in_region)) if len(scores_in_region) > 0 else 0.0

        if mean_score < MIN_MEAN_SCORE:
            continue

        # Quality score
        area_m2 = area_px * cs * cs
        quality = mean_score * np.log1p(area_px) * density

        regions.append({
            "centroid_row": centroid_rc[0],
            "centroid_col": centroid_rc[1],
            "area_px": area_px,
            "area_m2": round(area_m2, 3),
            "bbox": [by, bx, by + bh, bx + bw],  # [r_min, c_min, r_max, c_max]
            "density": round(density, 3),
            "mean_score": round(mean_score, 4),
            "peak_score": round(peak_score, 4),
            "quality": round(float(quality), 4),
        })

    # Sort by quality descending
    regions.sort(key=lambda r: -r["quality"])
    for i, r in enumerate(regions):
        r["rank"] = i + 1

    return regions


def visualize_regions(obstacle_map, heatmap_2d, regions, query, out_path, gs):
    """Save a visualization of the heatmap with numbered regions."""
    # Build base image
    obs_img = np.ones((gs, gs, 3), dtype=np.float32) * 0.85
    obs_img[~obstacle_map] = (0.1, 0.1, 0.1)

    # Overlay heatmap
    heat_u8 = (np.clip(heatmap_2d, 0, 1) * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET).astype(np.float32) / 255.0
    canvas = obs_img * 0.5 + heat_bgr * 0.5

    canvas_u8 = (canvas * 255).clip(0, 255).astype(np.uint8)

    # Draw regions
    for r in regions:
        row, col = r["centroid_row"], r["centroid_col"]
        cv2.circle(canvas_u8, (col, row), 6, (0, 255, 0), -1)
        cv2.circle(canvas_u8, (col, row), 8, (255, 255, 255), 1)
        label = f"#{r['rank']} q={r['quality']:.2f}"
        cv2.putText(canvas_u8, label, (col + 10, row + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Crop to building footprint
    nav_r, nav_c = np.where(obstacle_map)
    if len(nav_r) > 0:
        pad = 30
        rmin = max(0, nav_r.min() - pad)
        rmax = min(gs - 1, nav_r.max() + pad)
        cmin = max(0, nav_c.min() - pad)
        cmax = min(gs - 1, nav_c.max() + pad)
        canvas_u8 = canvas_u8[rmin:rmax+1, cmin:cmax+1]

    cv2.imwrite(str(out_path), canvas_u8)
    print(f"Visualization saved to {out_path}")

    cv2.imshow(f"Heatmap: {query}", canvas_u8)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


@hydra.main(
    version_base=None,
    config_path="../third_party/vlmaps/config",
    config_name="object_goal_navigation_cfg.yaml",
)
def main(config: DictConfig) -> None:
    query = config.get("query", "chair")
    gs = config.params.gs
    cs = config.params.cs

    # ── Resolve scene directory ──────────────────────────────────────────
    data_dir = Path(config.data_paths.vlmaps_data_dir)
    scene_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if config.scene_id >= len(scene_dirs):
        print(f"ERROR: scene_id={config.scene_id} but only {len(scene_dirs)} scenes.")
        sys.exit(1)
    scene_dir = scene_dirs[config.scene_id]

    print(f"\n{'='*60}")
    print(f"  Heatmap Region Analysis — scene {config.scene_id} ({scene_dir.name})")
    print(f"  Query: {query}")
    print(f"{'='*60}\n")

    # ── Load VLMap ───────────────────────────────────────────────────────
    vlmap = Map.create(config.map_config)
    vlmap.load_map(str(scene_dir))
    print("VLMap loaded.")

    vlmap._init_clip()
    categories = mp3dcat.copy()
    vlmap.init_categories(categories)
    print("Categories initialised.\n")

    # ── Load obstacle map ────────────────────────────────────────────────
    vlmap.generate_obstacle_map()
    obstacle_map = vlmap.obstacles_map

    # ── Extract heatmap and regions ──────────────────────────────────────
    print(f"Computing heatmap for '{query}'...")
    heatmap_2d, mask_2d = extract_heatmap_2d(vlmap, query, categories)

    mask_count = int(mask_2d.sum())
    print(f"  Mask cells: {mask_count}")

    regions = extract_regions(mask_2d, heatmap_2d, cs, obstacle_map)
    print(f"  Regions found: {len(regions)}")

    if not regions:
        print(f"\nNo valid regions found for '{query}'.")
        return

    # ── Print results ────────────────────────────────────────────────────
    print(f"\n{'Rank':<5} {'Area(m²)':<10} {'Score':<8} {'Density':<9} {'Quality':<9} {'Centroid'}")
    print("-" * 60)
    for r in regions:
        print(f"  {r['rank']:<3} {r['area_m2']:<10} {r['mean_score']:<8} "
              f"{r['density']:<9} {r['quality']:<9} ({r['centroid_row']}, {r['centroid_col']})")

    # ── Save JSON ────────────────────────────────────────────────────────
    out_json = scene_dir / f"heatmap_regions_{query}.json"
    payload = {
        "query": query,
        "scene": scene_dir.name,
        "total_mask_cells": mask_count,
        "regions": regions,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {out_json}")

    # ── Visualization ────────────────────────────────────────────────────
    out_png = scene_dir / f"heatmap_regions_{query}.png"
    visualize_regions(obstacle_map, heatmap_2d, regions, query, out_png, gs)


if __name__ == "__main__":
    main()
