"""Headless multi-class VLMap index validation.

This module mirrors the native VLMaps indexing path:

1. load a saved VLMap,
2. initialize a complete category vocabulary plus the implicit "other" class,
3. assign each voxel to the argmax category,
4. export per-category 2D masks/heatmaps and validation figures.

It intentionally avoids Open3D windows, cv2.imshow and input prompts so it can
run inside the Docker containers and CI-like terminals.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402

from .cli import _load_vlmap  # noqa: WPS436 - reuse the existing loader contract
from .heatmap_dumper import dump_heatmap_npy


@dataclass(frozen=True)
class CategoryResult:
    """Projected outputs and metrics for one requested category."""

    name: str
    safe_name: str
    category_id: int
    mask_2d: np.ndarray
    heatmap_2d: np.ndarray
    centroid_rc: list[float] | None
    native_centers_rc: list[list[float]]
    area_cells: int
    area_ratio: float


def _safe_name(name: str) -> str:
    safe = name.strip().replace(" ", "_").replace("/", "_")
    if not safe:
        raise ValueError("category names must not be empty")
    return safe


def _unique_preserve_order(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _full_category_list(requested: Sequence[str], dataset_type: str) -> list[str]:
    from vlmaps.utils.matterport3d_categories import get_categories  # noqa: WPS433

    base = get_categories(dataset_type)
    # Keep exact requested labels in the initialized vocabulary so
    # VLMap.index_map() never falls back to the OpenAI-based similarity helper.
    return _unique_preserve_order([*base, *requested])


def _pool_label_ids_to_2d(label_ids: np.ndarray, grid_pos: np.ndarray, gs: int) -> np.ndarray:
    """Project per-voxel integer labels to a top-down grid using highest voxel."""
    if label_ids.shape[0] != grid_pos.shape[0]:
        raise ValueError("label_ids and grid_pos must have the same length")
    labels_2d = -1 * np.ones((gs, gs), dtype=np.int16)
    height = -100 * np.ones((gs, gs), dtype=np.int32)
    for idx, pos in enumerate(grid_pos):
        row, col, h = pos
        if h > height[row, col]:
            labels_2d[row, col] = int(label_ids[idx])
            height[row, col] = int(h)
    return labels_2d


def _pool_bool_mask_to_2d(mask_3d: np.ndarray, grid_pos: np.ndarray, gs: int) -> np.ndarray:
    """Project a 3D boolean mask to a 2D boolean footprint."""
    if mask_3d.shape[0] != grid_pos.shape[0]:
        raise ValueError("mask_3d and grid_pos must have the same length")
    mask_2d = np.zeros((gs, gs), dtype=bool)
    for idx, pos in enumerate(grid_pos):
        row, col, _h = pos
        if mask_3d[idx]:
            mask_2d[row, col] = True
    return mask_2d


def _pool_rgb_to_2d(rgb: np.ndarray, grid_pos: np.ndarray, gs: int) -> np.ndarray:
    """Project per-voxel RGB to a top-down grid using highest voxel."""
    rgb_2d = np.zeros((gs, gs, 3), dtype=np.uint8)
    height = -100 * np.ones((gs, gs), dtype=np.int32)
    for idx, pos in enumerate(grid_pos):
        row, col, h = pos
        if h > height[row, col]:
            rgb_2d[row, col] = rgb[idx]
            height[row, col] = int(h)
    return rgb_2d


def _heatmap_from_mask(mask_2d: np.ndarray, cell_size: float, decay_rate: float) -> np.ndarray:
    """Return a smooth heatmap derived from an argmax mask."""
    if not np.any(mask_2d):
        return np.zeros(mask_2d.shape, dtype=np.float32)
    dists = distance_transform_edt(mask_2d == 0) / cell_size
    heatmap = np.clip(1.0 - (dists * decay_rate), 0.0, 1.0)
    return heatmap.astype(np.float32)


def _centroid(mask_2d: np.ndarray) -> list[float] | None:
    pts = np.argwhere(mask_2d)
    if pts.size == 0:
        return None
    row, col = pts.mean(axis=0)
    return [float(row), float(col)]


def _native_centers(vlmap, category: str) -> list[list[float]]:
    """Get native VLMaps centers via get_pos(), returning [] if it cannot."""
    try:
        contours, centers, _bbox_list = vlmap.get_pos(category)
    except Exception as exc:  # noqa: BLE001 - validation should continue
        print(f"[index_validation] get_pos({category!r}) skipped: {exc}")
        return []
    if not contours:
        return []
    return [[float(center[0]), float(center[1])] for center in centers]


def _occupied_bbox(occupied_2d: np.ndarray, masks: Sequence[np.ndarray], pad: int = 20) -> tuple[slice, slice]:
    union = occupied_2d.copy()
    for mask in masks:
        union |= mask
    pts = np.argwhere(union)
    if pts.size == 0:
        return slice(0, occupied_2d.shape[0]), slice(0, occupied_2d.shape[1])
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return (
        slice(max(int(r0) - pad, 0), min(int(r1) + pad + 1, occupied_2d.shape[0])),
        slice(max(int(c0) - pad, 0), min(int(c1) + pad + 1, occupied_2d.shape[1])),
    )


def _display_rgb(rgb_2d: np.ndarray, occupied_2d: np.ndarray) -> np.ndarray:
    """Build a visible top-down RGB image for PNG overlays."""
    rgb = rgb_2d.astype(np.float32)
    visible = occupied_2d & np.any(rgb_2d > 0, axis=2)
    if np.any(visible):
        vals = rgb[visible]
        lo = float(np.percentile(vals, 2))
        hi = float(np.percentile(vals, 98))
        if hi > lo:
            rgb = np.clip((rgb - lo) / (hi - lo), 0, 1) * 255.0
    canvas = np.full_like(rgb, 18.0)
    canvas[occupied_2d] = np.maximum(rgb[occupied_2d], 35.0)
    return canvas.astype(np.uint8)


def _overlay_png(
    rgb_2d: np.ndarray,
    occupied_2d: np.ndarray,
    result: CategoryResult,
    out_path: Path,
    crop: tuple[slice, slice],
) -> None:
    base = _display_rgb(rgb_2d, occupied_2d)
    rows, cols = crop
    mask = result.mask_2d[rows, cols]
    image = base[rows, cols].astype(np.float32) / 255.0
    overlay = image.copy()
    overlay[mask] = overlay[mask] * 0.30 + np.array([0.0, 0.95, 0.20]) * 0.70

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.imshow(overlay)
    title = f"{result.name}: {result.area_cells} cells ({result.area_ratio:.2%})"
    if result.centroid_rc is not None:
        row, col = result.centroid_rc
        ax.scatter([col - cols.start], [row - rows.start], c="cyan", s=36, marker="x", linewidths=1.8)
        title += f" | centroid=({row:.1f},{col:.1f})"
    for center in result.native_centers_rc:
        ax.scatter([center[1] - cols.start], [center[0] - rows.start], c="yellow", s=26, marker="o")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path)
    plt.close(fig)


def _summary_png(
    rgb_2d: np.ndarray,
    occupied_2d: np.ndarray,
    labels_2d: np.ndarray,
    categories: Sequence[str],
    results: Sequence[CategoryResult],
    out_path: Path,
    crop: tuple[slice, slice],
) -> None:
    base = _display_rgb(rgb_2d, occupied_2d)
    rows, cols = crop
    image = base[rows, cols].astype(np.float32) / 255.0
    summary = image * 0.40

    cmap = plt.get_cmap("tab10")
    legend_handles = []
    for idx, result in enumerate(results):
        color = np.array(cmap(idx % 10)[:3])
        cat_cells = labels_2d[rows, cols] == result.category_id
        summary[cat_cells] = summary[cat_cells] * 0.25 + color * 0.75
        handle = plt.Line2D([0], [0], marker="s", color="w", label=result.name, markerfacecolor=color, markersize=8)
        legend_handles.append(handle)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(np.clip(summary, 0, 1))
    for idx, result in enumerate(results):
        color = np.array(cmap(idx % 10)[:3])
        if result.centroid_rc is not None:
            row, col = result.centroid_rc
            ax.scatter([col - cols.start], [row - rows.start], c=[color], s=58, marker="x", linewidths=2.0)
            ax.text(
                col - cols.start + 3,
                row - rows.start + 3,
                result.name,
                color=color,
                fontsize=7,
                weight="bold",
            )
    ax.set_title(f"Dominant categories over {len(categories)} classes + other")
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.8)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path)
    plt.close(fig)


def _maybe_rotate(arr: np.ndarray, rotate_to_ros: bool) -> np.ndarray:
    if not rotate_to_ros:
        return arr
    if arr.ndim == 2:
        return np.flipud(arr.T).copy()
    if arr.ndim == 3:
        return np.flipud(np.transpose(arr, (1, 0, 2))).copy()
    raise ValueError(f"unsupported array rank for rotation: {arr.ndim}")


def _pair_metrics(results: Sequence[CategoryResult], cell_size: float) -> list[dict[str, float | str | None]]:
    rows: list[dict[str, float | str | None]] = []
    for i, left in enumerate(results):
        for right in results[i + 1 :]:
            inter = int(np.logical_and(left.mask_2d, right.mask_2d).sum())
            union = int(np.logical_or(left.mask_2d, right.mask_2d).sum())
            min_area = min(left.area_cells, right.area_cells)
            iou = float(inter / union) if union else 0.0
            overlap_min = float(inter / min_area) if min_area else 0.0
            dist_cells = None
            dist_m = None
            if left.centroid_rc is not None and right.centroid_rc is not None:
                dist_cells = float(math.dist(left.centroid_rc, right.centroid_rc))
                dist_m = dist_cells * cell_size
            rows.append(
                {
                    "left": left.name,
                    "right": right.name,
                    "intersection_cells": inter,
                    "union_cells": union,
                    "iou": iou,
                    "overlap_min": overlap_min,
                    "centroid_distance_cells": dist_cells,
                    "centroid_distance_m": dist_m,
                }
            )
    return rows


def validate_index(
    data_dir: str,
    output_dir: str,
    categories: Sequence[str],
    *,
    dataset_type: str = "mp3d",
    decay_rate: float = 0.01,
    rotate_to_ros: bool = False,
) -> dict:
    """Run the multi-class index validation and write artifacts."""
    requested = _unique_preserve_order(categories)
    if not requested:
        raise ValueError("at least one category is required")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[index_validation] loading VLMap from {data_dir} ...")
    vlmap = _load_vlmap(data_dir)
    grid_pos = np.asarray(vlmap.grid_pos)
    grid_rgb = np.asarray(vlmap.grid_rgb)
    gs = int(vlmap.gs)
    cell_size = float(vlmap.cs)

    full_categories = _full_category_list(requested, dataset_type)
    print(
        f"[index_validation] initializing {len(full_categories)} categories "
        f"from {dataset_type} plus requested labels ..."
    )
    scores_mat = vlmap.init_categories(full_categories)
    max_ids = np.argmax(scores_mat, axis=1).astype(np.int16)

    # get_pos() expects the obstacle crop bounds to exist.
    try:
        vlmap.generate_obstacle_map()
    except Exception as exc:  # noqa: BLE001 - not fatal for mask validation
        print(f"[index_validation] obstacle crop skipped: {exc}")

    occupied_2d = _pool_bool_mask_to_2d(np.ones(grid_pos.shape[0], dtype=bool), grid_pos, gs)
    rgb_2d = _pool_rgb_to_2d(grid_rgb, grid_pos, gs)
    labels_2d = _pool_label_ids_to_2d(max_ids, grid_pos, gs)

    results: list[CategoryResult] = []
    for cat in requested:
        safe = _safe_name(cat)
        cat_id = full_categories.index(cat)
        mask_3d = vlmap.index_map(cat, with_init_cat=True)
        mask_2d = _pool_bool_mask_to_2d(mask_3d, grid_pos, gs)
        heatmap_2d = _heatmap_from_mask(mask_2d, cell_size, decay_rate)
        area = int(mask_2d.sum())
        area_ratio = float(area / max(int(occupied_2d.sum()), 1))
        result = CategoryResult(
            name=cat,
            safe_name=safe,
            category_id=cat_id,
            mask_2d=mask_2d,
            heatmap_2d=heatmap_2d,
            centroid_rc=_centroid(mask_2d),
            native_centers_rc=_native_centers(vlmap, cat),
            area_cells=area,
            area_ratio=area_ratio,
        )
        results.append(result)

        heatmap_path = dump_heatmap_npy(
            _maybe_rotate(heatmap_2d, rotate_to_ros),
            str(out_dir),
            cat,
            rotate_to_ros=False,
        )
        np.save(out_dir / f"{safe}_mask.npy", _maybe_rotate(mask_2d.astype(np.uint8), rotate_to_ros))
        print(
            f"[index_validation] {cat}: area={area} cells "
            f"({area_ratio:.2%}) centroid={result.centroid_rc} -> {heatmap_path}"
        )

    crop = _occupied_bbox(occupied_2d, [result.mask_2d for result in results])
    for result in results:
        _overlay_png(
            rgb_2d,
            occupied_2d,
            result,
            out_dir / f"{result.safe_name}_overlay.png",
            crop,
        )

    np.save(out_dir / "dominant_labels.npy", _maybe_rotate(labels_2d, rotate_to_ros))
    _summary_png(
        rgb_2d,
        occupied_2d,
        labels_2d,
        full_categories,
        results,
        out_dir / "summary_dominant.png",
        crop,
    )

    pairs = _pair_metrics(results, cell_size)
    metrics = {
        "data_dir": data_dir,
        "output_dir": str(out_dir),
        "dataset_type": dataset_type,
        "grid_size": gs,
        "cell_size_m": cell_size,
        "occupied_cells": int(occupied_2d.sum()),
        "initialized_categories": full_categories,
        "requested_categories": requested,
        "rotate_to_ros": rotate_to_ros,
        "categories": [
            {
                "name": result.name,
                "category_id": result.category_id,
                "area_cells": result.area_cells,
                "area_ratio": result.area_ratio,
                "centroid_rc": result.centroid_rc,
                "native_centers_rc": result.native_centers_rc,
                "heatmap_npy": f"{result.safe_name}.npy",
                "mask_npy": f"{result.safe_name}_mask.npy",
                "overlay_png": f"{result.safe_name}_overlay.png",
            }
            for result in results
        ],
        "pairs": pairs,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (out_dir / "categories.json").open("w", encoding="utf-8") as f:
        json.dump({"categories": full_categories, "other_id": len(full_categories)}, f, indent=2)

    mean_iou = float(np.mean([row["iou"] for row in pairs])) if pairs else 0.0
    max_iou = float(max([row["iou"] for row in pairs], default=0.0))
    print(f"[index_validation] wrote {out_dir}")
    print(f"[index_validation] pairwise IoU: mean={mean_iou:.4f} max={max_iou:.4f}")
    return metrics


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and export native VLMap multi-class indexes.")
    parser.add_argument("--data-dir", required=True, help="Directory passed to VLMap.load_map.")
    parser.add_argument("--output-dir", required=True, help="Directory for .npy/.png/.json artifacts.")
    parser.add_argument("--categories", nargs="+", required=True, help="Landmark categories to validate/export.")
    parser.add_argument(
        "--dataset-type",
        default="mp3d",
        choices=["mp3d", "hssd"],
        help="Base VLMaps category vocabulary. Missing requested labels are appended exactly.",
    )
    parser.add_argument("--decay-rate", type=float, default=0.01, help="2D heatmap decay rate from argmax masks.")
    parser.add_argument(
        "--rotate-to-ros",
        action="store_true",
        help="Rotate saved arrays to the ROS occupancy convention. PNGs keep VLMap grid orientation.",
    )
    args = parser.parse_args(argv)

    validate_index(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        categories=args.categories,
        dataset_type=args.dataset_type,
        decay_rate=args.decay_rate,
        rotate_to_ros=args.rotate_to_ros,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
