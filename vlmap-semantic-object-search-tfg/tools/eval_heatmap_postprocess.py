#!/usr/bin/env python3
"""
Evaluate raw vs postprocess heatmap quality on a batch of queries.

For every query in a JSONL batch (format produced by the eval batch generator),
this script computes both:
  - the RAW heatmap (CLIP scores + 2D projection + distance decay), without postprocess.
  - the CLEAN heatmap (RAW → postprocess_heatmap), the version the navigator actually uses.

It then evaluates each version against ground truth derived from the room provider
(HSSD semantic_config.json polygons rasterised to the VLMap grid).

Outputs (under --out):
  - per_query.csv      one row per query with raw_* and clean_* metrics
  - summary.md         aggregated metrics + win/loss table
  - img/{id}.png       optional side-by-side overlays (--save-images)

Usage (inside docker):
  python tools/eval_heatmap_postprocess.py \\
      --scene-id 0 \\
      --queries tools/eval_queries/102344193_0.jsonl \\
      --out tools/eval_results/102344193_0/

Smoke test without a JSONL (uses inline queries, derives ground truth from room
priors when possible):
  python tools/eval_heatmap_postprocess.py \\
      --scene-id 0 \\
      --queries-inline "bed,microwave,kitchen" \\
      --out tools/eval_results/smoke/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
from hydra import compose, initialize_config_dir
from scipy.ndimage import distance_transform_edt


# ── Path setup so we can import the vendored vlmaps + baseline app ────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
VLMAPS_ROOT = REPO_ROOT / "third_party" / "vlmaps"
if str(VLMAPS_ROOT) not in sys.path:
    sys.path.insert(0, str(VLMAPS_ROOT))

CONFIG_DIR = str(VLMAPS_ROOT / "config")

from vlmaps.robot.habitat_lang_robot import HabitatLanguageRobot                       # noqa: E402
from vlmaps.utils.matterport3d_categories import get_categories                        # noqa: E402
from vlmaps.utils.index_utils import find_similar_category_id                          # noqa: E402

# Baseline owns compute_heatmap / postprocess_heatmap. We import postprocess as-is
# and re-implement the *pre-postprocess* portion locally so we can dump the RAW
# heatmap without modifying baseline code (which is shared with Codex's executor).
from application.interactive_object_nav import postprocess_heatmap                     # noqa: E402


# ── Raw heatmap (mirrors compute_heatmap up to, but not including, postprocess)
def compute_raw_heatmap(robot, category: str, score_thresh: float = 0.3) -> np.ndarray:
    """Return the heatmap *before* postprocess_heatmap is applied.

    Mirrors application.interactive_object_nav.compute_heatmap exactly except
    for the final postprocess step, so we can compare raw vs clean outputs.
    """
    cat_id = find_similar_category_id(category, robot.map.categories)
    scores = robot.map.scores_mat[:, cat_id]
    max_ids = np.argmax(robot.map.scores_mat, axis=1)

    valid = (max_ids == cat_id) & (scores > score_thresh)
    scores_filtered = np.where(valid, scores, 0.0)

    gs = robot.map.gs
    heat_2d = np.zeros((gs, gs), dtype=np.float32)
    for i, pos in enumerate(robot.map.grid_pos):
        row, col, _ = pos
        if scores_filtered[i] > heat_2d[row, col]:
            heat_2d[row, col] = scores_filtered[i]

    # Distance decay (same as baseline)
    mask = heat_2d > 0
    if mask.any():
        dist = distance_transform_edt(~mask)
        heat_2d = np.where(
            mask, heat_2d, np.clip(1.0 - dist * 0.3, 0, 1).astype(np.float32)
        )
        heat_2d[heat_2d < 0.15] = 0

    return heat_2d


# ── Query model ───────────────────────────────────────────────────────────────
@dataclass
class Query:
    id: str
    query: str
    target_label: str
    query_type: str = "object"            # object | room | room_object | compound
    expected_rooms: List[str] = field(default_factory=list)
    expected_room_polygons: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    scene_id: Optional[int] = None
    scene_name: Optional[str] = None


def load_queries_jsonl(path: Path) -> List[Query]:
    out: List[Query] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {path.name}:{i} not valid JSON ({e}); skipped")
                continue
            out.append(Query(
                id=str(d.get("id") or f"q{i:03d}"),
                query=str(d["query"]),
                target_label=str(d.get("target_label") or d["query"]),
                query_type=str(d.get("query_type") or "object"),
                expected_rooms=list(d.get("expected_rooms") or []),
                expected_room_polygons=list(d.get("expected_room_polygons") or []),
                tags=list(d.get("tags") or []),
                scene_id=d.get("scene_id"),
                scene_name=d.get("scene_name"),
            ))
    return out


def make_inline_queries(spec: str) -> List[Query]:
    """Smoke-test queries: comma-separated 'kitchen,bed,microwave'."""
    out = []
    for i, raw in enumerate([s.strip() for s in spec.split(",") if s.strip()], 1):
        out.append(Query(
            id=f"smoke{i:02d}",
            query=raw,
            target_label=raw,
            query_type="object",
            expected_rooms=[],
            expected_room_polygons=[],
            tags=["smoke"],
        ))
    return out


# ── Ground-truth helpers using room_provider regions ──────────────────────────
def regions_by_label(room_provider) -> Dict[str, List[Dict[str, Any]]]:
    """Group room_provider._regions by label (lowercased), preserving order."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for reg in getattr(room_provider, "_regions", []) or []:
        key = (reg.get("label") or reg.get("name") or "").lower()
        out.setdefault(key, []).append(reg)
    return out


def expected_mask(
    q: Query,
    region_grid: np.ndarray,
    label_index: Dict[str, List[Dict[str, Any]]],
) -> np.ndarray:
    """Boolean mask on the VLMap grid covering expected room polygon(s).

    Resolution order:
      1. expected_room_polygons -> (label, instance_idx) -> region.id.
      2. expected_rooms (label only) -> all regions with that label.
      3. Empty mask if neither resolves.
    """
    mask = np.zeros_like(region_grid, dtype=bool)
    matched = False

    if q.expected_room_polygons:
        for spec in q.expected_room_polygons:
            label = str(spec.get("label", "")).lower()
            idx = spec.get("instance_idx")
            regs = label_index.get(label, [])
            if not regs:
                continue
            chosen = regs if idx is None else (
                [regs[idx]] if 0 <= int(idx) < len(regs) else regs
            )
            for reg in chosen:
                mask |= (region_grid == int(reg["id"]))
                matched = True

    if not matched and q.expected_rooms:
        for label in q.expected_rooms:
            for reg in label_index.get(label.lower(), []):
                mask |= (region_grid == int(reg["id"]))
                matched = True

    return mask


# ── Per-version metrics ───────────────────────────────────────────────────────
def topk_cells(heat: np.ndarray, k: int) -> List[Tuple[int, int]]:
    """Top-K (row, col) cells by activation. Returns [] if all zero."""
    if heat.max() <= 0:
        return []
    flat = heat.flatten()
    if k >= flat.size:
        idx = np.argsort(-flat)
    else:
        # argpartition then sort the partition
        part = np.argpartition(-flat, k - 1)[:k]
        idx = part[np.argsort(-flat[part])]
    H, W = heat.shape
    out = []
    for f in idx:
        if heat.flat[f] <= 0:
            break
        out.append((int(f // W), int(f % W)))
    return out


def topmass_mask(heat: np.ndarray, frac: float = 0.5) -> np.ndarray:
    """Cells whose activation >= frac * heat.max(). Empty if heat is zero."""
    m = float(heat.max())
    if m <= 0:
        return np.zeros_like(heat, dtype=bool)
    return heat >= (frac * m)


def evaluate_heatmap(
    heat: np.ndarray,
    expected: np.ndarray,
    region_grid: np.ndarray,
    n_components_hint: Optional[int] = None,
) -> Dict[str, float]:
    """Compute metrics for a single heatmap version vs ground-truth mask."""
    out: Dict[str, float] = {}
    out["max_value"] = float(heat.max())
    out["total_mass"] = float(heat.sum())
    nz = (heat > 0)
    out["nonzero_cells"] = int(nz.sum())

    has_gt = bool(expected.any())
    has_signal = bool(nz.any())

    # Hit@K: any of the top-K cells inside expected mask
    for k in (1, 3, 5):
        cells = topk_cells(heat, k)
        if not has_gt or not cells:
            out[f"hit@{k}"] = float("nan")
        else:
            out[f"hit@{k}"] = float(any(expected[r, c] for r, c in cells))

    # Mass distribution
    if has_signal:
        if has_gt:
            in_mass = float(heat[expected].sum())
            out["mass_in_expected"] = in_mass
            out["mass_in_expected_ratio"] = in_mass / max(out["total_mass"], 1e-9)
        else:
            out["mass_in_expected"] = float("nan")
            out["mass_in_expected_ratio"] = float("nan")

        # IoU(top_mass>=0.5*max, expected)
        if has_gt:
            tmask = topmass_mask(heat, 0.5)
            inter = int((tmask & expected).sum())
            union = int((tmask | expected).sum())
            out["iou_topmass50"] = float(inter / union) if union else float("nan")
        else:
            out["iou_topmass50"] = float("nan")

        # Wrong-room mass: mass falling on labeled cells outside expected
        labeled = region_grid > 0
        if has_gt and labeled.any():
            wrong = labeled & (~expected)
            out["wrong_room_mass_ratio"] = float(
                heat[wrong].sum() / max(out["total_mass"], 1e-9)
            )
        else:
            out["wrong_room_mass_ratio"] = float("nan")
    else:
        out["mass_in_expected"] = float("nan")
        out["mass_in_expected_ratio"] = float("nan")
        out["iou_topmass50"] = float("nan")
        out["wrong_room_mass_ratio"] = float("nan")

    # Component count (informative). For raw we don't have it from postprocess,
    # so we just re-run the postprocess to get the component count, but only
    # for diagnostic purposes — the heatmap value passed in is unchanged.
    if n_components_hint is not None:
        out["n_components"] = float(n_components_hint)
    else:
        _, _, kept = postprocess_heatmap(heat)
        out["n_components"] = float(len(kept))

    return out


# ── Visualisation ─────────────────────────────────────────────────────────────
def overlay_heatmap(rgb_map: np.ndarray, heat: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """RGB map (H, W, 3) + heatmap (H, W) -> BGR image with jet overlay."""
    canvas = rgb_map.astype(np.float32).copy()
    if heat.max() > 0:
        h_u8 = (np.clip(heat / max(heat.max(), 1e-9), 0, 1) * 255).astype(np.uint8)
        bgr = cv2.applyColorMap(h_u8, cv2.COLORMAP_JET)
        rgb = bgr[:, :, ::-1].astype(np.float32)
        canvas = canvas * (1 - alpha) + rgb * alpha
    return cv2.cvtColor(np.clip(canvas, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def save_side_by_side(
    out_path: Path,
    rgb_map: np.ndarray,
    raw: np.ndarray,
    clean: np.ndarray,
    expected: Optional[np.ndarray],
    title: str,
) -> None:
    left = overlay_heatmap(rgb_map, raw)
    right = overlay_heatmap(rgb_map, clean)
    if expected is not None and expected.any():
        # Outline expected polygon(s)
        contours = cv2.findContours(
            expected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        for img in (left, right):
            cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    sep = np.full((left.shape[0], 6, 3), 200, dtype=np.uint8)
    combined = np.hstack([left, sep, right])
    cv2.putText(combined, f"RAW   |   CLEAN — {title}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(combined, f"RAW   |   CLEAN — {title}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)


# ── Main pipeline ─────────────────────────────────────────────────────────────
METRIC_KEYS = [
    "max_value", "total_mass", "nonzero_cells",
    "hit@1", "hit@3", "hit@5",
    "mass_in_expected", "mass_in_expected_ratio",
    "iou_topmass50", "wrong_room_mass_ratio",
    "n_components",
]


def aggregate_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean of each metric across queries (NaN-aware) + win/loss counts."""
    raw_means: Dict[str, float] = {}
    clean_means: Dict[str, float] = {}
    for k in METRIC_KEYS:
        raw_vals = [r[f"raw_{k}"] for r in rows
                    if isinstance(r.get(f"raw_{k}"), float) and not np.isnan(r[f"raw_{k}"])]
        cl_vals = [r[f"clean_{k}"] for r in rows
                   if isinstance(r.get(f"clean_{k}"), float) and not np.isnan(r[f"clean_{k}"])]
        raw_means[k] = float(np.mean(raw_vals)) if raw_vals else float("nan")
        clean_means[k] = float(np.mean(cl_vals)) if cl_vals else float("nan")

    # Wins per metric: clean better than raw (depending on metric direction)
    higher_better = {"hit@1", "hit@3", "hit@5", "mass_in_expected_ratio",
                     "iou_topmass50", "max_value"}
    lower_better = {"wrong_room_mass_ratio", "n_components"}
    wins = {"clean": {}, "raw": {}, "tie": {}}
    for k in higher_better | lower_better:
        c = w_c = w_r = t = 0
        for r in rows:
            a, b = r.get(f"raw_{k}"), r.get(f"clean_{k}")
            if a is None or b is None:
                continue
            if isinstance(a, float) and (np.isnan(a) or np.isnan(b)):
                continue
            c += 1
            if k in higher_better:
                if b > a: w_c += 1
                elif b < a: w_r += 1
                else: t += 1
            else:
                if b < a: w_c += 1
                elif b > a: w_r += 1
                else: t += 1
        wins["clean"][k] = w_c
        wins["raw"][k] = w_r
        wins["tie"][k] = t

    return {
        "n_queries": len(rows),
        "raw_means": raw_means,
        "clean_means": clean_means,
        "wins": wins,
    }


def write_summary_md(summary: Dict[str, Any], path: Path, scene_name: str) -> None:
    lines = [f"# Heatmap postprocess evaluation — `{scene_name}`",
             "",
             f"Queries evaluated: **{summary['n_queries']}**",
             "",
             "## Mean metrics (NaN-skipped)",
             "",
             "| metric | raw | clean | Δ (clean−raw) |",
             "|---|---|---|---|"]
    for k in METRIC_KEYS:
        a, b = summary["raw_means"][k], summary["clean_means"][k]
        if np.isnan(a) and np.isnan(b):
            lines.append(f"| {k} | — | — | — |")
        else:
            d = (b - a) if not (np.isnan(a) or np.isnan(b)) else float("nan")
            lines.append(f"| {k} | {a:.4f} | {b:.4f} | {d:+.4f} |")

    lines += ["", "## Per-metric win count (higher-better / lower-better aware)",
              "",
              "| metric | clean wins | raw wins | ties |",
              "|---|---|---|---|"]
    metrics_in_wins = sorted(set(summary["wins"]["clean"].keys()))
    for k in metrics_in_wins:
        lines.append(
            f"| {k} | {summary['wins']['clean'][k]} | "
            f"{summary['wins']['raw'][k]} | {summary['wins']['tie'][k]} |"
        )

    lines += ["",
              "_higher-better_: hit@1, hit@3, hit@5, mass_in_expected_ratio, "
              "iou_topmass50, max_value",
              "",
              "_lower-better_: wrong_room_mass_ratio, n_components",
              ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Driver ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", type=int, required=True)
    parser.add_argument("--dataset-type", default="hssd")
    parser.add_argument(
        "--scene-dataset-config-file",
        default="/workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json",
    )
    parser.add_argument("--data-paths", default="hssd")

    parser.add_argument("--queries", type=Path, default=None,
                        help="Path to JSONL file with queries (preferred).")
    parser.add_argument("--queries-inline", type=str, default=None,
                        help="Comma-separated smoke queries (fallback when no JSONL).")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--score-thresh", type=float, default=0.3)
    args = parser.parse_args()

    if args.queries is None and args.queries_inline is None:
        parser.error("Provide --queries <jsonl> or --queries-inline <csv>")

    # Load queries
    if args.queries is not None:
        queries = load_queries_jsonl(args.queries)
        src_label = str(args.queries)
    else:
        queries = make_inline_queries(args.queries_inline)
        src_label = f"inline:{args.queries_inline}"
    print(f"Loaded {len(queries)} queries from {src_label}")
    if not queries:
        print("No queries to evaluate. Exiting.")
        return

    args.out.mkdir(parents=True, exist_ok=True)

    # Hydra + robot setup
    overrides = [
        f"scene_id={args.scene_id}",
        f"dataset_type={args.dataset_type}",
        f"scene_dataset_config_file={args.scene_dataset_config_file}",
        f"data_paths={args.data_paths}",
    ]
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="object_goal_navigation_cfg", overrides=overrides)

    print("Setting up scene + LSeg categories (this is the slow one-time step)…")
    t0 = time.time()
    robot = HabitatLanguageRobot(cfg)
    robot.setup_scene(cfg.scene_id)
    robot.map.init_categories(get_categories(args.dataset_type))
    scene_name = robot.vlmaps_data_save_dirs[cfg.scene_id].name
    print(f"Scene '{scene_name}' ready in {time.time() - t0:.1f}s")

    room_provider = getattr(robot, "room_provider", None)
    if room_provider is None or not room_provider.is_available():
        print("[WARN] room_provider unavailable; metrics requiring polygons will be NaN.")
        region_grid = np.zeros((robot.map.gs, robot.map.gs), dtype=np.int32)
        label_idx: Dict[str, List[Dict[str, Any]]] = {}
    else:
        region_grid = room_provider._region_grid
        label_idx = regions_by_label(room_provider)
        print(f"room_provider: {len(label_idx)} unique room labels, "
              f"{int((region_grid > 0).sum())} labeled cells")

    # Pre-compute the rgb top-down map for visualisation (cheap)
    rgb_map = None
    if args.save_images:
        try:
            rgb_map = robot.map.generate_rgb_topdown_map()
        except Exception as exc:                                                  # noqa: BLE001
            print(f"[WARN] could not generate RGB topdown map: {exc}; "
                  "images will not be saved.")
            rgb_map = None

    # Per-query loop
    rows: List[Dict[str, Any]] = []
    for i, q in enumerate(queries, 1):
        try:
            raw = compute_raw_heatmap(robot, q.target_label, args.score_thresh)
        except Exception as exc:                                                  # noqa: BLE001
            print(f"[{i}/{len(queries)}] {q.id} '{q.target_label}': "
                  f"compute_raw failed ({exc}); skipping.")
            continue

        clean, _, kept = postprocess_heatmap(raw)
        gt_mask = expected_mask(q, region_grid, label_idx)

        m_raw = evaluate_heatmap(raw, gt_mask, region_grid, n_components_hint=None)
        m_clean = evaluate_heatmap(clean, gt_mask, region_grid,
                                   n_components_hint=len(kept))

        row: Dict[str, Any] = {
            "id": q.id,
            "query": q.query,
            "target_label": q.target_label,
            "query_type": q.query_type,
            "expected_rooms": "|".join(q.expected_rooms),
            "tags": "|".join(q.tags),
            "has_ground_truth": int(bool(gt_mask.any())),
        }
        for k, v in m_raw.items():
            row[f"raw_{k}"] = v
        for k, v in m_clean.items():
            row[f"clean_{k}"] = v
        rows.append(row)

        if args.save_images and rgb_map is not None:
            save_side_by_side(
                args.out / "img" / f"{q.id}.png",
                rgb_map, raw, clean, gt_mask if gt_mask.any() else None,
                title=f"{q.query} [{q.query_type}]",
            )

        # Concise progress
        h1 = m_clean.get("hit@1")
        h1s = "n/a" if (isinstance(h1, float) and np.isnan(h1)) else str(int(h1))
        print(f"[{i:3d}/{len(queries)}] {q.id:8s}  '{q.target_label[:24]:24s}'  "
              f"raw_mass={m_raw['total_mass']:8.2f}  clean_mass={m_clean['total_mass']:8.2f}  "
              f"clean_components={int(m_clean['n_components']):2d}  "
              f"clean_hit@1={h1s}")

    # CSV
    csv_path = args.out / "per_query.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {csv_path} ({len(rows)} rows)")
    else:
        print("No rows to write — all queries failed.")
        return

    # Summary
    summary = aggregate_summary(rows)
    write_summary_md(summary, args.out / "summary.md", scene_name)
    print(f"Wrote {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
