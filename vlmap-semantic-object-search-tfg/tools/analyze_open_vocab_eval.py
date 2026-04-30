#!/usr/bin/env python3
"""
Focused analyser for the open-vocabulary evaluation battery.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.aggregate_full_eval import _compute_query_metrics
from tools.compare_nav_runs import parse_manifest

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


MAIN_METRICS = ["found_rate", "object_sr", "wrong_visits", "mean_pose_updates"]
CONFIRM_SOURCES = [
    "arrival",
    "turn_to_face",
    "centering",
    "local_scan",
    "alternative_route",
    "none",
]
RESOLUTION_SOURCES = ["direct", "llm", "fallback", "room_command", "unknown"]


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _target_summary(row: dict) -> dict:
    payload = row.get("eval_summary") or {}
    targets = payload.get("target_summaries") or {}
    label = str(row.get("target_label", "")).strip()
    if label in targets:
        return targets[label]
    lowered = {str(k).lower(): v for k, v in targets.items()}
    return lowered.get(label.lower(), {}) if label else {}


def _iter_rows(manifest_path: Path) -> Iterable[dict]:
    for row in parse_manifest(manifest_path):
        metrics = _compute_query_metrics(row)
        tsum = _target_summary(row)
        source = str(tsum.get("final_confirmation_source") or "none")
        resolution_source = str(tsum.get("resolution_source") or "unknown")
        yield {
            "scene_name": str(row.get("scene_name") or ""),
            "method": str(row.get("method_key") or ""),
            "query_type": str(row.get("query_type") or "object"),
            "target_label": str(row.get("target_label") or ""),
            "tags": row.get("tags") or [],
            "found": float(tsum.get("found", row.get("found", False)) or row.get("found", False)),
            "object_sr": metrics.get("object_success", float("nan")),
            "wrong_visits": metrics.get("wrong_visits", float("nan")),
            "mean_pose_updates": metrics.get("pose_updates", float("nan")),
            "confirmation_source": source,
            "resolution_source": resolution_source,
        }


def _is_open_vocab_row(row: dict) -> bool:
    tags = row.get("tags") or []
    if isinstance(tags, str):
        tags = [chunk for chunk in tags.split("|") if chunk]
    return "open_vocab" in tags


def _aggregate_rows(rows: List[dict], group_keys: Tuple[str, ...]) -> List[dict]:
    grouped: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(k, "")) for k in group_keys)].append(row)

    out: List[dict] = []
    for key, bucket in sorted(grouped.items()):
        record = {group_keys[i]: key[i] for i in range(len(group_keys))}
        record["n_queries"] = len(bucket)
        record["found_rate"] = _safe_mean(r["found"] for r in bucket)
        record["object_sr"] = _safe_mean(r["object_sr"] for r in bucket)
        record["wrong_visits"] = _safe_mean(r["wrong_visits"] for r in bucket)
        record["mean_pose_updates"] = _safe_mean(r["mean_pose_updates"] for r in bucket)
        conf_counts = Counter(r["confirmation_source"] for r in bucket)
        for src in CONFIRM_SOURCES:
            record[f"confirm_{src}_rate"] = conf_counts.get(src, 0) / len(bucket)
        local_total = sum(conf_counts.get(src, 0) for src in ("arrival", "turn_to_face", "centering", "local_scan"))
        record["local_confirmation_rate"] = local_total / len(bucket)
        res_counts = Counter(r["resolution_source"] for r in bucket)
        for src in RESOLUTION_SOURCES:
            record[f"resolution_{src}_rate"] = res_counts.get(src, 0) / len(bucket)
        out.append(record)
    return out


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | int | str) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        return f"{value:.4f}"
    return str(value)


def _write_md(path: Path, title: str, rows: List[dict], leading_cols: List[str]) -> None:
    if not rows:
        return
    cols = leading_cols + MAIN_METRICS + [
        "confirm_arrival_rate",
        "confirm_turn_to_face_rate",
        "confirm_centering_rate",
        "confirm_local_scan_rate",
        "confirm_alternative_route_rate",
        "local_confirmation_rate",
        "resolution_direct_rate",
        "resolution_llm_rate",
        "resolution_fallback_rate",
    ]
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_metric(rows: List[dict], metric: str, out_path: Path) -> None:
    if plt is None or not rows:
        return
    labels = [f"{row.get('method', '')}:{row.get('target_label', '')}" for row in rows]
    values = [row.get(metric, float("nan")) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.45), 4.5))
    ax.bar(range(len(labels)), values, color="#4472c4")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(metric.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _discover_manifests(run_root: Path) -> Dict[str, Path]:
    manifests: Dict[str, Path] = {}
    root = run_root / "pipeline_full"
    if not root.exists():
        root = run_root
    for scene_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for method_dir in sorted([p for p in scene_dir.iterdir() if p.is_dir()]):
            manifest = method_dir / "manifest.json"
            if manifest.exists():
                manifests[f"{scene_dir.name}/{method_dir.name}"] = manifest
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--manifest", action="append", default=[],
                        help="Explicit scene/method=manifest mapping.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    manifests: Dict[str, Path] = {}
    if args.run:
        manifests.update(_discover_manifests(args.run))
    for item in args.manifest:
        if "=" not in item:
            raise SystemExit(f"--manifest expects key=path, got {item!r}")
        key, raw_path = item.split("=", 1)
        manifests[key.strip()] = Path(raw_path.strip())
    if not manifests:
        raise SystemExit("No manifests provided.")

    rows: List[dict] = []
    for key, manifest_path in manifests.items():
        scene_name, method = key.split("/", 1) if "/" in key else ("", key)
        for row in _iter_rows(manifest_path):
            row.setdefault("scene_name", scene_name)
            row.setdefault("method", method)
            if _is_open_vocab_row(row):
                rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_method = _aggregate_rows(rows, ("method",))
    by_scene_method = _aggregate_rows(rows, ("scene_name", "method"))
    by_target = _aggregate_rows(rows, ("method", "target_label"))

    _write_csv(args.out_dir / "open_vocab_by_method.csv", by_method)
    _write_csv(args.out_dir / "open_vocab_by_scene_method.csv", by_scene_method)
    _write_csv(args.out_dir / "open_vocab_by_target.csv", by_target)
    _write_md(args.out_dir / "open_vocab_by_method.md", "Open-vocab summary by method", by_method, ["method", "n_queries"])
    _write_md(
        args.out_dir / "open_vocab_by_scene_method.md",
        "Open-vocab summary by scene and method",
        by_scene_method,
        ["scene_name", "method", "n_queries"],
    )
    _write_md(
        args.out_dir / "open_vocab_by_target.md",
        "Open-vocab summary by target",
        by_target,
        ["method", "target_label", "n_queries"],
    )

    if plt is not None:
        plots_dir = args.out_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        _plot_metric(by_target, "found_rate", plots_dir / "found_rate_by_target.png")
        _plot_metric(by_target, "object_sr", plots_dir / "object_sr_by_target.png")
        _plot_metric(by_target, "wrong_visits", plots_dir / "wrong_visits_by_target.png")
        _plot_metric(by_target, "mean_pose_updates", plots_dir / "mean_pose_updates_by_target.png")

    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
