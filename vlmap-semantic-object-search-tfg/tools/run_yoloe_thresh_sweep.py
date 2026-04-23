#!/usr/bin/env python3
"""
Run a YOLOE confidence-threshold sweep for a single method on one scene.

This is intentionally orthogonal to the 2x2 harness: the method is fixed
(`Oe_Hp` by default) and only the visual verification threshold changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.aggregate_full_eval import _compute_query_metrics
from tools.compare_nav_runs import parse_manifest
from tools.eval_methods import method_spec_map, scene_name_from_id


def parse_thresholds(raw: str) -> List[float]:
    values = []
    for chunk in str(raw).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    return sorted(set(values))


def threshold_suffix(thresh: float) -> str:
    return f"t{int(round(float(thresh) * 100)):03d}"


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def aggregate_manifest(manifest_path: Path, *, thresh: float | None = None) -> dict:
    rows = parse_manifest(manifest_path)
    metrics_rows = [_compute_query_metrics(row) for row in rows]
    room_object_rows = [m for m in metrics_rows if m.get("query_type") == "room_object"]
    fp_proxy_room_object = float("nan")
    if room_object_rows:
        fp_proxy_room_object = _safe_mean(
            1.0 if (bool(m.get("found")) and not bool(m.get("success"))) else 0.0
            for m in room_object_rows
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if thresh is None:
        thresh = manifest.get("yoloe_conf_thresh")

    return {
        "thresh": float(thresh) if thresh is not None else float("nan"),
        "n_queries": len(metrics_rows),
        "sr": _safe_mean(m["success"] for m in metrics_rows),
        "object_sr": _safe_mean(m["object_success"] for m in metrics_rows),
        "found_rate": _safe_mean(m["found"] for m in metrics_rows),
        "wrong_visits": _safe_mean(m["wrong_visits"] for m in metrics_rows),
        "cfr": _safe_mean(m["cfr"] for m in metrics_rows),
        "ct2r": _safe_mean(m["ct2r"] for m in metrics_rows),
        "early_stop_rate": _safe_mean(m["early_stop"] for m in metrics_rows),
        "fp_proxy_room_object": fp_proxy_room_object,
        "n_room_object": len(room_object_rows),
    }


def _fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.4f}"


def _write_csv(rows: List[dict], out_path: Path) -> None:
    fieldnames = [
        "thresh",
        "n_queries",
        "sr",
        "object_sr",
        "found_rate",
        "wrong_visits",
        "cfr",
        "ct2r",
        "early_stop_rate",
        "fp_proxy_room_object",
        "n_room_object",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(rows: List[dict], out_path: Path, *, scene_name: str, method_key: str) -> None:
    lines = [
        f"## YOLO conf-thresh sweep — scene={scene_name} method={method_key}",
        "",
        "| thresh | n_queries | SR | Object SR | Found Rate | Wrong Visits | CFR | CT2R | Early Stop Rate | FP Proxy Room Object | n_room_object |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['thresh']:.2f} | {row['n_queries']} | {_fmt(row['sr'])} | {_fmt(row['object_sr'])} | "
            f"{_fmt(row['found_rate'])} | {_fmt(row['wrong_visits'])} | {_fmt(row['cfr'])} | {_fmt(row['ct2r'])} | "
            f"{_fmt(row['early_stop_rate'])} | {_fmt(row['fp_proxy_room_object'])} | {row['n_room_object']} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_command(cmd: List[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True, type=int)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--thresholds", default="0.30,0.40,0.50,0.60")
    parser.add_argument("--method-key", default="Oe_Hp")
    parser.add_argument("--dataset-type", default="hssd")
    parser.add_argument("--data-paths", default="hssd")
    parser.add_argument(
        "--scene-dataset-config-file",
        default="/workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json",
    )
    parser.add_argument("--policy-mode", choices=["heuristic", "hybrid", "llm"], default="hybrid")
    parser.add_argument("--per-query-timeout", type=int, default=180)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if not args.queries.exists():
        raise SystemExit(f"Queries JSONL not found: {args.queries}")

    thresholds = parse_thresholds(args.thresholds)
    if not thresholds:
        raise SystemExit("No thresholds parsed from --thresholds")

    specs = method_spec_map()
    if args.method_key not in specs:
        raise SystemExit(f"Unknown --method-key '{args.method_key}'")
    spec = specs[args.method_key]

    scene_name = scene_name_from_id(args.scene_id, args.dataset_type, args.data_paths)
    if not scene_name:
        raise SystemExit(f"Could not resolve scene_name for scene_id={args.scene_id}")

    args.out.mkdir(parents=True, exist_ok=True)
    sweep_root = args.out / "sweep"
    sweep_root.mkdir(exist_ok=True)
    queries_copy = args.out / "queries.jsonl"
    shutil.copy2(args.queries, queries_copy)

    config = {
        "created_at": datetime.now().isoformat(),
        "scene_id": args.scene_id,
        "scene_name": scene_name,
        "queries": str(args.queries),
        "queries_copy": str(queries_copy),
        "thresholds": thresholds,
        "method": {
            "key": args.method_key,
            "entrypoint": spec["entrypoint"],
            "heatmap_mode": spec["heatmap_mode"],
        },
        "dataset_type": args.dataset_type,
        "data_paths": args.data_paths,
        "scene_dataset_config_file": args.scene_dataset_config_file,
        "policy_mode": args.policy_mode,
        "per_query_timeout": args.per_query_timeout,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows = []
    for thresh in thresholds:
        method_out = sweep_root / f"{args.method_key}_{threshold_suffix(thresh)}"
        method_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_nav_eval.py"),
            "--queries", str(queries_copy),
            "--entrypoint", spec["entrypoint"],
            "--heatmap-mode", spec["heatmap_mode"],
            "--yoloe-conf-thresh", str(thresh),
            "--scene-id", str(args.scene_id),
            "--scene-name", scene_name,
            "--dataset-type", args.dataset_type,
            "--scene-dataset-config-file", args.scene_dataset_config_file,
            "--data-paths", args.data_paths,
            "--per-query-timeout", str(args.per_query_timeout),
            "--out", str(method_out),
        ]
        if spec["entrypoint"] == "executor":
            cmd += ["--policy-mode", args.policy_mode]

        rc = _run_command(cmd)
        if rc != 0:
            print(f"[WARN] run finished with rc={rc} for threshold {thresh:.2f}")

        manifest_path = method_out / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"Missing manifest after threshold {thresh:.2f}: {manifest_path}")
        rows.append(aggregate_manifest(manifest_path, thresh=thresh))

    rows.sort(key=lambda row: row["thresh"])
    _write_csv(rows, args.out / "compare_thresh.csv")
    _write_md(rows, args.out / "compare_thresh.md", scene_name=scene_name, method_key=args.method_key)
    print(f"Wrote CSV: {args.out / 'compare_thresh.csv'}")
    print(f"Wrote MD:  {args.out / 'compare_thresh.md'}")


if __name__ == "__main__":
    main()
