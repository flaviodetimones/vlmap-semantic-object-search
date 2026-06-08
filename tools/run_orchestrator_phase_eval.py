#!/usr/bin/env python3
"""
Run the stressed orchestrator phase over a fixed winning heatmap.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_methods import parse_scene_ids, scene_name_from_id


DEFAULT_STRESSED_DIR = REPO_ROOT / "tools" / "eval_queries" / "stressed_orchestrator"
DEFAULT_SCENE_CONFIG = "/workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json"
TARGET_QUERIES = 100


def _local_scene_root(dataset_type: str, data_paths: str) -> Path:
    candidates = []
    if str(dataset_type).lower() == "hssd" or str(data_paths).lower() == "hssd":
        candidates += [
            Path("/workspace/data/vlmaps_dataset_hssd"),
            REPO_ROOT.parent / "data" / "vlmaps_dataset_hssd",
        ]
    candidates += [
        Path("/workspace/data/vlmaps_dataset"),
        REPO_ROOT.parent / "data" / "vlmaps_dataset",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_scene_name(scene_id: int, dataset_type: str, data_paths: str) -> str | None:
    name = scene_name_from_id(scene_id, dataset_type, data_paths)
    if name:
        return name
    root = _local_scene_root(dataset_type, data_paths)
    scene_dirs = sorted([p for p in root.iterdir() if p.is_dir()]) if root.exists() else []
    if 0 <= scene_id < len(scene_dirs):
        return scene_dirs[scene_id].name
    return None


def _bootstrap_runner(argv: List[str]) -> int:
    repo_parent = REPO_ROOT.parent
    bootstrap = """
import sys
from pathlib import Path
repo = Path(sys.argv[1])
repo_parent = Path(sys.argv[2])
sys.path.insert(0, str(repo))
import tools.eval_methods as em
def _scenes_dir(dataset_type, data_paths):
    candidates = []
    if str(dataset_type).lower() == 'hssd' or str(data_paths).lower() == 'hssd':
        candidates += [Path('/workspace/data/vlmaps_dataset_hssd'), repo_parent / 'data' / 'vlmaps_dataset_hssd']
    candidates += [Path('/workspace/data/vlmaps_dataset'), repo_parent / 'data' / 'vlmaps_dataset']
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
def _scene_name_from_id(scene_id, dataset_type, data_paths):
    root = _scenes_dir(dataset_type, data_paths)
    if not root.exists():
        return None
    scene_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if scene_id < 0 or scene_id >= len(scene_dirs):
        return None
    return scene_dirs[scene_id].name
em.scenes_dir = _scenes_dir
em.scene_name_from_id = _scene_name_from_id
from tools.run_full_eval import main
sys.argv = [sys.argv[0]] + sys.argv[3:]
main()
"""
    cmd = [sys.executable, "-c", bootstrap, str(REPO_ROOT), str(repo_parent), *argv]
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(REPO_ROOT), text=True).returncode


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _expand_queries(rows: List[dict], target_count: int, seed: int) -> List[dict]:
    if not rows:
        raise SystemExit("Cannot expand an empty stressed orchestrator batch.")
    expanded: List[dict] = []
    order = list(range(len(rows)))
    import random

    rng = random.Random(seed)
    replica = 0
    while len(expanded) < target_count:
        rng.shuffle(order)
        for idx in order:
            src = dict(rows[idx])
            replica += 1
            src["source_id"] = src.get("id")
            src["replica_index"] = replica
            src["id"] = f"{src['id']}__rep{replica:04d}"
            expanded.append(src)
            if len(expanded) >= target_count:
                break
    return expanded


def _run_command(cmd: List[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(REPO_ROOT), text=True).returncode


def _subset_methods(fixed_heatmap: str) -> str:
    return "Ob_Hp,Oe_Hp" if fixed_heatmap == "postprocessed" else "Ob_Hb,Oe_Hb"


def _load_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: str) -> float:
    if value in {"", "—", "nan", "NaN", "None", None}:
        return float("nan")
    return float(value)


def _pruned_metrics(rows: List[dict]) -> List[str]:
    candidates = [
        "SR",
        "Object SR",
        "CFR",
        "CT2R",
        "Rooms Before Success",
        "Wrong Visits",
        "Mean Pose Updates",
        "Early Stop Rate",
    ]
    retained: List[str] = []
    for metric in candidates:
        vals = [_to_float(row.get(metric, "nan")) for row in rows]
        vals = [v for v in vals if v == v]
        if not vals:
            continue
        if all(abs(v) < 1e-12 for v in vals):
            continue
        retained.append(metric)
    return retained


def _write_phase_summary(out_dir: Path, rows: List[dict], retained_metrics: List[str], removed_metrics: List[str]) -> None:
    csv_path = out_dir / "aggregate_cross_scenes_pruned.csv"
    md_path = out_dir / "aggregate_cross_scenes_pruned.md"
    flat_rows = []
    for row in rows:
        flat = {"method": row["method"]}
        for metric in retained_metrics:
            flat[metric] = row.get(metric, "")
        flat_rows.append(flat)
    if flat_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)
    lines = [
        "# Orchestrator phase aggregate (pruned)",
        "",
        f"Retained metrics: {', '.join(retained_metrics) if retained_metrics else 'none'}",
        f"Removed all-zero/all-NaN metrics: {', '.join(removed_metrics) if removed_metrics else 'none'}",
        "",
    ]
    if retained_metrics:
        lines += [
            "| method | " + " | ".join(retained_metrics) + " |",
            "|---|" + "|".join(["---:"] * len(retained_metrics)) + "|",
        ]
        for row in flat_rows:
            lines.append("| " + " | ".join([row["method"], *[str(row.get(metric, "")) for metric in retained_metrics]]) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-ids", required=True)
    parser.add_argument("--sources", type=Path, default=DEFAULT_STRESSED_DIR)
    parser.add_argument("--fixed-heatmap", choices=["postprocessed", "baseline"], default="postprocessed")
    parser.add_argument("--dataset-type", default="hssd")
    parser.add_argument("--data-paths", default="hssd")
    parser.add_argument("--scene-dataset-config-file", default=DEFAULT_SCENE_CONFIG)
    parser.add_argument("--policy-mode", choices=["heuristic", "hybrid", "llm"], default="hybrid")
    parser.add_argument("--yoloe-conf-thresh", type=float, default=0.30)
    parser.add_argument("--yoloe-weights", default=None,
                        help="Optional YOLOE .pt weights path. Defaults to the base model.")
    parser.add_argument("--per-query-timeout", type=int, default=180)
    parser.add_argument("--target-queries", type=int, default=TARGET_QUERIES)
    parser.add_argument("--seed", type=int, default=24042026)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    scene_ids = parse_scene_ids(args.scene_ids)
    args.out.mkdir(parents=True, exist_ok=True)
    expanded_dir = args.out / "queries_100"
    expanded_dir.mkdir(parents=True, exist_ok=True)

    expansion = []
    for scene_id in scene_ids:
        scene_name = _resolve_scene_name(scene_id, args.dataset_type, args.data_paths)
        if not scene_name:
            raise SystemExit(f"Could not resolve scene_name for scene_id={scene_id}")
        src_path = args.sources / f"{scene_name}.jsonl"
        if not src_path.exists():
            raise SystemExit(f"Missing stressed orchestrator JSONL: {src_path}")
        source_rows = _load_jsonl(src_path)
        expanded = _expand_queries(source_rows, args.target_queries, args.seed + scene_id)
        _write_jsonl(expanded_dir / f"{scene_name}.jsonl", expanded)
        expansion.append(
            {
                "scene_name": scene_name,
                "source_queries": len(source_rows),
                "expanded_queries": len(expanded),
            }
        )

    argv = [
        "--scene-ids", args.scene_ids,
        "--queries", str(expanded_dir),
        "--dataset-type", args.dataset_type,
        "--data-paths", args.data_paths,
        "--scene-dataset-config-file", args.scene_dataset_config_file,
        "--policy-mode", args.policy_mode,
        "--yoloe-conf-thresh", str(args.yoloe_conf_thresh),
        "--methods", _subset_methods(args.fixed_heatmap),
        "--out", str(args.out),
        "--per-query-timeout", str(args.per_query_timeout),
    ]
    if args.yoloe_weights:
        argv.extend(["--yoloe-weights", args.yoloe_weights])
    rc = _bootstrap_runner(argv)
    if rc != 0:
        raise SystemExit(rc)

    rows = _load_csv_rows(args.out / "pipeline_full" / "aggregate_cross_scenes.csv")
    retained_metrics = _pruned_metrics(rows)
    all_metrics = [
        "SR", "Object SR", "CFR", "CT2R", "Rooms Before Success",
        "Wrong Visits", "Mean Pose Updates", "Early Stop Rate",
    ]
    removed_metrics = [metric for metric in all_metrics if metric not in retained_metrics]
    _write_phase_summary(args.out, rows, retained_metrics, removed_metrics)

    config = {
        "created_at": datetime.now().isoformat(),
        "scene_ids": scene_ids,
        "sources": str(args.sources),
        "expanded_queries_dir": str(expanded_dir),
        "target_queries_per_scene": args.target_queries,
        "seed": args.seed,
        "fixed_heatmap": args.fixed_heatmap,
        "methods": _subset_methods(args.fixed_heatmap).split(","),
        "dataset_type": args.dataset_type,
        "data_paths": args.data_paths,
        "scene_dataset_config_file": args.scene_dataset_config_file,
        "policy_mode": args.policy_mode,
        "yoloe_conf_thresh": args.yoloe_conf_thresh,
        "per_query_timeout": args.per_query_timeout,
        "removed_metrics": removed_metrics,
        "expansion": expansion,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
