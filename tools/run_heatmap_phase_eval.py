#!/usr/bin/env python3
"""
Run the stressed heatmap-only phase and decide the winning heatmap variant.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_methods import parse_scene_ids, scene_name_from_id


DEFAULT_STRESSED_DIR = REPO_ROOT / "tools" / "eval_queries" / "stressed_heatmap"
DEFAULT_SCENE_CONFIG = "/workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json"
TARGET_QUERIES = 500
HIGHER_BETTER = {"hit@1", "hit@5", "mass_in_expected_ratio", "iou_topmass50"}
LOWER_BETTER = {"wrong_room_mass_ratio", "n_components"}


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


def _bootstrap_runner(module_name: str, argv: List[str]) -> int:
    repo_parent = REPO_ROOT.parent
    bootstrap = f"""
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
from {module_name} import main
sys.argv = [sys.argv[0]] + sys.argv[3:]
main()
"""
    cmd = [sys.executable, "-c", bootstrap, str(REPO_ROOT), str(repo_parent), *argv]
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(REPO_ROOT), text=True).returncode


@dataclass
class ExpansionInfo:
    scene_name: str
    source_queries: int
    expanded_queries: int


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
        raise SystemExit("Cannot expand an empty stressed heatmap batch.")
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


def _load_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: str) -> float:
    if value in {"", "—", "nan", "NaN", "None", None}:
        return float("nan")
    return float(value)


def _filter_degenerate_metrics(global_rows: List[dict]) -> List[str]:
    retained: List[str] = []
    for metric in sorted(HIGHER_BETTER | LOWER_BETTER):
        values = []
        for row in global_rows:
            values.append(_to_float(row.get(f"raw_{metric}", "nan")))
            values.append(_to_float(row.get(f"clean_{metric}", "nan")))
        values = [v for v in values if v == v]
        if not values:
            continue
        if all(abs(v) < 1e-12 for v in values):
            continue
        retained.append(metric)
    return retained


def _mean(values: Iterable[float]) -> float:
    vals = [v for v in values if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def _choose_winner(global_rows: List[dict], retained_metrics: List[str]) -> str:
    clean_wins = 0
    raw_wins = 0
    for metric in retained_metrics:
        raw_mean = _mean(_to_float(row.get(f"raw_{metric}", "nan")) for row in global_rows)
        clean_mean = _mean(_to_float(row.get(f"clean_{metric}", "nan")) for row in global_rows)
        if raw_mean != raw_mean or clean_mean != clean_mean:
            continue
        if metric in HIGHER_BETTER:
            if clean_mean > raw_mean:
                clean_wins += 1
            elif clean_mean < raw_mean:
                raw_wins += 1
        else:
            if clean_mean < raw_mean:
                clean_wins += 1
            elif clean_mean > raw_mean:
                raw_wins += 1
    if clean_wins > raw_wins:
        return "postprocessed"
    if raw_wins > clean_wins:
        return "baseline"

    raw_hit = _mean(_to_float(row.get("raw_hit@1", "nan")) for row in global_rows)
    clean_hit = _mean(_to_float(row.get("clean_hit@1", "nan")) for row in global_rows)
    return "postprocessed" if clean_hit >= raw_hit else "baseline"


def _write_phase_summary(
    *,
    out_dir: Path,
    scene_rows: List[dict],
    retained_metrics: List[str],
    removed_metrics: List[str],
    winner: str,
) -> None:
    csv_rows = []
    for row in scene_rows:
        flat = {"scene_name": row["scene_name"], "n_queries": int(row["n_queries"])}
        for metric in retained_metrics:
            flat[f"raw_{metric}"] = row.get(f"raw_{metric}", "")
            flat[f"clean_{metric}"] = row.get(f"clean_{metric}", "")
        csv_rows.append(flat)

    csv_path = out_dir / "aggregate_cross_scenes.csv"
    md_path = out_dir / "aggregate_cross_scenes.md"
    if csv_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    lines = [
        "# Heatmap phase aggregate",
        "",
        f"Winner: **{winner}**",
        "",
        f"Retained metrics: {', '.join(retained_metrics) if retained_metrics else 'none'}",
        f"Removed all-zero/all-NaN metrics: {', '.join(removed_metrics) if removed_metrics else 'none'}",
        "",
    ]
    if retained_metrics:
        lines += [
            "| scene | n | " + " | ".join(f"raw {m} | clean {m}" for m in retained_metrics) + " |",
            "|---|---:|" + "|".join(["---:|---:"] * len(retained_metrics)),
        ]
        for row in csv_rows:
            parts = [row["scene_name"], str(row["n_queries"])]
            for metric in retained_metrics:
                parts.append(str(row.get(f"raw_{metric}", "")))
                parts.append(str(row.get(f"clean_{metric}", "")))
            lines.append("| " + " | ".join(parts) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-ids", required=True)
    parser.add_argument("--sources", type=Path, default=DEFAULT_STRESSED_DIR)
    parser.add_argument("--dataset-type", default="hssd")
    parser.add_argument("--data-paths", default="hssd")
    parser.add_argument("--scene-dataset-config-file", default=DEFAULT_SCENE_CONFIG)
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--target-queries", type=int, default=TARGET_QUERIES)
    parser.add_argument("--seed", type=int, default=24042026)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    scene_ids = parse_scene_ids(args.scene_ids)
    args.out.mkdir(parents=True, exist_ok=True)
    expanded_dir = args.out / "queries_500"
    expanded_dir.mkdir(parents=True, exist_ok=True)

    expansions: List[ExpansionInfo] = []
    for scene_id in scene_ids:
        scene_name = _resolve_scene_name(scene_id, args.dataset_type, args.data_paths)
        if not scene_name:
            raise SystemExit(f"Could not resolve scene_name for scene_id={scene_id}")
        src_path = args.sources / f"{scene_name}.jsonl"
        if not src_path.exists():
            raise SystemExit(f"Missing stressed heatmap JSONL: {src_path}")
        source_rows = _load_jsonl(src_path)
        expanded = _expand_queries(source_rows, args.target_queries, args.seed + scene_id)
        _write_jsonl(expanded_dir / f"{scene_name}.jsonl", expanded)
        expansions.append(ExpansionInfo(scene_name, len(source_rows), len(expanded)))

    argv = [
        "--scene-ids", args.scene_ids,
        "--queries", str(expanded_dir),
        "--dataset-type", args.dataset_type,
        "--data-paths", args.data_paths,
        "--scene-dataset-config-file", args.scene_dataset_config_file,
        "--score-thresh", str(args.score_thresh),
        "--out", str(args.out),
    ]
    if args.save_images:
        argv.append("--save-images")
    rc = _bootstrap_runner("tools.run_heatmap_offline_eval", argv)
    if rc != 0:
        raise SystemExit(rc)

    heatmap_root = args.out / "heatmap_offline"
    scene_rows = []
    for info in expansions:
        rows = _load_csv_rows(heatmap_root / info.scene_name / "aggregate_by_slice.csv")
        global_row = next((row for row in rows if row.get("slice_kind") == "global"), None)
        if global_row:
            global_row["scene_name"] = info.scene_name
            scene_rows.append(global_row)

    retained_metrics = _filter_degenerate_metrics(scene_rows)
    removed_metrics = sorted((HIGHER_BETTER | LOWER_BETTER) - set(retained_metrics))
    winner = _choose_winner(scene_rows, retained_metrics) if scene_rows else "unknown"
    _write_phase_summary(
        out_dir=args.out,
        scene_rows=scene_rows,
        retained_metrics=retained_metrics,
        removed_metrics=removed_metrics,
        winner=winner,
    )

    config = {
        "created_at": datetime.now().isoformat(),
        "scene_ids": scene_ids,
        "sources": str(args.sources),
        "expanded_queries_dir": str(expanded_dir),
        "target_queries_per_scene": args.target_queries,
        "seed": args.seed,
        "dataset_type": args.dataset_type,
        "data_paths": args.data_paths,
        "scene_dataset_config_file": args.scene_dataset_config_file,
        "score_thresh": args.score_thresh,
        "winner": winner,
        "removed_metrics": removed_metrics,
        "expansion": [info.__dict__ for info in expansions],
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[heatmap-phase] winner={winner}")


if __name__ == "__main__":
    main()
