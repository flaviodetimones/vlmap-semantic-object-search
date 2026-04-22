#!/usr/bin/env python3
"""
Run the offline heatmap analysis for one or more scenes, using the normalized
JSONL query set and storing outputs under a standard eval_runs directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tools.eval_methods import default_queries_path, parse_scene_ids, scene_name_from_id


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_queries_path(raw_queries: str | None, scene_name: str) -> Path:
    if not raw_queries:
        return default_queries_path(scene_name)
    path = Path(raw_queries)
    if path.is_dir():
        return path / f"{scene_name}.jsonl"
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-ids", required=True,
                        help="Comma-separated scene ids.")
    parser.add_argument("--queries", default=None,
                        help="Optional JSONL path (single scene) or directory holding {scene_name}.jsonl.")
    parser.add_argument("--dataset-type", default="hssd")
    parser.add_argument("--data-paths", default="hssd")
    parser.add_argument(
        "--scene-dataset-config-file",
        default="/workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json",
    )
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    scene_ids = parse_scene_ids(args.scene_ids)
    if args.queries and len(scene_ids) > 1:
        qpath = Path(args.queries)
        if qpath.exists() and qpath.is_file():
            raise SystemExit("With multiple scene ids, --queries must be a directory containing {scene_name}.jsonl files.")
    args.out.mkdir(parents=True, exist_ok=True)
    heatmap_root = args.out / "heatmap_offline"
    heatmap_root.mkdir(exist_ok=True)

    config = {
        "created_at": datetime.now().isoformat(),
        "scene_ids": scene_ids,
        "dataset_type": args.dataset_type,
        "data_paths": args.data_paths,
        "scene_dataset_config_file": args.scene_dataset_config_file,
        "save_images": args.save_images,
        "score_thresh": args.score_thresh,
        "scenes": [],
    }

    for scene_id in scene_ids:
        scene_name = scene_name_from_id(scene_id, args.dataset_type, args.data_paths)
        if not scene_name:
            raise SystemExit(f"Could not resolve scene_name for scene_id={scene_id}")
        queries_path = _resolve_queries_path(args.queries, scene_name)
        if not queries_path.exists():
            raise SystemExit(f"Query JSONL not found for scene '{scene_name}': {queries_path}")

        scene_out = heatmap_root / scene_name
        scene_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "eval_heatmap_postprocess.py"),
            "--scene-id", str(scene_id),
            "--dataset-type", args.dataset_type,
            "--scene-dataset-config-file", args.scene_dataset_config_file,
            "--data-paths", args.data_paths,
            "--queries", str(queries_path),
            "--out", str(scene_out),
            "--score-thresh", str(args.score_thresh),
        ]
        if args.save_images:
            cmd.append("--save-images")
        print(f"$ {' '.join(cmd)}")
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True).returncode
        config["scenes"].append({
            "scene_id": scene_id,
            "scene_name": scene_name,
            "queries_jsonl": str(queries_path),
            "returncode": rc,
            "outputs": {
                "per_query_csv": str(scene_out / "per_query.csv"),
                "aggregate_by_slice_csv": str(scene_out / "aggregate_by_slice.csv"),
                "summary_md": str(scene_out / "summary.md"),
                "img_dir": str(scene_out / "img"),
            },
        })

    config_path = args.out / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote {config_path}")


if __name__ == "__main__":
    main()
