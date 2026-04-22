#!/usr/bin/env python3
"""
Run the full online 2x2 evaluation:

  Ob_Hb, Oe_Hb, Ob_Hp, Oe_Hp

and aggregate the resulting manifests into ready-to-read CSV/MD tables.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from tools.eval_methods import (
    METHOD_SPECS,
    default_queries_path,
    parse_scene_ids,
    scene_name_from_id,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_queries_path(raw_queries: str | None, scene_name: str) -> Path:
    if not raw_queries:
        return default_queries_path(scene_name)
    path = Path(raw_queries)
    if path.is_dir():
        return path / f"{scene_name}.jsonl"
    return path


def _run_command(cmd: List[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True)
    return proc.returncode


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
    parser.add_argument("--policy-mode", choices=["heuristic", "hybrid", "llm"], default="hybrid")
    parser.add_argument("--per-query-timeout", type=int, default=180)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    scene_ids = parse_scene_ids(args.scene_ids)
    if args.queries and len(scene_ids) > 1:
        qpath = Path(args.queries)
        if qpath.exists() and qpath.is_file():
            raise SystemExit("With multiple scene ids, --queries must be a directory containing {scene_name}.jsonl files.")
    args.out.mkdir(parents=True, exist_ok=True)

    run_config = {
        "created_at": datetime.now().isoformat(),
        "scene_ids": scene_ids,
        "dataset_type": args.dataset_type,
        "data_paths": args.data_paths,
        "scene_dataset_config_file": args.scene_dataset_config_file,
        "policy_mode": args.policy_mode,
        "per_query_timeout": args.per_query_timeout,
        "methods": METHOD_SPECS,
        "scenes": [],
    }

    queries_root = args.out / "queries"
    pipeline_root = args.out / "pipeline_2x2"
    queries_root.mkdir(exist_ok=True)
    pipeline_root.mkdir(exist_ok=True)

    for scene_id in scene_ids:
        scene_name = scene_name_from_id(scene_id, args.dataset_type, args.data_paths)
        if not scene_name:
            raise SystemExit(f"Could not resolve scene_name for scene_id={scene_id}")

        queries_path = _resolve_queries_path(args.queries, scene_name)
        if not queries_path.exists():
            raise SystemExit(f"Query JSONL not found for scene '{scene_name}': {queries_path}")

        copied_queries = queries_root / f"{scene_name}.jsonl"
        shutil.copy2(queries_path, copied_queries)

        scene_out = pipeline_root / scene_name
        scene_out.mkdir(parents=True, exist_ok=True)
        method_manifests: Dict[str, str] = {}

        for spec in METHOD_SPECS:
            method_out = scene_out / spec["key"]
            method_out.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(REPO_ROOT / "tools" / "run_nav_eval.py"),
                "--queries", str(copied_queries),
                "--entrypoint", spec["entrypoint"],
                "--heatmap-mode", spec["heatmap_mode"],
                "--scene-id", str(scene_id),
                "--scene-name", scene_name,
                "--dataset-type", args.dataset_type,
                "--scene-dataset-config-file", args.scene_dataset_config_file,
                "--data-paths", args.data_paths,
                "--per-query-timeout", str(args.per_query_timeout),
                "--out", str(method_out),
            ]
            if spec["entrypoint"] == "executor":
                cmd.extend(["--policy-mode", args.policy_mode])
            rc = _run_command(cmd)
            manifest_path = method_out / "manifest.json"
            if not manifest_path.exists():
                raise SystemExit(
                    f"Method {spec['key']} did not produce manifest.json (rc={rc})"
                )
            method_manifests[spec["key"]] = str(manifest_path)

        agg_cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "aggregate_full_2x2_eval.py"),
            "--out-csv", str(scene_out / "compare_full.csv"),
            "--out-md", str(scene_out / "compare_full.md"),
            "--aggregate-csv", str(scene_out / "aggregate_metrics.csv"),
            "--aggregate-md", str(scene_out / "aggregate_metrics.md"),
        ]
        for key in [spec["key"] for spec in METHOD_SPECS]:
            agg_cmd.extend(["--manifest", f"{key}={method_manifests[key]}"])
        _run_command(agg_cmd)

        run_config["scenes"].append({
            "scene_id": scene_id,
            "scene_name": scene_name,
            "queries_jsonl": str(copied_queries),
            "method_manifests": method_manifests,
            "outputs": {
                "compare_full_csv": str(scene_out / "compare_full.csv"),
                "compare_full_md": str(scene_out / "compare_full.md"),
                "aggregate_metrics_csv": str(scene_out / "aggregate_metrics.csv"),
                "aggregate_metrics_md": str(scene_out / "aggregate_metrics.md"),
            },
        })

    config_path = args.out / "config.json"
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"Wrote {config_path}")


if __name__ == "__main__":
    main()
