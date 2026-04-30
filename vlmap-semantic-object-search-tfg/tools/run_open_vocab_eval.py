#!/usr/bin/env python3
"""
Run a short small-object benchmark batch with the current base config.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY_DIR = REPO_ROOT / "tools" / "eval_queries" / "small_objects"


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True).returncode
    if rc != 0:
        raise SystemExit(rc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-ids", default="0,1,2")
    parser.add_argument("--queries", default=str(DEFAULT_QUERY_DIR))
    parser.add_argument("--dataset-type", default="hssd")
    parser.add_argument("--data-paths", default="hssd")
    parser.add_argument(
        "--scene-dataset-config-file",
        default="/workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json",
    )
    parser.add_argument("--policy-mode", default="hybrid", choices=["heuristic", "hybrid", "llm"])
    parser.add_argument("--yoloe-conf-thresh", type=float, default=0.30)
    parser.add_argument("--per-query-timeout", type=int, default=180)
    parser.add_argument("--methods", default="Ob_Hp")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_full_eval.py"),
            "--scene-ids", str(args.scene_ids),
            "--queries", str(args.queries),
            "--dataset-type", str(args.dataset_type),
            "--data-paths", str(args.data_paths),
            "--scene-dataset-config-file", str(args.scene_dataset_config_file),
            "--policy-mode", str(args.policy_mode),
            "--yoloe-conf-thresh", str(args.yoloe_conf_thresh),
            "--per-query-timeout", str(args.per_query_timeout),
            "--methods", str(args.methods),
            "--out", str(args.out),
        ]
    )
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "analyze_open_vocab_eval.py"),
            "--run", str(args.out),
            "--out-dir", str(args.out / "open_vocab_analysis"),
        ]
    )


if __name__ == "__main__":
    main()
