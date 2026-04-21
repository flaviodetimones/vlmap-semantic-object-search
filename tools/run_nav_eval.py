#!/usr/bin/env python3
"""
Run a JSONL query batch against the baseline OR executor entrypoint and
produce id-tagged per-query log segments + a manifest.

The interactive entrypoints read instructions from stdin via ``input(...)``
in a ``while True`` loop. We pipe the batch in, capture the entire stdout,
then split the output by occurrences of the
``Enter navigation instruction`` prompt. Each segment after the setup
banner corresponds to exactly one query, in submission order.

Usage (inside docker):

  python tools/run_nav_eval.py \\
      --queries tools/eval_queries/102344193_0.jsonl \\
      --entrypoint baseline \\
      --scene-id 0 \\
      --out tools/eval_results/102344193_0/baseline/

  python tools/run_nav_eval.py \\
      --queries tools/eval_queries/102344193_0.jsonl \\
      --entrypoint executor \\
      --scene-id 0 \\
      --out tools/eval_results/102344193_0/executor/

After both have run, feed the manifests to ``compare_nav_runs.py``:

  python tools/compare_nav_runs.py \\
      --baseline-manifest tools/eval_results/102344193_0/baseline/manifest.json \\
      --executor-manifest tools/eval_results/102344193_0/executor/manifest.json \\
      --out-csv  tools/eval_results/102344193_0/compare.csv \\
      --out-md   tools/eval_results/102344193_0/compare.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
VLMAPS_ROOT = REPO_ROOT / "third_party" / "vlmaps"
APP_DIR = VLMAPS_ROOT / "application"

ENTRYPOINTS = {
    "baseline": APP_DIR / "interactive_object_nav.py",
    "executor": APP_DIR / "interactive_object_nav_executor.py",
}

PROMPT_RE = re.compile(r"Enter navigation instruction \(or 'quit'\):")


def load_queries(path: Path) -> List[Dict]:
    queries = []
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[WARN] {path.name}:{i} bad JSON ({exc}); skipped",
                      file=sys.stderr)
                continue
            d.setdefault("id", f"q{i:03d}")
            queries.append(d)
    return queries


def run_entrypoint(
    entrypoint_path: Path,
    overrides: List[str],
    stdin_text: str,
    log_path: Path,
    timeout: int,
) -> int:
    """Spawn the entrypoint, pipe stdin, tee stdout into log_path."""
    env = os.environ.copy()
    # Encourage headless behaviour. Existing safe_imshow swallows window errors,
    # but xkb/qt complaining on stderr is just noise.
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("MPLBACKEND", "Agg")

    cmd = [sys.executable, str(entrypoint_path), *overrides]
    print(f"$ {' '.join(cmd)}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(VLMAPS_ROOT),
            env=env,
            text=True,
            bufsize=1,
        )
        try:
            proc.communicate(input=stdin_text, timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(f"[TIMEOUT] killed after {timeout}s", file=sys.stderr)
            return 124
    return proc.returncode


def split_log_into_segments(log_path: Path, n_queries: int) -> List[str]:
    """Split the captured log on the 'Enter navigation instruction' prompts.

    Returns a list of strings of length n_queries+1:
      - segment[0]: setup banner before the first prompt.
      - segment[i]: text emitted between prompt i and prompt i+1, i.e. the
        portion produced while query i was being processed.

    If the actual number of segments differs (early exit, crash, extra
    prompts), pads or truncates to n_queries+1.
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")
    parts = PROMPT_RE.split(text)
    # parts[0] = setup banner; parts[1..K] = one per prompt seen
    if len(parts) < n_queries + 1:
        # Pad with empties so callers can still index by query order.
        parts.extend([""] * (n_queries + 1 - len(parts)))
    elif len(parts) > n_queries + 1:
        # Trim trailing junk after we sent 'quit'.
        parts = parts[: n_queries + 1]
    return parts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True, type=Path,
                   help="JSONL with one query per line (build_eval_queries format).")
    p.add_argument("--entrypoint", required=True, choices=list(ENTRYPOINTS),
                   help="Which entrypoint to drive: 'baseline' or 'executor'.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output directory. Will hold raw_log.txt, segments/, manifest.json.")
    p.add_argument("--scene-id", type=int, required=True)
    p.add_argument("--dataset-type", default="hssd")
    p.add_argument(
        "--scene-dataset-config-file",
        default="/workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json",
    )
    p.add_argument("--data-paths", default="hssd")
    p.add_argument("--per-query-timeout", type=int, default=180,
                   help="Soft per-query budget (seconds). Total timeout = N * this.")
    p.add_argument("--extra-overrides", nargs="*", default=[],
                   help="Additional Hydra overrides forwarded to the entrypoint.")
    args = p.parse_args()

    queries = load_queries(args.queries)
    if not queries:
        print(f"No queries in {args.queries}; aborting.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(queries)} queries from {args.queries}")

    entrypoint_path = ENTRYPOINTS[args.entrypoint]
    if not entrypoint_path.exists():
        print(f"Entrypoint not found: {entrypoint_path}", file=sys.stderr)
        sys.exit(2)

    args.out.mkdir(parents=True, exist_ok=True)
    seg_dir = args.out / "segments"
    seg_dir.mkdir(exist_ok=True)
    log_path = args.out / "raw_log.txt"

    # Build stdin: one query per line, then 'quit'. Trailing newline matters.
    stdin_text = "\n".join([q["query"] for q in queries] + ["quit"]) + "\n"

    overrides = [
        f"scene_id={args.scene_id}",
        f"dataset_type={args.dataset_type}",
        f"scene_dataset_config_file={args.scene_dataset_config_file}",
        f"data_paths={args.data_paths}",
        *args.extra_overrides,
    ]
    timeout = max(args.per_query_timeout * len(queries), 600)

    t0 = time.time()
    rc = run_entrypoint(entrypoint_path, overrides, stdin_text, log_path, timeout)
    elapsed = time.time() - t0
    print(f"Subprocess finished rc={rc} in {elapsed:.1f}s "
          f"(~{elapsed / max(len(queries), 1):.1f}s/query)")

    segments = split_log_into_segments(log_path, len(queries))
    # segments[0] is the setup banner; queries align with segments[1..N].
    setup_path = args.out / "setup.log"
    setup_path.write_text(segments[0], encoding="utf-8")

    manifest = {
        "entrypoint": args.entrypoint,
        "queries_jsonl": str(args.queries),
        "scene_id": args.scene_id,
        "dataset_type": args.dataset_type,
        "subprocess_returncode": rc,
        "subprocess_elapsed_sec": round(elapsed, 2),
        "n_queries": len(queries),
        "queries": [],
    }
    for i, q in enumerate(queries, 1):
        seg_text = segments[i] if i < len(segments) else ""
        seg_path = seg_dir / f"{q['id']}.log"
        seg_path.write_text(seg_text, encoding="utf-8")
        manifest["queries"].append({
            "id": q["id"],
            "query": q["query"],
            "query_type": q.get("query_type", "object"),
            "target_label": q.get("target_label", q["query"]),
            "expected_rooms": q.get("expected_rooms", []),
            "tags": q.get("tags", []),
            "segment_path": str(seg_path.relative_to(args.out)),
            "segment_chars": len(seg_text),
        })

    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}  ({len(queries)} segments under {seg_dir})")
    if rc != 0:
        print(f"[NOTE] subprocess returncode {rc}; segments may be partial.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
