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
from typing import Dict, List, Optional


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
    env_extra: Optional[Dict[str, str]] = None,
) -> int:
    """Spawn the entrypoint, pipe stdin, tee stdout into log_path."""
    env = os.environ.copy()
    # Batch evaluation must be headless, but interactive menu runs stay unchanged.
    # We avoid forcing a Qt platform plugin here and instead disable all UI
    # drawing inside the navigation scripts via a dedicated env var.
    env.setdefault("VLMAPS_EVAL_HEADLESS", "1")
    env.setdefault("MPLBACKEND", "Agg")
    py_path = str(VLMAPS_ROOT)
    env["PYTHONPATH"] = py_path if not env.get("PYTHONPATH") else f"{py_path}:{env['PYTHONPATH']}"
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items() if v is not None})

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
    p.add_argument("--scene-name", default=None,
                   help="Optional scene name; defaults to the single scene_name present in the JSONL, if any.")
    p.add_argument("--dataset-type", default="hssd")
    p.add_argument(
        "--scene-dataset-config-file",
        default="/workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json",
    )
    p.add_argument("--data-paths", default="hssd")
    p.add_argument("--heatmap-mode", choices=["baseline", "postprocessed"], default="postprocessed")
    p.add_argument("--room-aware", choices=["on", "off"], default="on",
                   help="Executor-only: enable/disable room-aware selection and candidate gating.")
    p.add_argument("--policy-mode", choices=["heuristic", "hybrid", "llm"], default=None,
                   help="Executor-only policy mode, forwarded as VLMAPS_POLICY_MODE.")
    p.add_argument("--yoloe-conf-thresh", type=float, default=0.30,
                   help="YOLOE confidence threshold forwarded to runtime.")
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

    scene_names = sorted({str(q.get("scene_name")) for q in queries if q.get("scene_name")})
    scene_name = args.scene_name
    if scene_name is None and len(scene_names) == 1:
        scene_name = scene_names[0]

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
    effective_room_aware = args.room_aware if args.entrypoint == "executor" else "off"
    env_extra = {
        "VLMAPS_HEATMAP_MODE": args.heatmap_mode,
        "VLMAPS_YOLOE_CONF_THRESH": str(args.yoloe_conf_thresh),
    }
    if args.entrypoint == "executor":
        env_extra["VLMAPS_ROOM_AWARE"] = effective_room_aware
        if args.policy_mode:
            env_extra["VLMAPS_POLICY_MODE"] = args.policy_mode

    t0 = time.time()
    rc = run_entrypoint(
        entrypoint_path,
        overrides,
        stdin_text,
        log_path,
        timeout,
        env_extra=env_extra,
    )
    elapsed = time.time() - t0
    print(f"Subprocess finished rc={rc} in {elapsed:.1f}s "
          f"(~{elapsed / max(len(queries), 1):.1f}s/query)")

    segments = split_log_into_segments(log_path, len(queries))
    # segments[0] is the setup banner; queries align with segments[1..N].
    setup_path = args.out / "setup.log"
    setup_header = (
        f"Entrypoint: {args.entrypoint}\n"
        f"Heatmap mode: {args.heatmap_mode}\n"
        f"YOLOE conf thresh: {args.yoloe_conf_thresh:.2f}\n"
        f"Room-aware: {effective_room_aware}\n"
        f"Policy mode: {args.policy_mode or '-'}\n"
        f"Scene: {scene_name or args.scene_id}\n\n"
    )
    setup_path.write_text(setup_header + segments[0], encoding="utf-8")

    manifest = {
        "entrypoint": args.entrypoint,
        "heatmap_mode": args.heatmap_mode,
        "yoloe_conf_thresh": args.yoloe_conf_thresh,
        "room_aware": effective_room_aware,
        "policy_mode": args.policy_mode,
        "queries_jsonl": str(args.queries),
        "scene_id": args.scene_id,
        "scene_name": scene_name,
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
            "expected_room_polygons": q.get("expected_room_polygons", []),
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
