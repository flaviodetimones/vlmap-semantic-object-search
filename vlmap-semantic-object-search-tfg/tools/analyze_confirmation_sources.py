#!/usr/bin/env python3
"""
Offline analyser for YOLOE confirmation sources.

Reads one or more ``manifest.json`` files (or an ``eval_runs/{stamp}/`` root)
produced by ``run_nav_eval.py`` / ``run_full_eval.py`` and writes a CSV + MD
with, per method:

- total_queries
- found_total (queries where YOLOE confirmed at any stage)
- found_on_arrival_rate
- found_after_turn_to_face_rate
- found_after_centering_rate
- found_after_local_scan_rate
- found_after_alternative_route_rate
- local_confirmation_rate (sum of arrival + turn_to_face + centering + local_scan)

Also splits by ``query_type`` (object / room_object).

Usage examples::

    python tools/analyze_confirmation_sources.py \
        --run /workspace/results/eval_runs/20260424_144522 \
        --out-dir /workspace/results/eval_runs/20260424_144522/confirmation_sources

    python tools/analyze_confirmation_sources.py \
        --manifest Ob_Hb=/path/to/Ob_Hb/manifest.json \
        --manifest Oe_Hp=/path/to/Oe_Hp/manifest.json \
        --out-dir /tmp/conf

The analyser parses the last ``[eval-summary] { ... }`` line of each per-query
log segment. Queries without an eval-summary (crashes, timeouts) are counted
as *not found*.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


EVAL_SUMMARY_RE = re.compile(r"\[eval-summary\]\s*(\{.*\})\s*$")

SOURCE_FIELDS = [
    "found_on_arrival",
    "found_after_turn_to_face",
    "found_after_centering",
    "found_after_local_scan",
    "found_after_alternative_route",
]
LOCAL_SOURCES = {"arrival", "turn_to_face", "centering", "local_scan"}


def _parse_eval_summary_from_segment(segment_path: Path) -> Optional[dict]:
    if not segment_path.exists():
        return None
    last_match: Optional[str] = None
    try:
        with segment_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = EVAL_SUMMARY_RE.search(line.rstrip())
                if m:
                    last_match = m.group(1)
    except OSError:
        return None
    if not last_match:
        return None
    try:
        return json.loads(last_match)
    except json.JSONDecodeError:
        return None


def _extract_target_summary(payload: dict, target: str) -> dict:
    targets = payload.get("target_summaries") or {}
    if target in targets:
        return targets[target]
    # Fallback: pick the first/only target if keys differ in case
    lowered = {str(k).lower(): v for k, v in targets.items()}
    return lowered.get(str(target).lower(), {})


def _iter_manifest_queries(manifest_path: Path) -> Iterable[Tuple[dict, dict]]:
    """Yield (query_meta, target_summary) pairs for a manifest."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    base_dir = Path(manifest_path).parent
    for query in manifest.get("queries", []):
        seg = base_dir / str(query.get("segment_path", ""))
        payload = _parse_eval_summary_from_segment(seg) if seg.name else None
        target = str(query.get("target_label") or query.get("query") or "")
        tsum = _extract_target_summary(payload, target) if payload else {}
        yield query, tsum


def _aggregate(manifests: Dict[str, Path]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return agg[method][slice] = {total, found_total, *_rate, local_conf_rate}.

    Slices: ``all``, ``object``, ``room_object``.
    """
    slices = ["all", "object", "room_object"]
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method, manifest_path in manifests.items():
        counters = {
            sl: {
                "total": 0,
                "found_total": 0,
                **{f: 0 for f in SOURCE_FIELDS},
                "local_confirmation": 0,
                "no_eval_summary": 0,
            }
            for sl in slices
        }
        for query, tsum in _iter_manifest_queries(manifest_path):
            qtype = str(query.get("query_type") or "object").lower()
            buckets = ["all"]
            if qtype in ("object", "room_object"):
                buckets.append(qtype)
            for sl in buckets:
                counters[sl]["total"] += 1
                if not tsum:
                    counters[sl]["no_eval_summary"] += 1
                    continue
                if tsum.get("found"):
                    counters[sl]["found_total"] += 1
                for f in SOURCE_FIELDS:
                    if tsum.get(f):
                        counters[sl][f] += 1
                src = tsum.get("final_confirmation_source")
                if src in LOCAL_SOURCES:
                    counters[sl]["local_confirmation"] += 1
        # Convert to rates
        metrics = {}
        for sl, cnt in counters.items():
            tot = cnt["total"]
            if tot == 0:
                continue
            row: Dict[str, float] = {
                "total_queries": tot,
                "found_total": cnt["found_total"],
                "no_eval_summary": cnt["no_eval_summary"],
            }
            for f in SOURCE_FIELDS:
                row[f + "_rate"] = cnt[f] / tot
            row["local_confirmation_rate"] = cnt["local_confirmation"] / tot
            metrics[sl] = row
        out[method] = metrics
    return out


def _fmt(v: float) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if v != int(v) else f"{int(v)}"
    return str(v)


def _write_csv(agg: Dict[str, Dict[str, Dict[str, float]]], path: Path) -> None:
    cols = [
        "method",
        "slice",
        "total_queries",
        "found_total",
        "no_eval_summary",
        "found_on_arrival_rate",
        "found_after_turn_to_face_rate",
        "found_after_centering_rate",
        "found_after_local_scan_rate",
        "found_after_alternative_route_rate",
        "local_confirmation_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for method in sorted(agg):
            for sl in ("all", "object", "room_object"):
                row = agg[method].get(sl)
                if not row:
                    continue
                w.writerow([method, sl] + [_fmt(row.get(c, "")) for c in cols[2:]])


def _write_md(agg: Dict[str, Dict[str, Dict[str, float]]], path: Path) -> None:
    lines = ["# Confirmation source analysis", ""]
    for sl in ("all", "object", "room_object"):
        methods = [m for m in sorted(agg) if sl in agg[m]]
        if not methods:
            continue
        lines.append(f"## Slice: {sl}")
        lines.append("")
        lines.append(
            "| method | n | found | arrival | turn | centering | local_scan | alt_route | local_conf |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for m in methods:
            r = agg[m][sl]
            lines.append(
                f"| {m} | {_fmt(r['total_queries'])} | "
                f"{_fmt(r['found_total'])} | "
                f"{_fmt(r['found_on_arrival_rate'])} | "
                f"{_fmt(r['found_after_turn_to_face_rate'])} | "
                f"{_fmt(r['found_after_centering_rate'])} | "
                f"{_fmt(r['found_after_local_scan_rate'])} | "
                f"{_fmt(r['found_after_alternative_route_rate'])} | "
                f"{_fmt(r['local_confirmation_rate'])} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _discover_from_run_root(run_root: Path) -> Dict[str, Path]:
    """Find all manifests inside ``eval_runs/{stamp}/pipeline_full/{scene}/{method}/``."""
    out: Dict[str, Path] = {}
    root = run_root / "pipeline_full"
    if not root.exists():
        # maybe the caller passed pipeline_full directly
        root = run_root
    for scene_dir in sorted(root.glob("*/")):
        for method_dir in sorted(scene_dir.glob("*/")):
            m = method_dir / "manifest.json"
            if m.is_file():
                key = f"{scene_dir.name}/{method_dir.name}"
                out[key] = m
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=None,
                   help="Run root, e.g. /workspace/results/eval_runs/20260424_144522")
    p.add_argument("--manifest", action="append", default=[],
                   help="Explicit method=path mapping, repeatable.")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    manifests: Dict[str, Path] = {}
    if args.run:
        manifests.update(_discover_from_run_root(args.run))
    for item in args.manifest:
        if "=" not in item:
            raise SystemExit(f"--manifest expects method=path, got {item!r}")
        k, v = item.split("=", 1)
        manifests[k.strip()] = Path(v.strip())
    if not manifests:
        raise SystemExit("No manifests provided. Use --run or --manifest.")

    agg = _aggregate(manifests)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "confirmation_sources.csv"
    md_path = args.out_dir / "confirmation_sources.md"
    _write_csv(agg, csv_path)
    _write_md(agg, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
