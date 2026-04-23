#!/usr/bin/env python3
"""
Summarise baseline vs executor navigation logs query-by-query.

The script parses the interactive logs produced by:
  - application/interactive_object_nav.py
  - application/interactive_object_nav_executor.py

and emits a CSV plus a compact markdown summary that can be inspected without
manually scrolling through the raw logs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


PROMPT_RE = re.compile(r"Enter navigation instruction \(or 'quit'\):\s*(.*)")
FOUND_RE = re.compile(r"FOUND:\s*(.+)")
ACTUAL_ROOM_RE = re.compile(r"Actual room after path:\s*(.+)")
ROOM_STAGE_RE = re.compile(r"\[room-stage\] Actual room after staging:\s*(.+)")
ARRIVED_ROOM_RE = re.compile(r"Arrived at room '(.+?)'")
ROOM_FAIL_RE = re.compile(r"Room navigation FAILED: requested '(.+?)', ended in '(.+?)'")
EXEC_ROOM_FAIL_RE = re.compile(r"\[executor\] Room navigation failed for '(.+?)'\.")
TARGETS_RE = re.compile(r"Targets:\s*(\[.*\])")
EVAL_SUMMARY_RE = re.compile(r"\[eval-summary\]\s*(\{.*\})")


def _normalize_room(text: str) -> str:
    return str(text or "").strip()


def _new_row() -> dict:
    return {
        "query": "",
        "targets": "",
        "found": False,
        "final_room": "",
        "room_success": False,
        "room_failure": False,
        "path_stopped_early": False,
        "preview_no_action": False,
        "pose_updates": 0,
        "executor_actions": 0,
        "eval_summary": None,
    }


def parse_segment_text(text: str) -> dict:
    """Run the per-line parser on a single id-tagged segment and return one row.

    Used by the manifest path so each query already comes pre-isolated.
    """
    current = _new_row()
    for raw_line in text.splitlines():
        _consume_line(raw_line, current)
    return current


def _consume_line(line: str, current: dict) -> None:
    line = line.rstrip("\n")
    if "set curr pose:" in line:
        current["pose_updates"] += 1

    if "[executor-policy] Planned actions:" in line:
        current["executor_actions"] = 0
        return
    if line.strip().startswith(tuple(f"{i}." for i in range(1, 10))):
        if "[executor-policy]" not in line and current.get("executor_actions") is not None:
            current["executor_actions"] += 1

    m = TARGETS_RE.search(line)
    if m:
        try:
            targets = ast.literal_eval(m.group(1))
            current["targets"] = ", ".join(str(t) for t in targets)
        except Exception:
            current["targets"] = m.group(1)
        return

    m = EVAL_SUMMARY_RE.search(line)
    if m:
        try:
            current["eval_summary"] = json.loads(m.group(1))
        except Exception:
            current["eval_summary"] = None
        return

    if "Path execution stopped early" in line:
        current["path_stopped_early"] = True
    if "Preview produced no action" in line:
        current["preview_no_action"] = True

    m = FOUND_RE.search(line)
    if m:
        current["found"] = True
        return

    m = ACTUAL_ROOM_RE.search(line)
    if m:
        current["final_room"] = _normalize_room(m.group(1))
        return

    m = ROOM_STAGE_RE.search(line)
    if m:
        current["final_room"] = _normalize_room(m.group(1))
        return

    m = ARRIVED_ROOM_RE.search(line)
    if m:
        current["room_success"] = True
        current["final_room"] = _normalize_room(m.group(1))
        return

    m = ROOM_FAIL_RE.search(line)
    if m:
        current["room_failure"] = True
        current["final_room"] = _normalize_room(m.group(2))
        return

    m = EXEC_ROOM_FAIL_RE.search(line)
    if m:
        current["room_failure"] = True
        return


def parse_log(path: Path) -> list[dict]:
    rows: list[dict] = []
    current: dict | None = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            m = PROMPT_RE.search(line)
            if m:
                query = m.group(1).strip()
                if query.lower() in {"quit", "exit", "q"}:
                    continue
                current = {
                    "query": query,
                    "targets": "",
                    "found": False,
                    "final_room": "",
                    "room_success": False,
                    "room_failure": False,
                    "path_stopped_early": False,
                    "preview_no_action": False,
                    "pose_updates": 0,
                    "executor_actions": 0,
                    "eval_summary": None,
                }
                rows.append(current)
                continue

            if current is None:
                continue

            if "set curr pose:" in line:
                current["pose_updates"] += 1

            if "[executor-policy] Planned actions:" in line:
                current["executor_actions"] = 0
                continue
            if line.strip().startswith(tuple(f"{i}." for i in range(1, 10))):
                if "[executor-policy]" not in line and current.get("executor_actions") is not None:
                    current["executor_actions"] += 1

            m = TARGETS_RE.search(line)
            if m:
                try:
                    targets = ast.literal_eval(m.group(1))
                    current["targets"] = ", ".join(str(t) for t in targets)
                except Exception:
                    current["targets"] = m.group(1)
                continue

            m = EVAL_SUMMARY_RE.search(line)
            if m:
                try:
                    current["eval_summary"] = json.loads(m.group(1))
                except Exception:
                    current["eval_summary"] = None
                continue

            if "Path execution stopped early" in line:
                current["path_stopped_early"] = True
            if "Preview produced no action" in line:
                current["preview_no_action"] = True

            m = FOUND_RE.search(line)
            if m:
                current["found"] = True
                continue

            m = ACTUAL_ROOM_RE.search(line)
            if m:
                current["final_room"] = _normalize_room(m.group(1))
                continue

            m = ROOM_STAGE_RE.search(line)
            if m:
                current["final_room"] = _normalize_room(m.group(1))
                continue

            m = ARRIVED_ROOM_RE.search(line)
            if m:
                current["room_success"] = True
                current["final_room"] = _normalize_room(m.group(1))
                continue

            m = ROOM_FAIL_RE.search(line)
            if m:
                current["room_failure"] = True
                current["final_room"] = _normalize_room(m.group(2))
                continue

            m = EXEC_ROOM_FAIL_RE.search(line)
            if m:
                current["room_failure"] = True
                continue

    return rows


def parse_manifest(path: Path) -> List[dict]:
    """Read a manifest.json (produced by run_nav_eval.py) and parse each segment.

    Returns rows with the same fields as parse_log() plus 'id', 'query_type',
    'expected_rooms', 'tags' for downstream id-aligned merging.
    """
    manifest = json.loads(path.read_text(encoding="utf-8"))
    base_dir = path.parent
    entrypoint = manifest.get("entrypoint", "")
    room_aware = manifest.get("room_aware")
    if room_aware is None:
        room_aware = "on" if entrypoint == "executor" else "off"
    rows: List[dict] = []
    for entry in manifest.get("queries", []):
        seg_path = base_dir / entry["segment_path"]
        text = seg_path.read_text(encoding="utf-8", errors="replace") if seg_path.exists() else ""
        row = parse_segment_text(text)
        row["id"] = entry["id"]
        row["query"] = entry.get("query", row.get("query", ""))
        row["query_type"] = entry.get("query_type", "")
        row["target_label"] = entry.get("target_label", row.get("query", ""))
        row["expected_rooms"] = "|".join(entry.get("expected_rooms", []))
        row["expected_room_polygons"] = json.dumps(entry.get("expected_room_polygons", []))
        row["tags"] = "|".join(entry.get("tags", []))
        row["entrypoint"] = manifest.get("entrypoint", "")
        row["heatmap_mode"] = manifest.get("heatmap_mode", "")
        row["yoloe_conf_thresh"] = manifest.get("yoloe_conf_thresh")
        row["room_aware"] = room_aware
        row["policy_mode"] = manifest.get("policy_mode", "")
        row["scene_id"] = manifest.get("scene_id")
        row["scene_name"] = manifest.get("scene_name", "")
        rows.append(row)
    return rows


def compare_rows_by_id(baseline_rows: List[dict], executor_rows: List[dict]) -> List[dict]:
    by_id_b = {r["id"]: r for r in baseline_rows}
    by_id_e = {r["id"]: r for r in executor_rows}
    all_ids = list(dict.fromkeys(list(by_id_b.keys()) + list(by_id_e.keys())))
    merged: List[dict] = []
    for i, qid in enumerate(all_ids, 1):
        b = by_id_b.get(qid, {})
        e = by_id_e.get(qid, {})
        merged.append({
            "idx": i,
            "id": qid,
            "query": b.get("query") or e.get("query", ""),
            "query_type": b.get("query_type") or e.get("query_type", ""),
            "expected_rooms": b.get("expected_rooms") or e.get("expected_rooms", ""),
            "tags": b.get("tags") or e.get("tags", ""),
            "targets": b.get("targets") or e.get("targets", ""),
            "baseline_found": bool(b.get("found", False)),
            "executor_found": bool(e.get("found", False)),
            "baseline_final_room": b.get("final_room", ""),
            "executor_final_room": e.get("final_room", ""),
            "baseline_room_success": bool(b.get("room_success", False)),
            "executor_room_success": bool(e.get("room_success", False)),
            "baseline_path_stopped_early": bool(b.get("path_stopped_early", False)),
            "executor_path_stopped_early": bool(e.get("path_stopped_early", False)),
            "baseline_preview_no_action": bool(b.get("preview_no_action", False)),
            "executor_preview_no_action": bool(e.get("preview_no_action", False)),
            "baseline_pose_updates": b.get("pose_updates", 0),
            "executor_pose_updates": e.get("pose_updates", 0),
            "executor_actions": e.get("executor_actions", 0),
        })
    return merged


def compare_rows(baseline_rows: list[dict], executor_rows: list[dict]) -> list[dict]:
    n = max(len(baseline_rows), len(executor_rows))
    merged = []
    for i in range(n):
        b = baseline_rows[i] if i < len(baseline_rows) else {}
        e = executor_rows[i] if i < len(executor_rows) else {}
        merged.append(
            {
                "idx": i + 1,
                "query": b.get("query") or e.get("query", ""),
                "targets": b.get("targets") or e.get("targets", ""),
                "baseline_found": b.get("found", False),
                "executor_found": e.get("found", False),
                "baseline_final_room": b.get("final_room", ""),
                "executor_final_room": e.get("final_room", ""),
                "baseline_room_success": b.get("room_success", False),
                "executor_room_success": e.get("room_success", False),
                "baseline_path_stopped_early": b.get("path_stopped_early", False),
                "executor_path_stopped_early": e.get("path_stopped_early", False),
                "baseline_preview_no_action": b.get("preview_no_action", False),
                "executor_preview_no_action": e.get("preview_no_action", False),
                "baseline_pose_updates": b.get("pose_updates", 0),
                "executor_pose_updates": e.get("pose_updates", 0),
                "executor_actions": e.get("executor_actions", 0),
            }
        )
    return merged


def write_csv(rows: list[dict], out_csv: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else [
        "idx",
        "query",
        "targets",
        "baseline_found",
        "executor_found",
        "baseline_final_room",
        "executor_final_room",
        "baseline_room_success",
        "executor_room_success",
        "baseline_path_stopped_early",
        "executor_path_stopped_early",
        "baseline_preview_no_action",
        "executor_preview_no_action",
        "baseline_pose_updates",
        "executor_pose_updates",
        "executor_actions",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows: list[dict], out_md: Path) -> None:
    lines = [
        "# Baseline vs Executor",
        "",
        "| # | query | baseline | executor | final room (b/e) | poses (b/e) | notes |",
        "|---:|---|---|---|---|---:|---|",
    ]
    for row in rows:
        b_status = "found" if row["baseline_found"] else ("room-ok" if row["baseline_room_success"] else "fail")
        e_status = "found" if row["executor_found"] else ("room-ok" if row["executor_room_success"] else "fail")
        notes = []
        if row["baseline_path_stopped_early"]:
            notes.append("b:stopped")
        if row["executor_path_stopped_early"]:
            notes.append("e:stopped")
        if row["baseline_preview_no_action"]:
            notes.append("b:preview")
        if row["executor_preview_no_action"]:
            notes.append("e:preview")
        lines.append(
            f"| {row['idx']} | {row['query']} | {b_status} | {e_status} | "
            f"{row['baseline_final_room'] or '-'} / {row['executor_final_room'] or '-'} | "
            f"{row['baseline_pose_updates']} / {row['executor_pose_updates']} | "
            f"{', '.join(notes) or '-'} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", type=Path, default=None,
                        help="(Legacy) raw entrypoint log; pairs queries by order.")
    parser.add_argument("--executor-log", type=Path, default=None,
                        help="(Legacy) raw entrypoint log; pairs queries by order.")
    parser.add_argument("--baseline-manifest", type=Path, default=None,
                        help="manifest.json from run_nav_eval.py --entrypoint baseline.")
    parser.add_argument("--executor-manifest", type=Path, default=None,
                        help="manifest.json from run_nav_eval.py --entrypoint executor.")
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    args = parser.parse_args()

    use_manifests = args.baseline_manifest is not None and args.executor_manifest is not None
    use_logs = args.baseline_log is not None and args.executor_log is not None
    if not (use_manifests or use_logs):
        parser.error("Provide either --baseline/--executor manifest pair "
                     "or --baseline-log/--executor-log pair.")

    if use_manifests:
        baseline_rows = parse_manifest(args.baseline_manifest)
        executor_rows = parse_manifest(args.executor_manifest)
        rows = compare_rows_by_id(baseline_rows, executor_rows)
        baseline_meta = baseline_rows[0] if baseline_rows else {}
        executor_meta = executor_rows[0] if executor_rows else {}
    else:
        baseline_rows = parse_log(args.baseline_log)
        executor_rows = parse_log(args.executor_log)
        rows = compare_rows(baseline_rows, executor_rows)
        baseline_meta = {}
        executor_meta = {}

    write_csv(rows, args.out_csv)
    write_markdown(rows, args.out_md)
    if use_manifests:
        header = (
            f"Baseline: heatmap={baseline_meta.get('heatmap_mode', '-')}, yoloe_conf_thresh={baseline_meta.get('yoloe_conf_thresh', '-')}, room_aware={baseline_meta.get('room_aware', '-')}\n"
            f"Executor: heatmap={executor_meta.get('heatmap_mode', '-')}, yoloe_conf_thresh={executor_meta.get('yoloe_conf_thresh', '-')}, room_aware={executor_meta.get('room_aware', '-')}\n\n"
        )
        args.out_md.write_text(header + args.out_md.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote CSV: {args.out_csv}")
    print(f"Wrote MD:  {args.out_md}")


if __name__ == "__main__":
    main()
