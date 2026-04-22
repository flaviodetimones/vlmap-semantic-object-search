#!/usr/bin/env python3
"""
Aggregate the full 2x2 online evaluation:

  Ob_Hb, Oe_Hb, Ob_Hp, Oe_Hp

from four manifest.json files produced by run_nav_eval.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tools.compare_nav_runs import parse_manifest
from tools.eval_methods import METHOD_SPECS, method_key


EXPECTED_KEYS = [spec["key"] for spec in METHOD_SPECS]


def _split_pipe(value: str) -> List[str]:
    if not value:
        return []
    return [chunk for chunk in str(value).split("|") if chunk]


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _parse_eval_summary(row: dict) -> dict:
    return row.get("eval_summary") or {}


def _target_summary_for_row(row: dict) -> dict:
    payload = _parse_eval_summary(row)
    target_summaries = payload.get("target_summaries") or {}
    target_label = str(row.get("target_label", "")).strip()
    if target_label in target_summaries:
        return target_summaries[target_label]

    target_label_l = target_label.lower()
    for key, value in target_summaries.items():
        if str(key).lower() == target_label_l:
            return value

    if len(target_summaries) == 1:
        return next(iter(target_summaries.values()))

    targets = payload.get("targets") or []
    for key in targets:
        if key in target_summaries:
            return target_summaries[key]
    return {}


def _compute_query_metrics(row: dict) -> dict:
    query_type = str(row.get("query_type", "")).strip().lower() or "object"
    expected_rooms = _split_pipe(row.get("expected_rooms", ""))
    tags = _split_pipe(row.get("tags", ""))

    eval_summary = _parse_eval_summary(row)
    target_summary = _target_summary_for_row(row)
    visit_history = list(target_summary.get("visit_history") or [])
    found = bool(target_summary.get("found", row.get("found", False)) or row.get("found", False))
    final_room = str(eval_summary.get("final_room") or row.get("final_room", "")).strip()
    expected_room_hit = bool(expected_rooms) and final_room in expected_rooms
    visited_expected_room = bool(expected_rooms) and (
        any(room in expected_rooms for room in visit_history) or expected_room_hit
    )

    tried = int(target_summary.get("total_candidates_tried", 0) or 0)
    confirmed = int(target_summary.get("total_candidates_confirmed", 0) or 0)
    wrong_visits = max(tried - confirmed, 0)
    room_transitions = int(
        target_summary.get("room_transitions", max(len(visit_history) - 1, 0)) or 0
    )

    cfr = float("nan")
    ct2r = float("nan")
    rooms_before_success = float("nan")
    final_room_accuracy = float("nan")
    if expected_rooms:
        cfr = float(bool(visit_history) and visit_history[0] in expected_rooms)
        ct2r = float(any(room in expected_rooms for room in visit_history[:2]))
        first_hit_idx = next((idx for idx, room in enumerate(visit_history) if room in expected_rooms), None)
        rooms_before_success = float(first_hit_idx if first_hit_idx is not None else len(visit_history))
        final_room_accuracy = float(expected_room_hit)

    inspections_before_success = float("nan")
    if query_type in {"object", "room_object"}:
        inspections_before_success = float(max(tried - confirmed, 0) if found and confirmed > 0 else tried)

    if query_type == "room":
        success = expected_room_hit
    elif query_type == "room_object":
        success = found and visited_expected_room
    elif query_type == "compound":
        success = found or expected_room_hit
    else:
        success = found

    return {
        "query_type": query_type,
        "expected_rooms": expected_rooms,
        "tags": tags,
        "success": float(bool(success)),
        "object_success": float(bool(success)) if query_type in {"object", "room_object"} else float("nan"),
        "room_success_metric": float(bool(success)) if query_type == "room" else float("nan"),
        "found": float(found),
        "final_room": final_room,
        "final_room_accuracy": final_room_accuracy,
        "cfr": cfr,
        "ct2r": ct2r,
        "rooms_before_success": rooms_before_success,
        "inspections_before_success": inspections_before_success,
        "wrong_visits": float(wrong_visits) if query_type in {"object", "room_object"} else float("nan"),
        "pose_updates": float(row.get("pose_updates", 0) or 0),
        "early_stop": float(bool(row.get("path_stopped_early", False) or row.get("preview_no_action", False))),
        "preview_no_action": float(bool(row.get("preview_no_action", False))),
        "path_stopped_early": float(bool(row.get("path_stopped_early", False))),
        "executor_actions": float(row.get("executor_actions", 0) or 0),
        "room_transitions": float(room_transitions),
        "visited_expected_room": float(visited_expected_room) if expected_rooms else float("nan"),
    }


def _aggregate_slice(rows: List[dict]) -> dict:
    return {
        "n_queries": len(rows),
        "sr": _safe_mean(r["success"] for r in rows),
        "object_sr": _safe_mean(r["object_success"] for r in rows),
        "room_sr": _safe_mean(r["room_success_metric"] for r in rows),
        "cfr": _safe_mean(r["cfr"] for r in rows),
        "ct2r": _safe_mean(r["ct2r"] for r in rows),
        "rooms_before_success": _safe_mean(r["rooms_before_success"] for r in rows),
        "inspections_before_success": _safe_mean(r["inspections_before_success"] for r in rows),
        "wrong_visits": _safe_mean(r["wrong_visits"] for r in rows),
        "mean_pose_updates": _safe_mean(r["pose_updates"] for r in rows),
        "early_stop_rate": _safe_mean(r["early_stop"] for r in rows),
        "final_room_accuracy": _safe_mean(r["final_room_accuracy"] for r in rows),
        "preview_no_action_rate": _safe_mean(r["preview_no_action"] for r in rows),
        "path_stopped_early_rate": _safe_mean(r["path_stopped_early"] for r in rows),
        "mean_executor_actions": _safe_mean(r["executor_actions"] for r in rows),
        "mean_room_transitions": _safe_mean(r["room_transitions"] for r in rows),
    }


def _fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.4f}"


def _load_method_rows(manifest_path: Path) -> List[dict]:
    rows = parse_manifest(manifest_path)
    for row in rows:
        row["method_key"] = method_key(row.get("entrypoint", ""), row.get("heatmap_mode", ""))
        row["metrics"] = _compute_query_metrics(row)
    return rows


def _build_compare_rows(rows_by_method: Dict[str, List[dict]]) -> List[dict]:
    all_ids: List[str] = []
    for rows in rows_by_method.values():
        for row in rows:
            qid = row["id"]
            if qid not in all_ids:
                all_ids.append(qid)

    per_method = {key: {row["id"]: row for row in rows} for key, rows in rows_by_method.items()}
    compare_rows: List[dict] = []
    for idx, qid in enumerate(all_ids, 1):
        row0 = next((per_method[k][qid] for k in EXPECTED_KEYS if qid in per_method[k]), {})
        record = {
            "idx": idx,
            "id": qid,
            "query": row0.get("query", ""),
            "query_type": row0.get("query_type", ""),
            "target_label": row0.get("target_label", ""),
            "expected_rooms": row0.get("expected_rooms", ""),
            "tags": row0.get("tags", ""),
        }
        for key in EXPECTED_KEYS:
            row = per_method.get(key, {}).get(qid, {})
            metrics = row.get("metrics", {})
            record[f"{key}_success"] = metrics.get("success", float("nan"))
            record[f"{key}_found"] = row.get("found", False)
            record[f"{key}_final_room"] = metrics.get("final_room", "")
            record[f"{key}_final_room_accuracy"] = metrics.get("final_room_accuracy", float("nan"))
            record[f"{key}_cfr"] = metrics.get("cfr", float("nan"))
            record[f"{key}_ct2r"] = metrics.get("ct2r", float("nan"))
            record[f"{key}_rooms_before_success"] = metrics.get("rooms_before_success", float("nan"))
            record[f"{key}_inspections_before_success"] = metrics.get("inspections_before_success", float("nan"))
            record[f"{key}_wrong_visits"] = metrics.get("wrong_visits", float("nan"))
            record[f"{key}_pose_updates"] = metrics.get("pose_updates", 0)
            record[f"{key}_early_stop"] = metrics.get("early_stop", float("nan"))
            record[f"{key}_executor_actions"] = metrics.get("executor_actions", float("nan"))
        compare_rows.append(record)
    return compare_rows


def _build_aggregate_rows(rows_by_method: Dict[str, List[dict]]) -> List[dict]:
    aggregate_rows: List[dict] = []
    for key in EXPECTED_KEYS:
        method_rows = rows_by_method.get(key, [])
        if not method_rows:
            continue
        enriched = [r["metrics"] | {"query_type": r["query_type"], "tags": _split_pipe(r.get("tags", ""))} for r in method_rows]

        slices = [("global", "all", enriched)]
        query_types = sorted({r["query_type"] for r in enriched if r.get("query_type")})
        for qt in query_types:
            slices.append(("query_type", qt, [r for r in enriched if r.get("query_type") == qt]))
        tags = sorted({tag for r in enriched for tag in r.get("tags", [])})
        for tag in tags:
            slices.append(("tag", tag, [r for r in enriched if tag in r.get("tags", [])]))

        for slice_kind, slice_value, slice_rows in slices:
            agg = _aggregate_slice(slice_rows)
            aggregate_rows.append({
                "method": key,
                "slice_kind": slice_kind,
                "slice_value": slice_value,
                **agg,
            })
    return aggregate_rows


def _write_csv(rows: List[dict], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_compare_md(rows: List[dict], out_path: Path) -> None:
    lines = [
        "# Full 2x2 per-query comparison",
        "",
        "| # | query | type | Ob_Hb | Oe_Hb | Ob_Hp | Oe_Hp |",
        "|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        statuses = []
        for key in EXPECTED_KEYS:
            success = row.get(f"{key}_success")
            final_room = row.get(f"{key}_final_room") or "-"
            pose = row.get(f"{key}_pose_updates", 0)
            state = "ok" if success == 1.0 else ("fail" if success == 0.0 else "n/a")
            statuses.append(f"{state} / {final_room} / {pose:.0f}")
        lines.append(
            f"| {row['idx']} | {row['query']} | {row['query_type']} | "
            + " | ".join(statuses)
            + " |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_aggregate_md(rows: List[dict], out_path: Path) -> None:
    global_rows = [r for r in rows if r["slice_kind"] == "global"]
    query_type_rows = [r for r in rows if r["slice_kind"] == "query_type"]
    tag_rows = [r for r in rows if r["slice_kind"] == "tag"]

    lines = [
        "# Full 2x2 aggregate metrics",
        "",
        "## Main table",
        "",
        "| method | SR | Object SR | CFR | Rooms Before Success | Inspections Before Success | Wrong Visits | Mean Pose Updates | Early Stop Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in global_rows:
        lines.append(
            f"| {row['method']} | {_fmt(row['sr'])} | {_fmt(row['object_sr'])} | "
            f"{_fmt(row['cfr'])} | {_fmt(row['rooms_before_success'])} | "
            f"{_fmt(row['inspections_before_success'])} | {_fmt(row['wrong_visits'])} | "
            f"{_fmt(row['mean_pose_updates'])} | {_fmt(row['early_stop_rate'])} |"
        )

    lines += [
        "",
        "## Additional metrics",
        "",
        "| method | Room SR | CT2R | Final Room Acc. | Preview No Action | Path Stopped Early | Mean Executor Actions | Mean Room Transitions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in global_rows:
        lines.append(
            f"| {row['method']} | {_fmt(row['room_sr'])} | {_fmt(row['ct2r'])} | "
            f"{_fmt(row['final_room_accuracy'])} | {_fmt(row['preview_no_action_rate'])} | "
            f"{_fmt(row['path_stopped_early_rate'])} | {_fmt(row['mean_executor_actions'])} | "
            f"{_fmt(row['mean_room_transitions'])} |"
        )

    if query_type_rows:
        lines += ["", "## By query_type", ""]
        current_method = None
        for row in query_type_rows:
            if row["method"] != current_method:
                current_method = row["method"]
                lines += [
                    f"### {current_method}",
                    "",
                    "| query_type | n | SR | Object SR | Room SR | CFR | Early Stop |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            lines.append(
                f"| {row['slice_value']} | {row['n_queries']} | {_fmt(row['sr'])} | "
                f"{_fmt(row['object_sr'])} | {_fmt(row['room_sr'])} | "
                f"{_fmt(row['cfr'])} | {_fmt(row['early_stop_rate'])} |"
            )

    if tag_rows:
        lines += ["", "## By tag", ""]
        current_method = None
        for row in tag_rows:
            if row["method"] != current_method:
                current_method = row["method"]
                lines += [
                    f"### {current_method}",
                    "",
                    "| tag | n | SR | CFR | Mean Pose Updates | Early Stop |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            lines.append(
                f"| {row['slice_value']} | {row['n_queries']} | {_fmt(row['sr'])} | "
                f"{_fmt(row['cfr'])} | {_fmt(row['mean_pose_updates'])} | {_fmt(row['early_stop_rate'])} |"
            )

    lines += [
        "",
        "Notes:",
        "- `Rooms Before Success` = index of the first expected room in `visit_history` (0 means first room was already correct).",
        "- `Inspections Before Success` is approximated as `candidates_tried - candidates_confirmed` when the query succeeds.",
        "- `Wrong Visits` = `candidates_tried - candidates_confirmed`; reliable for object-like queries, undefined for room-only queries.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", default=[],
                        help="Method manifest in the form KEY=/path/to/manifest.json .")
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--aggregate-csv", required=True, type=Path)
    parser.add_argument("--aggregate-md", required=True, type=Path)
    args = parser.parse_args()

    manifests: Dict[str, Path] = {}
    for item in args.manifest:
        if "=" not in item:
            raise SystemExit(f"Bad --manifest entry '{item}', expected KEY=PATH")
        key, raw_path = item.split("=", 1)
        key = key.strip()
        if key not in EXPECTED_KEYS:
            raise SystemExit(f"Unknown method key '{key}'")
        manifests[key] = Path(raw_path)

    missing = [key for key in EXPECTED_KEYS if key not in manifests]
    if missing:
        raise SystemExit(f"Missing manifests for: {', '.join(missing)}")

    rows_by_method = {key: _load_method_rows(path) for key, path in manifests.items()}
    compare_rows = _build_compare_rows(rows_by_method)
    aggregate_rows = _build_aggregate_rows(rows_by_method)

    _write_csv(compare_rows, args.out_csv)
    _write_compare_md(compare_rows, args.out_md)
    _write_csv(aggregate_rows, args.aggregate_csv)
    _write_aggregate_md(aggregate_rows, args.aggregate_md)
    print(f"Wrote CSV: {args.out_csv}")
    print(f"Wrote MD:  {args.out_md}")
    print(f"Wrote CSV: {args.aggregate_csv}")
    print(f"Wrote MD:  {args.aggregate_md}")


if __name__ == "__main__":
    main()
