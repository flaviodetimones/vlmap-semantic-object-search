from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_yoloe_thresh_sweep import (
    aggregate_manifest,
    parse_thresholds,
    threshold_suffix,
)


def _segment(
    *,
    instruction: str,
    target: str,
    query_type: str,
    found: bool,
    success_room: str,
    visit_history: list[str],
    tried: int,
    confirmed: int,
) -> str:
    payload = {
        "instruction": instruction,
        "targets": [target],
        "final_room": success_room,
        "target_summaries": {
            target: {
                "target": target,
                "found": found,
                "current_room": success_room,
                "visit_history": visit_history,
                "room_transitions": max(len(visit_history) - 1, 0),
                "total_candidates_tried": tried,
                "total_candidates_confirmed": confirmed,
                "wrong_visits": max(tried - confirmed, 0),
                "action_log": [],
                "visited_cells_count": 0,
                "rooms": {},
            }
        },
    }
    text = ""
    if found:
        text += f"FOUND: {target}\n"
    text += "[eval-summary] " + json.dumps(payload)
    return text


def _write_manifest(tmp_path: Path, queries: list[dict], *, thresh: float = 0.40) -> Path:
    seg_dir = tmp_path / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "entrypoint": "executor",
        "heatmap_mode": "postprocessed",
        "yoloe_conf_thresh": thresh,
        "room_aware": "on",
        "policy_mode": "hybrid",
        "scene_id": 0,
        "scene_name": "scene0",
        "queries_jsonl": "/tmp/scene0.jsonl",
        "queries": [],
    }
    for q in queries:
        seg_path = seg_dir / f"{q['id']}.log"
        seg_path.write_text(q["segment"], encoding="utf-8")
        manifest["queries"].append(
            {
                "id": q["id"],
                "query": q["query"],
                "query_type": q["query_type"],
                "target_label": q["target_label"],
                "expected_rooms": q["expected_rooms"],
                "expected_room_polygons": [],
                "tags": q.get("tags", []),
                "segment_path": f"segments/{q['id']}.log",
                "segment_chars": len(q["segment"]),
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def test_parse_thresholds_sorts_and_deduplicates():
    assert parse_thresholds("0.40,0.30,0.40,0.35") == [0.30, 0.35, 0.40]


def test_threshold_suffix():
    assert threshold_suffix(0.30) == "t030"
    assert threshold_suffix(0.35) == "t035"
    assert threshold_suffix(0.4) == "t040"
    assert threshold_suffix(0.6) == "t060"


def test_aggregate_manifest_with_and_without_room_object(tmp_path: Path):
    manifest = _write_manifest(
        tmp_path / "case1",
        [
            {
                "id": "q001",
                "query": "cabinet",
                "query_type": "object",
                "target_label": "cabinet",
                "expected_rooms": ["bathroom"],
                "segment": _segment(
                    instruction="cabinet",
                    target="cabinet",
                    query_type="object",
                    found=True,
                    success_room="bathroom",
                    visit_history=["bathroom"],
                    tried=1,
                    confirmed=1,
                ),
            },
            {
                "id": "q002",
                "query": "sofa in living room",
                "query_type": "room_object",
                "target_label": "sofa",
                "expected_rooms": ["living room"],
                "segment": _segment(
                    instruction="sofa in living room",
                    target="sofa",
                    query_type="room_object",
                    found=True,
                    success_room="dining room",
                    visit_history=["dining room"],
                    tried=1,
                    confirmed=1,
                ),
            },
        ],
        thresh=0.40,
    )
    row = aggregate_manifest(manifest)
    assert row["thresh"] == 0.40
    assert row["n_queries"] == 2
    assert row["n_room_object"] == 1
    assert row["found_rate"] == 1.0
    assert row["sr"] == 0.5
    assert row["object_sr"] == 0.5
    assert row["wrong_visits"] == 0.0
    assert row["fp_proxy_room_object"] == 1.0

    manifest_no_room_object = _write_manifest(
        tmp_path / "case2",
        [
            {
                "id": "q001",
                "query": "entryway",
                "query_type": "room",
                "target_label": "entryway",
                "expected_rooms": ["entryway"],
                "segment": _segment(
                    instruction="entryway",
                    target="entryway",
                    query_type="room",
                    found=False,
                    success_room="entryway",
                    visit_history=["entryway"],
                    tried=0,
                    confirmed=0,
                ),
            }
        ],
        thresh=0.50,
    )
    row2 = aggregate_manifest(manifest_no_room_object)
    assert row2["thresh"] == 0.50
    assert row2["n_room_object"] == 0
    assert math.isnan(row2["fp_proxy_room_object"])
