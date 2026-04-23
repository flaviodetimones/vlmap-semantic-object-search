from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.aggregate_full_eval import (
    _build_aggregate_rows,
    _build_compare_rows,
    _load_method_rows,
    _write_aggregate_md,
    build_cross_scene_rows,
)
from tools.eval_methods import METHOD_SPECS, method_key


def _write_manifest(
    tmp_path: Path,
    key: str,
    entrypoint: str,
    heatmap_mode: str,
    room_aware: str,
    queries: list[dict],
) -> Path:
    method_dir = tmp_path / key
    seg_dir = method_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "entrypoint": entrypoint,
        "heatmap_mode": heatmap_mode,
        "room_aware": room_aware,
        "policy_mode": "hybrid" if entrypoint == "executor" else None,
        "scene_id": 0,
        "scene_name": "scene0",
        "queries_jsonl": "/tmp/scene0.jsonl",
        "n_queries": len(queries),
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
                "tags": q["tags"],
                "segment_path": f"segments/{q['id']}.log",
                "segment_chars": len(q["segment"]),
            }
        )

    manifest_path = method_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _base_eval(
    instruction: str,
    target_label: str,
    *,
    found: bool,
    final_room: str,
    visit_history: list[str],
    tried: int,
    confirmed: int,
    pose_updates: int,
) -> tuple[dict, str]:
    payload = {
        "instruction": instruction,
        "targets": [target_label],
        "final_room": final_room,
        "heatmap_mode": "baseline",
        "target_summaries": {
            target_label: {
                "target": target_label,
                "found": found,
                "current_room": final_room,
                "visit_history": visit_history,
                "room_transitions": max(len(visit_history) - 1, 0),
                "total_candidates_tried": tried,
                "total_candidates_confirmed": confirmed,
                "wrong_visits": max(tried - confirmed, 0),
                "action_log": [],
                "visited_cells_count": pose_updates,
                "rooms": {},
            }
        },
    }
    segment = ""
    if found:
        segment += f"FOUND: {target_label}\n"
    segment += "[eval-summary] " + json.dumps(payload)
    return payload, segment


def _scene_payloads() -> dict[str, list[dict]]:
    # q001 is where room-aware should help, q002 is a room query, q003 is a
    # case where postprocessing should help the baseline, and q004 is a stable
    # object query to avoid degenerate slices.
    q1_fail, q1_fail_seg = _base_eval(
        "cabinet",
        "cabinet",
        found=False,
        final_room="hallway",
        visit_history=["hallway", "bathroom"],
        tried=2,
        confirmed=0,
        pose_updates=160,
    )
    q1_ron, q1_ron_seg = _base_eval(
        "cabinet",
        "cabinet",
        found=True,
        final_room="bathroom",
        visit_history=["bathroom"],
        tried=1,
        confirmed=1,
        pose_updates=95,
    )
    q2_room, q2_room_seg = _base_eval(
        "entryway",
        "entryway",
        found=False,
        final_room="entryway",
        visit_history=["entryway"],
        tried=0,
        confirmed=0,
        pose_updates=70,
    )
    q3_fail, q3_fail_seg = _base_eval(
        "lamp",
        "lamp",
        found=False,
        final_room="bedroom",
        visit_history=["bedroom"],
        tried=1,
        confirmed=0,
        pose_updates=110,
    )
    q3_hp, q3_hp_seg = _base_eval(
        "lamp",
        "lamp",
        found=True,
        final_room="bedroom",
        visit_history=["bedroom"],
        tried=1,
        confirmed=1,
        pose_updates=92,
    )
    q4_ok, q4_ok_seg = _base_eval(
        "bed",
        "bed",
        found=True,
        final_room="bedroom",
        visit_history=["hallway", "bedroom"],
        tried=2,
        confirmed=1,
        pose_updates=140,
    )

    return {
        "Ob_Hb_Roff": [
            {
                "id": "q001",
                "query": "cabinet",
                "query_type": "object",
                "target_label": "cabinet",
                "expected_rooms": ["bathroom"],
                "tags": ["ambiguous_room"],
                "segment": q1_fail_seg,
            },
            {
                "id": "q002",
                "query": "entryway",
                "query_type": "room",
                "target_label": "entryway",
                "expected_rooms": ["entryway"],
                "tags": ["single_room"],
                "segment": q2_room_seg,
            },
            {
                "id": "q003",
                "query": "lamp",
                "query_type": "object",
                "target_label": "lamp",
                "expected_rooms": ["bedroom"],
                "tags": ["single_object"],
                "segment": q3_fail_seg,
            },
            {
                "id": "q004",
                "query": "bed",
                "query_type": "object",
                "target_label": "bed",
                "expected_rooms": ["bedroom"],
                "tags": ["single_object", "multi_instance"],
                "segment": q4_ok_seg,
            },
        ],
        "Ob_Hp_Roff": [
            {
                "id": "q001",
                "query": "cabinet",
                "query_type": "object",
                "target_label": "cabinet",
                "expected_rooms": ["bathroom"],
                "tags": ["ambiguous_room"],
                "segment": q1_fail_seg,
            },
            {
                "id": "q002",
                "query": "entryway",
                "query_type": "room",
                "target_label": "entryway",
                "expected_rooms": ["entryway"],
                "tags": ["single_room"],
                "segment": q2_room_seg,
            },
            {
                "id": "q003",
                "query": "lamp",
                "query_type": "object",
                "target_label": "lamp",
                "expected_rooms": ["bedroom"],
                "tags": ["single_object"],
                "segment": q3_hp_seg,
            },
            {
                "id": "q004",
                "query": "bed",
                "query_type": "object",
                "target_label": "bed",
                "expected_rooms": ["bedroom"],
                "tags": ["single_object", "multi_instance"],
                "segment": q4_ok_seg,
            },
        ],
        "Oe_Hb_Roff": [
            {
                "id": "q001",
                "query": "cabinet",
                "query_type": "object",
                "target_label": "cabinet",
                "expected_rooms": ["bathroom"],
                "tags": ["ambiguous_room"],
                "segment": q1_fail_seg,
            },
            {
                "id": "q002",
                "query": "entryway",
                "query_type": "room",
                "target_label": "entryway",
                "expected_rooms": ["entryway"],
                "tags": ["single_room"],
                "segment": q2_room_seg,
            },
            {
                "id": "q003",
                "query": "lamp",
                "query_type": "object",
                "target_label": "lamp",
                "expected_rooms": ["bedroom"],
                "tags": ["single_object"],
                "segment": q3_fail_seg,
            },
            {
                "id": "q004",
                "query": "bed",
                "query_type": "object",
                "target_label": "bed",
                "expected_rooms": ["bedroom"],
                "tags": ["single_object", "multi_instance"],
                "segment": q4_ok_seg,
            },
        ],
        "Oe_Hp_Roff": [
            {
                "id": "q001",
                "query": "cabinet",
                "query_type": "object",
                "target_label": "cabinet",
                "expected_rooms": ["bathroom"],
                "tags": ["ambiguous_room"],
                "segment": q1_fail_seg,
            },
            {
                "id": "q002",
                "query": "entryway",
                "query_type": "room",
                "target_label": "entryway",
                "expected_rooms": ["entryway"],
                "tags": ["single_room"],
                "segment": q2_room_seg,
            },
            {
                "id": "q003",
                "query": "lamp",
                "query_type": "object",
                "target_label": "lamp",
                "expected_rooms": ["bedroom"],
                "tags": ["single_object"],
                "segment": q3_hp_seg,
            },
            {
                "id": "q004",
                "query": "bed",
                "query_type": "object",
                "target_label": "bed",
                "expected_rooms": ["bedroom"],
                "tags": ["single_object", "multi_instance"],
                "segment": q4_ok_seg,
            },
        ],
        "Oe_Hb_Ron": [
            {
                "id": "q001",
                "query": "cabinet",
                "query_type": "object",
                "target_label": "cabinet",
                "expected_rooms": ["bathroom"],
                "tags": ["ambiguous_room"],
                "segment": q1_ron_seg,
            },
            {
                "id": "q002",
                "query": "entryway",
                "query_type": "room",
                "target_label": "entryway",
                "expected_rooms": ["entryway"],
                "tags": ["single_room"],
                "segment": q2_room_seg,
            },
            {
                "id": "q003",
                "query": "lamp",
                "query_type": "object",
                "target_label": "lamp",
                "expected_rooms": ["bedroom"],
                "tags": ["single_object"],
                "segment": q3_fail_seg,
            },
            {
                "id": "q004",
                "query": "bed",
                "query_type": "object",
                "target_label": "bed",
                "expected_rooms": ["bedroom"],
                "tags": ["single_object", "multi_instance"],
                "segment": q4_ok_seg,
            },
        ],
        "Oe_Hp_Ron": [
            {
                "id": "q001",
                "query": "cabinet",
                "query_type": "object",
                "target_label": "cabinet",
                "expected_rooms": ["bathroom"],
                "tags": ["ambiguous_room"],
                "segment": q1_ron_seg,
            },
            {
                "id": "q002",
                "query": "entryway",
                "query_type": "room",
                "target_label": "entryway",
                "expected_rooms": ["entryway"],
                "tags": ["single_room"],
                "segment": q2_room_seg,
            },
            {
                "id": "q003",
                "query": "lamp",
                "query_type": "object",
                "target_label": "lamp",
                "expected_rooms": ["bedroom"],
                "tags": ["single_object"],
                "segment": q3_hp_seg,
            },
            {
                "id": "q004",
                "query": "bed",
                "query_type": "object",
                "target_label": "bed",
                "expected_rooms": ["bedroom"],
                "tags": ["single_object", "multi_instance"],
                "segment": q4_ok_seg,
            },
        ],
    }


def test_method_key_names_and_invalid_room_aware():
    assert method_key("baseline", "baseline", "off") == "Ob_Hb_Roff"
    assert method_key("baseline", "postprocessed", "off") == "Ob_Hp_Roff"
    assert method_key("executor", "baseline", "off") == "Oe_Hb_Roff"
    assert method_key("executor", "postprocessed", "off") == "Oe_Hp_Roff"
    assert method_key("executor", "baseline", "on") == "Oe_Hb_Ron"
    assert method_key("executor", "postprocessed", "on") == "Oe_Hp_Ron"

    with pytest.raises(ValueError, match="baseline does not implement room-aware mode"):
        method_key("baseline", "baseline", "on")
    with pytest.raises(ValueError, match="baseline does not implement room-aware mode"):
        method_key("baseline", "postprocessed", "on")


def test_aggregate_full_metrics_contrasts_and_cross_scene(tmp_path: Path):
    queries_by_method = _scene_payloads()
    manifests = {}
    spec_by_key = {spec["key"]: spec for spec in METHOD_SPECS}
    for key, queries in queries_by_method.items():
        spec = spec_by_key[key]
        manifests[key] = _write_manifest(
            tmp_path,
            key,
            spec["entrypoint"],
            spec["heatmap_mode"],
            spec["room_aware"],
            queries,
        )

    rows_by_method = {key: _load_method_rows(path) for key, path in manifests.items()}
    compare_rows = _build_compare_rows(rows_by_method)
    aggregate_rows = _build_aggregate_rows(rows_by_method)

    assert len(compare_rows) == 4
    q1 = next(r for r in compare_rows if r["id"] == "q001")
    assert q1["Oe_Hb_Roff_success"] == 0.0
    assert q1["Oe_Hb_Ron_success"] == 1.0
    assert q1["Ob_Hp_Roff_success"] == 0.0

    global_rows = [r for r in aggregate_rows if r["slice_kind"] == "global"]
    assert {r["method"] for r in global_rows} == {spec["key"] for spec in METHOD_SPECS}

    ron = next(r for r in global_rows if r["method"] == "Oe_Hb_Ron")
    roff = next(r for r in global_rows if r["method"] == "Oe_Hb_Roff")
    assert ron["sr"] > roff["sr"]
    assert ron["object_sr"] > roff["object_sr"]

    object_slice = next(
        r
        for r in aggregate_rows
        if r["method"] == "Oe_Hb_Ron" and r["slice_kind"] == "query_type" and r["slice_value"] == "object"
    )
    assert object_slice["n_queries"] == 3

    tag_slice = next(
        r
        for r in aggregate_rows
        if r["method"] == "Oe_Hb_Ron" and r["slice_kind"] == "tag" and r["slice_value"] == "ambiguous_room"
    )
    assert tag_slice["sr"] == 1.0

    md_path = tmp_path / "aggregate_metrics.md"
    _write_aggregate_md(aggregate_rows, md_path)
    md_text = md_path.read_text(encoding="utf-8")
    assert "Action vocabulary (postproc)" in md_text
    assert "Room-aware (raw)" in md_text
    assert "Heatmap postproc (Oe postproc, room-on)" in md_text

    scene_a = {r["method"]: [dict(r)] for r in global_rows}
    scene_b = {}
    for row in global_rows:
        varied = dict(row)
        varied["sr"] = row["sr"] + 0.1
        varied["object_sr"] = row["object_sr"] + 0.1
        varied["mean_pose_updates"] = row["mean_pose_updates"] + 5.0
        scene_b[row["method"]] = [varied]
    combined = {method: scene_a[method] + scene_b[method] for method in scene_a}
    cross_rows = build_cross_scene_rows(combined)
    assert len(cross_rows) == 6
    cross_ron = next(r for r in cross_rows if r["method"] == "Oe_Hp_Ron")
    assert cross_ron["n_scenes"] == 2
    assert cross_ron["sr_std"] > 0.0
    assert cross_ron["mean_pose_updates_std"] > 0.0
    assert pytest.approx(cross_ron["sr_mean"], rel=1e-6) == (
        next(r for r in global_rows if r["method"] == "Oe_Hp_Ron")["sr"] + 0.05
    )


def test_menu_shell_syntax():
    menu_path = REPO_ROOT / "docker" / "menu.sh"
    proc = subprocess.run(["bash", "-n", str(menu_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
