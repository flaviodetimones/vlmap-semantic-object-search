from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tools.aggregate_full_2x2_eval import (
    _build_aggregate_rows,
    _build_compare_rows,
    _load_method_rows,
)
from tools.eval_methods import method_key


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(tmp_path: Path, key: str, entrypoint: str, heatmap_mode: str, queries: list[dict]) -> Path:
    method_dir = tmp_path / key
    seg_dir = method_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "entrypoint": entrypoint,
        "heatmap_mode": heatmap_mode,
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
        manifest["queries"].append({
            "id": q["id"],
            "query": q["query"],
            "query_type": q["query_type"],
            "target_label": q["target_label"],
            "expected_rooms": q["expected_rooms"],
            "expected_room_polygons": [],
            "tags": q["tags"],
            "segment_path": f"segments/{q['id']}.log",
            "segment_chars": len(q["segment"]),
        })

    manifest_path = method_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def test_method_key_names():
    assert method_key("baseline", "baseline") == "Ob_Hb"
    assert method_key("executor", "baseline") == "Oe_Hb"
    assert method_key("baseline", "postprocessed") == "Ob_Hp"
    assert method_key("executor", "postprocessed") == "Oe_Hp"


def test_aggregate_full_2x2_metrics_and_slices(tmp_path: Path):
    q1_eval = {
        "instruction": "bed",
        "targets": ["bed"],
        "final_room": "bedroom",
        "heatmap_mode": "baseline",
        "target_summaries": {
            "bed": {
                "target": "bed",
                "found": True,
                "current_room": "bedroom",
                "visit_history": ["hallway", "bedroom"],
                "room_transitions": 1,
                "total_candidates_tried": 2,
                "total_candidates_confirmed": 1,
                "wrong_visits": 1,
                "action_log": [],
                "visited_cells_count": 25,
                "rooms": {},
            }
        },
    }
    q2_eval = {
        "instruction": "kitchen",
        "targets": ["kitchen"],
        "final_room": "kitchen",
        "heatmap_mode": "baseline",
        "target_summaries": {
            "kitchen": {
                "target": "kitchen",
                "found": False,
                "current_room": "kitchen",
                "visit_history": ["entryway", "kitchen"],
                "room_transitions": 1,
                "total_candidates_tried": 0,
                "total_candidates_confirmed": 0,
                "wrong_visits": 0,
                "action_log": [],
                "visited_cells_count": 18,
                "rooms": {},
            }
        },
    }

    queries_by_method = {
        "Ob_Hb": [
            {
                "id": "q001",
                "query": "bed",
                "query_type": "object",
                "target_label": "bed",
                "expected_rooms": ["bedroom"],
                "tags": ["single_object", "multi_instance"],
                "segment": "FOUND: bed\n[eval-summary] " + json.dumps(q1_eval),
            },
            {
                "id": "q002",
                "query": "kitchen",
                "query_type": "room",
                "target_label": "kitchen",
                "expected_rooms": ["kitchen"],
                "tags": ["single_room"],
                "segment": "[eval-summary] " + json.dumps(q2_eval),
            },
        ],
        "Oe_Hb": [
            {
                "id": "q001",
                "query": "bed",
                "query_type": "object",
                "target_label": "bed",
                "expected_rooms": ["bedroom"],
                "tags": ["single_object", "multi_instance"],
                "segment": "FOUND: bed\n[eval-summary] " + json.dumps(q1_eval),
            },
            {
                "id": "q002",
                "query": "kitchen",
                "query_type": "room",
                "target_label": "kitchen",
                "expected_rooms": ["kitchen"],
                "tags": ["single_room"],
                "segment": "[eval-summary] " + json.dumps(q2_eval),
            },
        ],
        "Ob_Hp": [
            {
                "id": "q001",
                "query": "bed",
                "query_type": "object",
                "target_label": "bed",
                "expected_rooms": ["bedroom"],
                "tags": ["single_object", "multi_instance"],
                "segment": "[eval-summary] " + json.dumps({**q1_eval, "target_summaries": {"bed": {**q1_eval["target_summaries"]["bed"], "found": False}}}),
            },
            {
                "id": "q002",
                "query": "kitchen",
                "query_type": "room",
                "target_label": "kitchen",
                "expected_rooms": ["kitchen"],
                "tags": ["single_room"],
                "segment": "[eval-summary] " + json.dumps(q2_eval),
            },
        ],
        "Oe_Hp": [
            {
                "id": "q001",
                "query": "bed",
                "query_type": "object",
                "target_label": "bed",
                "expected_rooms": ["bedroom"],
                "tags": ["single_object", "multi_instance"],
                "segment": "FOUND: bed\n[eval-summary] " + json.dumps(q1_eval),
            },
            {
                "id": "q002",
                "query": "kitchen",
                "query_type": "room",
                "target_label": "kitchen",
                "expected_rooms": ["kitchen"],
                "tags": ["single_room"],
                "segment": "[eval-summary] " + json.dumps(q2_eval),
            },
        ],
    }

    manifests = {}
    for key, queries in queries_by_method.items():
        entrypoint = "executor" if key.startswith("Oe") else "baseline"
        heatmap_mode = "baseline" if key.endswith("Hb") else "postprocessed"
        manifests[key] = _write_manifest(tmp_path, key, entrypoint, heatmap_mode, queries)

    rows_by_method = {key: _load_method_rows(path) for key, path in manifests.items()}
    compare_rows = _build_compare_rows(rows_by_method)
    aggregate_rows = _build_aggregate_rows(rows_by_method)

    assert len(compare_rows) == 2
    q1 = next(r for r in compare_rows if r["id"] == "q001")
    assert q1["Ob_Hb_success"] == 1.0
    assert q1["Ob_Hp_success"] == 0.0

    global_rows = [r for r in aggregate_rows if r["slice_kind"] == "global"]
    assert {r["method"] for r in global_rows} == {"Ob_Hb", "Oe_Hb", "Ob_Hp", "Oe_Hp"}

    object_slice = next(r for r in aggregate_rows if r["method"] == "Ob_Hb" and r["slice_kind"] == "query_type" and r["slice_value"] == "object")
    assert object_slice["object_sr"] == 1.0

    tag_slice = next(r for r in aggregate_rows if r["method"] == "Ob_Hb" and r["slice_kind"] == "tag" and r["slice_value"] == "multi_instance")
    assert tag_slice["n_queries"] == 1


def test_menu_shell_syntax():
    menu_path = REPO_ROOT / "docker" / "menu.sh"
    proc = subprocess.run(["bash", "-n", str(menu_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
