from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gen = _load_module("build_open_vocab_eval_queries", "tools/build_open_vocab_eval_queries.py")
ana = _load_module("analyze_open_vocab_eval", "tools/analyze_open_vocab_eval.py")


def test_collect_target_hits_and_entries(tmp_path: Path):
    scene_path = tmp_path / "scene.scene_instance.json"
    semantic_path = tmp_path / "scene.semantic_config.json"
    metadata_path = tmp_path / "hssd_obj_semantics_condensed.csv"

    scene = {
        "object_instances": [
            {"template_name": "T_BOOK", "translation": [1.0, 0.0, 1.0]},
            {"template_name": "T_BOOK2", "translation": [5.0, 0.0, 5.0]},
            {"template_name": "T_PRINTER", "translation": [1.5, 0.0, 1.5]},
        ],
        "articulated_object_instances": [],
    }
    semantic = {
        "region_annotations": [
            {"name": "office", "poly_loop": [[0, 0, 0], [3, 0, 0], [3, 0, 3], [0, 0, 3]]},
            {"name": "living room", "poly_loop": [[4, 0, 4], [6, 0, 4], [6, 0, 6], [4, 0, 6]]},
        ]
    }
    metadata_path.write_text(
        "Object Hash,Articulated,Pickable,Condensed,Primary\n"
        "T_BOOK,No,Yes,books,book\n"
        "T_BOOK2,No,Yes,book,book\n"
        "T_PRINTER,No,Yes,printer,printer\n",
        encoding="utf-8",
    )
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    semantic_path.write_text(json.dumps(semantic), encoding="utf-8")

    category_map = gen._load_semantic_category_map(metadata_path)
    rooms = gen._load_rooms(semantic_path)
    hits = gen._collect_target_hits(scene, category_map, "book", rooms)
    assert len(hits) == 2
    assert {hit["room_id"] for hit in hits} == {"office__0", "living room__0"}

    meta = {"scene_id": 0, "scene_name": "demo_0"}
    entry = gen._build_object_entry(meta, "book", hits, "ov001")
    assert entry["query_type"] == "object"
    assert entry["target_label"] == "book"
    assert "open_vocab" in entry["tags"]
    assert "ambiguous_room" in entry["tags"]
    assert "canonical_alias" in entry["tags"]


def test_aggregate_rows_focuses_metrics_and_sources():
    rows = [
        {
            "scene_name": "s0",
            "method": "Ob_Hp",
            "query_type": "object",
            "target_label": "book",
            "tags": ["open_vocab"],
            "found": 1.0,
            "object_sr": 1.0,
            "wrong_visits": 0.0,
            "mean_pose_updates": 10.0,
            "confirmation_source": "arrival",
            "resolution_source": "llm",
        },
        {
            "scene_name": "s0",
            "method": "Ob_Hp",
            "query_type": "object",
            "target_label": "book",
            "tags": ["open_vocab"],
            "found": 0.0,
            "object_sr": 0.0,
            "wrong_visits": 2.0,
            "mean_pose_updates": 30.0,
            "confirmation_source": "none",
            "resolution_source": "fallback",
        },
    ]
    agg = ana._aggregate_rows(rows, ("method", "target_label"))
    assert len(agg) == 1
    row = agg[0]
    assert row["found_rate"] == 0.5
    assert row["object_sr"] == 0.5
    assert row["wrong_visits"] == 1.0
    assert row["mean_pose_updates"] == 20.0
    assert row["confirm_arrival_rate"] == 0.5
    assert row["resolution_llm_rate"] == 0.5
    assert row["resolution_fallback_rate"] == 0.5
