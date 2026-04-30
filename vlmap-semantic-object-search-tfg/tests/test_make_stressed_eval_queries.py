from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.make_stressed_eval_queries import (
    select_heatmap_entries,
    select_orchestrator_entries,
)


def _entry(
    qid: str,
    *,
    query_type: str,
    tags: list[str],
) -> tuple[str, dict]:
    payload = {
        "id": qid,
        "query": qid,
        "query_type": query_type,
        "target_label": "lamp",
        "expected_rooms": ["living room"],
        "expected_room_polygons": [],
        "tags": tags,
        "scene_id": 0,
        "scene_name": "scene0",
    }
    return json.dumps(payload, ensure_ascii=True), payload


def test_heatmap_filter_tags():
    entries = [
        _entry("q1", query_type="object", tags=["multi_instance"]),
        _entry("q2", query_type="object", tags=["ambiguous_room"]),
        _entry("q3", query_type="object", tags=["explicit_instance"]),
        _entry("q4", query_type="object", tags=["single_object"]),
        _entry("q5", query_type="room_object", tags=["room_guided"]),
        _entry("q6", query_type="object", tags=[]),
    ]
    selected = select_heatmap_entries(entries)
    assert [payload["id"] for _, payload in selected] == ["q1", "q2", "q3"]


def test_orchestrator_filter_logic():
    entries = [
        _entry("q1", query_type="room_object", tags=["ambiguous_room"]),
        _entry("q2", query_type="room_object", tags=["multi_instance"]),
        _entry("q3", query_type="room_object", tags=["single_room"]),
        _entry("q4", query_type="object", tags=["multi_instance", "ambiguous_room"]),
        _entry("q5", query_type="object", tags=["multi_instance"]),
        _entry("q6", query_type="object", tags=["ambiguous_room"]),
    ]
    selected = select_orchestrator_entries(entries)
    assert [payload["id"] for _, payload in selected] == ["q1", "q2", "q4"]


def test_script_warns_and_preserves_entries(tmp_path: Path):
    sources = tmp_path / "sources"
    out_heatmap = tmp_path / "heat"
    out_orch = tmp_path / "orch"
    sources.mkdir()

    lines = []
    for idx in range(6):
        raw, _payload = _entry(
            f"q{idx + 1}",
            query_type="room_object" if idx % 2 else "object",
            tags=["ambiguous_room"] if idx < 3 else ["single_object"],
        )
        lines.append(raw)
    (sources / "scene0.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "make_stressed_eval_queries.py"),
        "--sources", str(sources),
        "--out-heatmap", str(out_heatmap),
        "--out-orchestrator", str(out_orch),
        "--scene-names", "scene0",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
    assert "[stress] scene0" in proc.stdout
    assert "[stress][warn]" in proc.stderr

    heat_lines = (out_heatmap / "scene0.jsonl").read_text(encoding="utf-8").strip().splitlines()
    orch_lines = (out_orch / "scene0.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert sorted(heat_lines) == sorted(lines[:3])
    assert orch_lines == [lines[1]]

    heat_payload = json.loads(heat_lines[0])
    assert heat_payload["expected_rooms"] == ["living room"]
    assert heat_payload["scene_name"] == "scene0"
