#!/usr/bin/env python3
"""
Shared helpers for 2x2 evaluation naming and scene/query path resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


METHOD_SPECS = [
    {"key": "Ob_Hb", "entrypoint": "baseline", "heatmap_mode": "baseline"},
    {"key": "Ob_Hp", "entrypoint": "baseline", "heatmap_mode": "postprocessed"},
    {"key": "Oe_Hb", "entrypoint": "executor", "heatmap_mode": "baseline"},
    {"key": "Oe_Hp", "entrypoint": "executor", "heatmap_mode": "postprocessed"},
]


def parse_scene_ids(raw: str) -> List[int]:
    return [int(chunk.strip()) for chunk in str(raw).split(",") if chunk.strip()]


def normalize_room_aware(entrypoint: str, room_aware: str | None) -> str:
    ep = str(entrypoint).strip().lower()
    if room_aware is None:
        return "off"
    raw = str(room_aware).strip().lower()
    if raw in {"1", "true", "on", "yes"}:
        return "on"
    if raw in {"0", "false", "off", "no"}:
        return "off"
    raise ValueError(f"Unsupported room_aware value: {room_aware}")


def method_key(entrypoint: str, heatmap_mode: str, room_aware: str | None = None) -> str:
    ep = str(entrypoint).strip().lower()
    hm = str(heatmap_mode).strip().lower()
    for spec in METHOD_SPECS:
        if spec["entrypoint"] == ep and spec["heatmap_mode"] == hm:
            return spec["key"]
    raise ValueError(
        f"Unsupported method combination: entrypoint={entrypoint}, heatmap_mode={heatmap_mode}"
    )


def method_spec_map() -> Dict[str, Dict[str, str]]:
    return {spec["key"]: spec for spec in METHOD_SPECS}


def scenes_dir(dataset_type: str, data_paths: str) -> Path:
    dataset_type = str(dataset_type).strip().lower()
    data_paths = str(data_paths).strip().lower()
    if dataset_type == "hssd" or data_paths == "hssd":
        return Path("/workspace/data/vlmaps_dataset_hssd")
    return Path("/workspace/data/vlmaps_dataset")


def scene_name_from_id(scene_id: int, dataset_type: str, data_paths: str) -> Optional[str]:
    root = scenes_dir(dataset_type, data_paths)
    if not root.exists():
        return None
    scene_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if scene_id < 0 or scene_id >= len(scene_dirs):
        return None
    return scene_dirs[scene_id].name


def default_queries_path(scene_name: str) -> Path:
    return REPO_ROOT / "tools" / "eval_queries" / f"{scene_name}.jsonl"
