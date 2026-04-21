#!/usr/bin/env python3
"""
Build normalized evaluation query JSONL files for HSSD scenes.

The output is intended to be consumable by both:
1. the online navigation/orchestrator evaluation tooling, and
2. an offline heatmap comparison pipeline.

Each scene gets one JSONL file under tools/eval_queries/{scene_name}.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from hydra import compose, initialize_config_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
VLMAPS_ROOT = REPO_ROOT / "third_party" / "vlmaps"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VLMAPS_ROOT) not in sys.path:
    sys.path.insert(0, str(VLMAPS_ROOT))

CONFIG_DIR = str(VLMAPS_ROOT / "config")
DEFAULT_HSSD_CONFIG = (
    "/workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json"
)

OBJECT_SHARE = 0.60
ROOM_SHARE = 0.20
ROOM_OBJECT_SHARE = 0.15

from tools.nav_batch_queries import _build_present_categories
from vlmaps.robot.habitat_lang_robot import HabitatLanguageRobot
from vlmaps.utils.matterport3d_categories import get_categories
from vlmaps.utils.room_priors import (
    _normalize_room_for_priors,
    compatible_room_types,
    compute_room_priors,
)
from vlmaps.utils.room_provider import _parse_room_instance
from vlmaps.utils.search_state import SearchState


@dataclass(frozen=True)
class RoomRef:
    raw_label: str
    base_label: str
    family_index: int


def _parse_scene_ids(raw: str) -> List[int]:
    return [int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]


def _scene_query_counts(total: int) -> Dict[str, int]:
    object_count = int(round(total * OBJECT_SHARE))
    room_count = int(round(total * ROOM_SHARE))
    room_object_count = int(round(total * ROOM_OBJECT_SHARE))
    compound_count = total - object_count - room_count - room_object_count
    return {
        "object": object_count,
        "room": room_count,
        "room_object": room_object_count,
        "compound": compound_count,
    }


def _build_cfg(scene_id: int, dataset_type: str, data_paths: str, scene_dataset_config: str):
    overrides = [
        f"scene_id={scene_id}",
        f"dataset_type={dataset_type}",
        f"data_paths={data_paths}",
        f"scene_dataset_config_file={scene_dataset_config}",
    ]
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="object_goal_navigation_cfg", overrides=overrides)


def _sorted_family_members(room_names: Sequence[str]) -> Dict[str, List[str]]:
    families: Dict[str, List[str]] = {}
    for room in room_names:
        base, _idx = _parse_room_instance(room)
        families.setdefault(base, []).append(room)

    def _sort_key(room: str):
        _base, idx = _parse_room_instance(room)
        return (0 if idx is None else 1, idx if idx is not None else -1, room.lower())

    return {base: sorted(members, key=_sort_key) for base, members in families.items()}


def _build_room_refs(room_names: Sequence[str]) -> Tuple[Dict[str, RoomRef], Dict[str, List[RoomRef]]]:
    family_members = _sorted_family_members(room_names)
    by_label: Dict[str, RoomRef] = {}
    by_base: Dict[str, List[RoomRef]] = {}
    for base, members in family_members.items():
        refs = []
        for family_index, room in enumerate(members):
            ref = RoomRef(raw_label=room, base_label=base, family_index=family_index)
            by_label[room] = ref
            refs.append(ref)
        by_base[base] = refs
    return by_label, by_base


def _human_room_phrase(room_label: str) -> str:
    base, idx = _parse_room_instance(room_label)
    if idx is None:
        return base
    return f"{base} {idx}"


def _dedupe_tags(tags: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for tag in tags:
        if tag and tag not in seen:
            ordered.append(tag)
            seen.add(tag)
    return ordered


def _repeat_tags(i: int, pool_size: int) -> List[str]:
    return ["repeat_query"] if pool_size > 0 and i >= pool_size else []


def _scene_family_lookup(room_refs_by_base: Dict[str, List[RoomRef]]) -> Dict[str, List[str]]:
    canonical_to_scene_bases: Dict[str, List[str]] = {}
    for base in room_refs_by_base:
        canonical = _normalize_room_for_priors(base)
        canonical_to_scene_bases.setdefault(canonical, [])
        if base not in canonical_to_scene_bases[canonical]:
            canonical_to_scene_bases[canonical].append(base)
    return canonical_to_scene_bases


def _room_polygons_for_room_label(room_label: str, room_ref_by_label: Dict[str, RoomRef]) -> List[dict]:
    ref = room_ref_by_label[room_label]
    return [{"label": ref.base_label, "instance_idx": ref.family_index}]


def _room_tags_for_room_label(room_label: str, room_refs_by_base: Dict[str, List[RoomRef]]) -> List[str]:
    base, idx = _parse_room_instance(room_label)
    tags = ["single_room"]
    if len(room_refs_by_base.get(base, [])) > 1:
        tags.append("multi_instance")
    if idx is not None:
        tags.append("explicit_instance")
    return tags


def _object_expected_room_info(
    obj: str,
    known_rooms: Sequence[str],
    canonical_to_scene_bases: Dict[str, List[str]],
    room_refs_by_base: Dict[str, List[RoomRef]],
) -> Tuple[List[str], List[dict], List[str]]:
    room_scores = compute_room_priors(
        obj,
        list(known_rooms),
        objects_seen_by_room={},
        llm_output={},
        query_type="indirect",
    )
    expected_rooms = sorted(
        {_normalize_room_for_priors(room) for room, score in room_scores.items() if score > 0.05}
    )
    polygons: List[dict] = []
    tags = ["single_object"]

    if len(expected_rooms) > 1:
        tags.append("ambiguous_room")
    if not expected_rooms:
        tags.append("low_prior")
        return expected_rooms, polygons, tags

    for canonical_room in expected_rooms:
        for scene_base in canonical_to_scene_bases.get(canonical_room, []):
            for ref in room_refs_by_base.get(scene_base, []):
                polygons.append({"label": ref.base_label, "instance_idx": ref.family_index})

    if len(polygons) > 1:
        tags.append("multi_instance")
    if any(poly["label"] not in expected_rooms for poly in polygons):
        tags.append("canonical_alias")
    return expected_rooms, polygons, tags


def _compatible_room_object_pairs(
    room_pool: Sequence[str],
    object_pool: Sequence[str],
) -> List[Tuple[str, str, List[str]]]:
    pairs: List[Tuple[str, str, List[str]]] = []
    for room_label in room_pool:
        room_base, _idx = _parse_room_instance(room_label)
        for obj in object_pool:
            room_scores = compute_room_priors(
                obj,
                list(room_pool),
                objects_seen_by_room={},
                llm_output={},
                query_type="indirect",
            )
            compatible = sorted(compatible_room_types(room_scores, threshold=0.05))
            if room_base in compatible:
                pairs.append((room_label, obj, compatible))
    return pairs


def _build_object_queries(
    scene_id: int,
    scene_name: str,
    count: int,
    object_pool: Sequence[str],
    known_rooms: Sequence[str],
    canonical_to_scene_bases: Dict[str, List[str]],
    room_refs_by_base: Dict[str, List[RoomRef]],
) -> List[dict]:
    if not object_pool:
        return []
    obj_cycle = cycle(object_pool)
    records = []
    for i in range(count):
        obj = next(obj_cycle)
        expected_rooms, polygons, tags = _object_expected_room_info(
            obj, known_rooms, canonical_to_scene_bases, room_refs_by_base
        )
        records.append(
            {
                "scene_id": scene_id,
                "scene_name": scene_name,
                "query": obj,
                "query_type": "object",
                "target_label": obj,
                "expected_rooms": expected_rooms,
                "expected_room_polygons": polygons,
                "tags": _dedupe_tags([*tags, *_repeat_tags(i, len(object_pool))]),
            }
        )
    return records


def _build_room_queries(
    scene_id: int,
    scene_name: str,
    count: int,
    room_pool: Sequence[str],
    room_ref_by_label: Dict[str, RoomRef],
    room_refs_by_base: Dict[str, List[RoomRef]],
) -> List[dict]:
    if not room_pool:
        return []
    room_cycle = cycle(room_pool)
    records = []
    for i in range(count):
        room_label = next(room_cycle)
        base, idx = _parse_room_instance(room_label)
        query = _human_room_phrase(room_label) if idx is not None else base
        records.append(
            {
                "scene_id": scene_id,
                "scene_name": scene_name,
                "query": query,
                "query_type": "room",
                "target_label": room_label,
                "expected_rooms": [base],
                "expected_room_polygons": _room_polygons_for_room_label(
                    room_label, room_ref_by_label
                ),
                "tags": _dedupe_tags(
                    [
                        *_room_tags_for_room_label(room_label, room_refs_by_base),
                        *_repeat_tags(i, len(room_pool)),
                    ]
                ),
            }
        )
    return records


def _build_room_object_queries(
    scene_id: int,
    scene_name: str,
    count: int,
    pair_pool: Sequence[Tuple[str, str, List[str]]],
    room_ref_by_label: Dict[str, RoomRef],
    room_refs_by_base: Dict[str, List[RoomRef]],
) -> List[dict]:
    if not pair_pool:
        return []
    templates = [
        "the {obj} in the {room}",
        "a {obj} in the {room}",
        "{obj} in the {room}",
    ]
    pair_cycle = cycle(pair_pool)
    records = []
    for i in range(count):
        room_label, obj, compatible = next(pair_cycle)
        room_base, idx = _parse_room_instance(room_label)
        room_phrase = _human_room_phrase(room_label) if idx is not None else room_base
        records.append(
            {
                "scene_id": scene_id,
                "scene_name": scene_name,
                "query": templates[i % len(templates)].format(obj=obj, room=room_phrase),
                "query_type": "room_object",
                "target_label": obj,
                "expected_rooms": [room_base],
                "expected_room_polygons": _room_polygons_for_room_label(
                    room_label, room_ref_by_label
                ),
                "tags": _dedupe_tags(
                    [
                        "room_guided",
                        *(["ambiguous_room"] if len(compatible) > 1 else []),
                        *_room_tags_for_room_label(room_label, room_refs_by_base),
                        *_repeat_tags(i, len(pair_pool)),
                    ]
                ),
            }
        )
    return records


def _build_compound_queries(
    scene_id: int,
    scene_name: str,
    count: int,
    pair_pool: Sequence[Tuple[str, str, List[str]]],
    room_ref_by_label: Dict[str, RoomRef],
    room_refs_by_base: Dict[str, List[RoomRef]],
) -> List[dict]:
    if not pair_pool:
        return []
    templates = [
        "go to the {room} and find the {obj}",
        "first go to the {room}, then inspect the {obj}",
        "enter the {room} and look for the {obj}",
    ]
    pair_cycle = cycle(pair_pool)
    records = []
    for i in range(count):
        room_label, obj, _compatible = next(pair_cycle)
        room_base, idx = _parse_room_instance(room_label)
        room_phrase = _human_room_phrase(room_label) if idx is not None else room_base
        records.append(
            {
                "scene_id": scene_id,
                "scene_name": scene_name,
                "query": templates[i % len(templates)].format(obj=obj, room=room_phrase),
                "query_type": "compound",
                "target_label": obj,
                "expected_rooms": [room_base],
                "expected_room_polygons": _room_polygons_for_room_label(
                    room_label, room_ref_by_label
                ),
                "tags": _dedupe_tags(
                    [
                        "compound",
                        "multi_step",
                        *_room_tags_for_room_label(room_label, room_refs_by_base),
                        *_repeat_tags(i, len(pair_pool)),
                    ]
                ),
            }
        )
    return records


def _scene_metadata(
    scene_id: int,
    dataset_type: str,
    data_paths: str,
    scene_dataset_config: str,
    min_room_navigable: float,
) -> dict:
    cfg = _build_cfg(scene_id, dataset_type, data_paths, scene_dataset_config)
    robot = HabitatLanguageRobot(cfg)
    if scene_id < 0 or scene_id >= len(robot.vlmaps_data_save_dirs):
        raise IndexError(f"scene_id={scene_id} is out of range for the indexed dataset")
    scene_dir = robot.vlmaps_data_save_dirs[scene_id]
    vlmap_file = scene_dir / "vlmap" / "vlmaps.h5df"
    if not vlmap_file.exists():
        raise FileNotFoundError(
            f"scene_id={scene_id} points to '{scene_dir.name}', but {vlmap_file} is missing. "
            "Generate/index the VLMap first or pick a different scene_id."
        )
    try:
        robot.setup_scene(scene_id)
        robot.map.init_categories(get_categories(dataset_type))

        scene_name = scene_dir.name
        room_provider = getattr(robot, "room_provider", None)
        if room_provider is None or not room_provider.is_available():
            raise RuntimeError(f"Room provider unavailable for scene_id={scene_id}")

        search_state = SearchState("__eval__", room_provider, robot.map.obstacles_map)
        room_stats = search_state.rooms
        all_room_names = list(room_provider.list_rooms())
        room_ref_by_label, room_refs_by_base = _build_room_refs(all_room_names)
        canonical_to_scene_bases = _scene_family_lookup(room_refs_by_base)

        room_pool = [
            room_name
            for room_name, rs in room_stats.items()
            if rs.explored_ratio >= min_room_navigable
        ]
        if not room_pool:
            room_pool = list(all_room_names)

        present_objects = sorted(_build_present_categories(robot))
        if not present_objects:
            structural = {"void", "wall", "floor", "ceiling"}
            present_objects = sorted(
                c for c in getattr(robot.map, "categories", []) if c and c not in structural
            )
        supported_objects = []
        for obj in present_objects:
            room_scores = compute_room_priors(
                obj, list(room_pool), objects_seen_by_room={}, llm_output={}, query_type="indirect"
            )
            if compatible_room_types(room_scores, threshold=0.05):
                supported_objects.append(obj)
        object_pool = supported_objects or present_objects
        if not object_pool:
            object_pool = ["chair", "table", "door"]

        return {
            "scene_name": scene_name,
            "room_pool": room_pool,
            "room_ref_by_label": room_ref_by_label,
            "room_refs_by_base": room_refs_by_base,
            "canonical_to_scene_bases": canonical_to_scene_bases,
            "object_pool": object_pool,
        }
    finally:
        sim = getattr(robot, "sim", None)
        if sim is not None:
            try:
                sim.close()
            except Exception:
                pass


def _assign_ids(records: List[dict]) -> List[dict]:
    for i, record in enumerate(records, start=1):
        record["id"] = f"q{i:03d}"
    return records


def build_scene_queries(
    *,
    scene_id: int,
    dataset_type: str,
    data_paths: str,
    scene_dataset_config: str,
    min_room_navigable: float,
    queries_per_scene: int,
    seed: int,
) -> Tuple[str, List[dict]]:
    rng = np.random.default_rng(seed + scene_id)
    meta = _scene_metadata(
        scene_id, dataset_type, data_paths, scene_dataset_config, min_room_navigable
    )

    scene_name = meta["scene_name"]
    room_pool = list(meta["room_pool"])
    object_pool = list(meta["object_pool"])
    room_ref_by_label = meta["room_ref_by_label"]
    room_refs_by_base = meta["room_refs_by_base"]
    canonical_to_scene_bases = meta["canonical_to_scene_bases"]

    rng.shuffle(room_pool)
    rng.shuffle(object_pool)

    counts = _scene_query_counts(queries_per_scene)
    pair_pool = _compatible_room_object_pairs(room_pool, object_pool)
    rng.shuffle(pair_pool)
    if not pair_pool and room_pool and object_pool:
        pair_pool = [(room_pool[0], object_pool[0], [room_pool[0]])]

    records: List[dict] = []
    records.extend(
        _build_object_queries(
            scene_id,
            scene_name,
            counts["object"],
            object_pool,
            room_pool,
            canonical_to_scene_bases,
            room_refs_by_base,
        )
    )
    records.extend(
        _build_room_queries(
            scene_id,
            scene_name,
            counts["room"],
            room_pool,
            room_ref_by_label,
            room_refs_by_base,
        )
    )
    records.extend(
        _build_room_object_queries(
            scene_id,
            scene_name,
            counts["room_object"],
            pair_pool,
            room_ref_by_label,
            room_refs_by_base,
        )
    )
    records.extend(
        _build_compound_queries(
            scene_id,
            scene_name,
            counts["compound"],
            pair_pool,
            room_ref_by_label,
            room_refs_by_base,
        )
    )
    return scene_name, _assign_ids(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-ids", default="0,1")
    parser.add_argument("--queries-per-scene", type=int, default=50)
    parser.add_argument("--dataset-type", default="hssd")
    parser.add_argument("--data-paths", default="hssd")
    parser.add_argument("--scene-dataset-config-file", default=DEFAULT_HSSD_CONFIG)
    parser.add_argument("--min-room-navigable", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=21042026)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "tools" / "eval_queries"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for scene_id in _parse_scene_ids(args.scene_ids):
        scene_name, records = build_scene_queries(
            scene_id=scene_id,
            dataset_type=args.dataset_type,
            data_paths=args.data_paths,
            scene_dataset_config=args.scene_dataset_config_file,
            min_room_navigable=args.min_room_navigable,
            queries_per_scene=args.queries_per_scene,
            seed=args.seed,
        )
        out_path = output_dir / f"{scene_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[eval-queries] wrote {len(records)} queries -> {out_path}")


if __name__ == "__main__":
    main()
