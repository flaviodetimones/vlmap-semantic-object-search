#!/usr/bin/env python3
"""
Build open-vocabulary evaluation JSONLs from real HSSD scene objects.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "tools" / "eval_queries"
DEFAULT_OUT_DIR = DEFAULT_SOURCE_DIR / "small_objects"
DEFAULT_DATASET_ROOT = Path("/home/mario/tfg/data/versioned_data/hssd-hab")
DEFAULT_TARGETS = [
    "bottle",
    "laptop",
    "mug",
    "toaster",
    "teapot",
    "book",
]

_SCENE_CATEGORY_ALIASES = {
    "bottle": {"bottle", "spray_bottle"},
    "laptop": {"laptop"},
    "mug": {"mug", "cup"},
    "toaster": {"toaster"},
    "teapot": {"teapot"},
    "book": {"book"},
}


def _normalize_label(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower())
    return re.sub(r"_+", "_", text).strip("_")


def _strip_scene_variant(scene_name: str) -> str:
    return re.sub(r"_[0-9]+$", "", scene_name)


def _iter_scene_jsonls(source_dir: Path) -> Iterable[Path]:
    for path in sorted(source_dir.glob("*.jsonl")):
        if not path.name.startswith("."):
            yield path


def _scene_instance_path(dataset_root: Path, base_scene: str) -> Path:
    preferred = dataset_root / "scenes" / f"{base_scene}.scene_instance.json"
    if preferred.exists():
        return preferred
    fallback = dataset_root / "scenes-articulated" / f"{base_scene}.scene_instance.json"
    return fallback


def _scene_metadata_from_jsonl(path: Path) -> Optional[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            return {
                "scene_id": int(payload["scene_id"]),
                "scene_name": str(payload["scene_name"]),
            }
    return None


def _load_semantic_category_map(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as fh:
        rows = csv.DictReader(fh)
        fields = rows.fieldnames or []
        hash_key = fields[0]
        category_key = fields[3]
        return {
            row[hash_key].strip(): _normalize_label(row[category_key])
            for row in rows
            if row.get(hash_key, "").strip() and row.get(category_key, "").strip()
        }


def _load_small_object_category_map(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as fh:
        rows = csv.DictReader(fh)
        return {
            row["id"].strip(): _normalize_label(row["clean_category"])
            for row in rows
            if row.get("id", "").strip() and row.get("clean_category", "").strip()
        }


def _load_rooms(semantic_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    with semantic_path.open("r", encoding="utf-8") as fh:
        sem = json.load(fh)
    rooms: Dict[str, List[Tuple[float, float]]] = {}
    counts: Dict[str, int] = defaultdict(int)
    for region in sem.get("region_annotations", []):
        name = (region.get("name") or region.get("label") or "room").strip().lower()
        idx = counts[name]
        counts[name] += 1
        loop = region.get("poly_loop") or []
        poly_xz = [(float(p[0]), float(p[2])) for p in loop if len(p) >= 3]
        rooms[f"{name}__{idx}"] = poly_xz
    return rooms


def _point_in_polygon(x: float, z: float, poly_xz: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(poly_xz)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, zi = poly_xz[i]
        xj, zj = poly_xz[j]
        if ((zi > z) != (zj > z)) and (x < (xj - xi) * (z - zi) / ((zj - zi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _room_for_xz(x: float, z: float, rooms: Dict[str, List[Tuple[float, float]]]) -> Optional[str]:
    for room_id, poly in rooms.items():
        if _point_in_polygon(x, z, poly):
            return room_id
    return None


def _parse_room_instance_label(label: str) -> Tuple[str, Optional[int]]:
    match = re.match(r"^(.*)\.(\d+)$", str(label or "").strip())
    if not match:
        return str(label or "").strip(), None
    return match.group(1), int(match.group(2))


def _room_ref(room_id: str) -> Tuple[str, int]:
    raw_label, idx = room_id.rsplit("__", 1)
    base, explicit_idx = _parse_room_instance_label(raw_label)
    return base, explicit_idx if explicit_idx is not None else int(idx)


def _room_phrase(room_id: str, family_sizes: Dict[str, int]) -> str:
    base, idx = _room_ref(room_id)
    if family_sizes.get(base, 1) <= 1:
        return base
    if idx <= 0:
        return base
    return f"{base} {idx}"


def _dedupe_tags(tags: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for tag in tags:
        if tag and tag not in seen:
            out.append(tag)
            seen.add(tag)
    return out


def _target_aliases(target: str) -> set[str]:
    normalized = _normalize_label(target)
    aliases = set(_SCENE_CATEGORY_ALIASES.get(normalized, {normalized}))
    aliases.add(normalized)
    return aliases


def _collect_target_hits(
    scene_data: dict,
    category_map: Dict[str, str],
    target: str,
    rooms: Dict[str, List[Tuple[float, float]]],
) -> List[dict]:
    aliases = _target_aliases(target)
    hits: List[dict] = []
    for group in ("object_instances", "articulated_object_instances"):
        for inst in scene_data.get(group, []):
            tid = str(inst.get("template_name", "")).split("/")[-1].strip()
            source_category = category_map.get(tid)
            if source_category not in aliases:
                continue
            translation = [float(v) for v in inst.get("translation", [0.0, 0.0, 0.0])]
            room_id = _room_for_xz(translation[0], translation[2], rooms)
            if room_id is None:
                continue
            hits.append(
                {
                    "template_id": tid,
                    "group": group,
                    "translation": translation,
                    "room_id": room_id,
                    "source_category": source_category,
                }
            )
    hits.sort(key=lambda hit: (hit["room_id"], hit["translation"][0], hit["translation"][2]))
    return hits


def _build_object_entry(scene_meta: dict, target: str, hits: Sequence[dict], qid: str) -> dict:
    expected_rooms = sorted({_room_ref(hit["room_id"])[0] for hit in hits})
    polygons = [{"label": _room_ref(hit["room_id"])[0], "instance_idx": _room_ref(hit["room_id"])[1]} for hit in hits]
    tags = ["open_vocab", "natural_scene_object", "single_object"]
    if len(expected_rooms) > 1:
        tags.append("ambiguous_room")
    if len(polygons) > 1:
        tags.append("multi_instance")
    if any(hit["source_category"] != _normalize_label(target) for hit in hits):
        tags.append("canonical_alias")
    return {
        "scene_id": scene_meta["scene_id"],
        "scene_name": scene_meta["scene_name"],
        "query": target,
        "query_type": "object",
        "target_label": target,
        "expected_rooms": expected_rooms,
        "expected_room_polygons": polygons,
        "tags": _dedupe_tags(tags),
        "id": qid,
    }


def _build_room_object_entries(
    scene_meta: dict,
    target: str,
    hits: Sequence[dict],
    qid_prefix: str,
    family_sizes: Dict[str, int],
) -> List[dict]:
    by_room: Dict[str, List[dict]] = defaultdict(list)
    for hit in hits:
        by_room[hit["room_id"]].append(hit)

    entries: List[dict] = []
    qidx = 1
    for room_id, room_hits in sorted(by_room.items()):
        base, idx = _room_ref(room_id)
        if family_sizes.get(base, 1) > 1:
            continue
        tags = ["open_vocab", "natural_scene_object", "room_guided", "single_room"]
        if len(room_hits) > 1:
            tags.append("multi_instance")
        if any(hit["source_category"] != _normalize_label(target) for hit in room_hits):
            tags.append("canonical_alias")
        entries.append(
            {
                "scene_id": scene_meta["scene_id"],
                "scene_name": scene_meta["scene_name"],
                "query": f"the {target} in the {_room_phrase(room_id, family_sizes)}",
                "query_type": "room_object",
                "target_label": target,
                "expected_rooms": [base],
                "expected_room_polygons": [{"label": base, "instance_idx": idx}],
                "tags": _dedupe_tags(tags),
                "id": f"{qid_prefix}r{qidx:02d}",
            }
        )
        qidx += 1
    return entries


def _build_scene_entries(
    scene_meta: dict,
    dataset_root: Path,
    targets: Sequence[str],
    include_room_object: bool,
) -> Tuple[List[dict], Dict[str, int]]:
    base_scene = _strip_scene_variant(scene_meta["scene_name"])
    scene_path = _scene_instance_path(dataset_root, base_scene)
    semantic_path = dataset_root / "semantics" / "scenes" / f"{base_scene}.semantic_config.json"
    category_map = _load_semantic_category_map(dataset_root / "metadata" / "hssd_obj_semantics_condensed.csv")
    # Augmented open-vocab placements are inserted from object_categories_filtered.csv.
    # Merge that catalog so newly inserted templates such as bottle/basketball
    # resolve to the canonical categories expected by the query builder.
    category_map.update(
        _load_small_object_category_map(dataset_root / "metadata" / "object_categories_filtered.csv")
    )
    scene_data = json.loads(scene_path.read_text(encoding="utf-8"))
    rooms = _load_rooms(semantic_path)
    family_sizes: Dict[str, int] = defaultdict(int)
    for room_id in rooms:
        family_sizes[_room_ref(room_id)[0]] += 1

    entries: List[dict] = []
    counts: Dict[str, int] = {}
    obj_idx = 1
    for target in targets:
        hits = _collect_target_hits(scene_data, category_map, target, rooms)
        counts[target] = len(hits)
        if not hits:
            continue
        qid = f"ov{obj_idx:03d}"
        entries.append(_build_object_entry(scene_meta, target, hits, qid))
        if include_room_object:
            entries.extend(_build_room_object_entries(scene_meta, target, hits, qid, family_sizes))
        obj_idx += 1
    return entries, counts


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--scene-names", default=None,
                        help="Comma-separated scene names; defaults to all source JSONLs.")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                        help="Comma-separated canonical open-vocab targets.")
    parser.add_argument("--include-room-object", action="store_true")
    args = parser.parse_args()

    if args.scene_names:
        chosen = {chunk.strip() for chunk in args.scene_names.split(",") if chunk.strip()}
        source_files = [args.source_dir / f"{scene}.jsonl" for scene in sorted(chosen)]
    else:
        source_files = list(_iter_scene_jsonls(args.source_dir))
    targets = [chunk.strip() for chunk in str(args.targets).split(",") if chunk.strip()]

    for source in source_files:
        if not source.exists():
            raise SystemExit(f"Missing source JSONL: {source}")
        scene_meta = _scene_metadata_from_jsonl(source)
        if not scene_meta:
            raise SystemExit(f"Could not read scene metadata from {source}")
        entries, counts = _build_scene_entries(scene_meta, args.dataset_root, targets, args.include_room_object)
        out_path = args.out_dir / f"{scene_meta['scene_name']}.jsonl"
        _write_jsonl(out_path, entries)
        present = {k: v for k, v in counts.items() if v > 0}
        missing = sorted([k for k, v in counts.items() if v <= 0])
        print(
            f"[open-vocab] {scene_meta['scene_name']} -> {len(entries)} queries "
            f"({len(present)}/{len(targets)} targets present)"
        )
        if present:
            print("  present:", ", ".join(f"{k}={v}" for k, v in sorted(present.items())))
        if missing:
            print("  missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
