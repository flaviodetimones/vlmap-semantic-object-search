#!/usr/bin/env python3
"""
place_small_objects.py
======================
Insert small (level-3) objects on top of existing furniture in an HSSD
scene, then emit a JSONL of evaluation queries with room ground truth.

Why this exists
---------------
The VLMap built for HSSD covers ~36 furniture categories. Bottles, cups,
laptops, etc. are not represented. The strategic policy (Phase G) handles
this by swapping the heatmap target for a *surrogate furniture* category
(bottle → counter, cup → table, ...). To validate that surrogate path
end-to-end we need scenes that actually contain those small objects in
known rooms — that is what this script produces.

What it does
------------
1. Loads an HSSD ``*.scene_instance.json`` and its sibling
   ``*.semantic_config.json``.
2. Resolves each existing furniture instance's category by looking up
   its ``template_name`` hash in
   ``metadata/object_categories_filtered.csv``.
3. For every entry in the placement spec (YAML/JSON), finds candidate
   furniture instances whose category matches ``furniture`` and (if
   given) whose translation falls inside the polygon of ``room_hint``.
4. Picks ``count`` deterministic furniture instances (sorted by
   translation), and for each one inserts a new ``object_instances``
   entry with translation = furniture_translation + (0, height, 0).
5. Writes the augmented ``*.scene_instance.json`` (creating a
   ``*.before_placements.json`` backup the first time) and a JSONL with
   one query per inserted object.

The script does NOT touch Habitat-Sim, so placements are approximate —
heights are per-furniture-type defaults (counter ~0.95, table ~0.75,
desk ~0.75, shelf ~1.0, sofa ~0.5) and small objects are STATIC.
Floating by a few centimetres is acceptable: YOLOE only needs the
object visible from the approach pose.

Usage
-----
  python tools/place_small_objects.py \\
      --scene /workspace/data/versioned_data/hssd-hab/scenes/102344280.scene_instance.json \\
      --spec  tools/eval_queries/specs/small_objects_102344280.yaml \\
      --out-jsonl tools/eval_queries/102344280_small_objects.jsonl

The augmented scene file overwrites the original (after backing up to
``*.before_placements.json``); pass ``--out-scene`` to write to a
different path instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# Approximate "top of furniture" Y offsets (metres). HSSD furniture
# translations are at the asset's local origin, which is typically near
# the bottom-centre. These defaults are conservative — small objects
# settle visibly on top without intersecting geometry from above.
_FURNITURE_HEIGHT_M: Dict[str, float] = {
    "counter":     0.95,
    "table":       0.75,
    "desk":        0.75,
    "shelf":       1.20,
    "shelving":    1.20,
    "cabinet":     1.05,
    "sofa":        0.45,
    "bed":         0.50,
    "chair":       0.50,
    "stool":       0.50,
    "refrigerator": 1.70,
    "oven":        0.95,
    "sink":        0.90,
}
_DEFAULT_HEIGHT_M = 0.85


# Furniture aliases: VLMap vocabulary (left) ↔ HSSD main_category (right).
# Lookups are done after lower-casing both sides; either side is accepted.
_FURNITURE_ALIASES: Dict[str, str] = {
    "sofa":  "couch",
    "shelf": "shelves",
    "tv":    "tv_screen",
}


def _normalize_furniture_label(label: str) -> str:
    s = label.strip().lower()
    # Two-way alias resolution: fold both spellings to a canonical key set.
    return _FURNITURE_ALIASES.get(s, s)


def _available_object_handles(metadata_csv: Path) -> Set[str]:
    dataset_root = metadata_csv.resolve().parents[1]
    objects_root = dataset_root / "objects"
    if not objects_root.exists():
        return set()
    return {
        p.name[: -len(".object_config.json")]
        for p in objects_root.rglob("*.object_config.json")
    }


# ── Extra category → template hash mappings ─────────────────────────────────
#
# `object_categories_filtered.csv` ships only the YCB-style identifiers
# (e.g. `Laptop_1`, `Laptop_24`) for some categories. These IDs do not have
# `.object_config.json` files in HSSD, so the on-disk filter strips them
# entirely and the placer cannot insert a laptop. Many actual laptop assets
# DO ship with full configs but live under the regular HSSD furniture
# objects/ tree under their hash IDs.
#
# This table re-injects those hash IDs as extra catalog entries for the
# affected categories. Entries whose `.object_config.json` is missing on
# disk are dropped by the same `_available_object_handles` filter, so this
# map is safe to extend optimistically.
_EXTRA_CATALOG: Dict[str, List[str]] = {
    "laptop": [
        # Curated laptop hashes that ship with both .glb and .object_config.json.
        # First entry wins via _safe_template_for_category's deterministic sort.
        "00b3040d6d75d68ba6d1bdf96e31c6b8432cbc1f",  # Toshiba Satellite Laptop
        "688e8dde29b7a0f6d35a36a478f05050b2e5c262",  # intel celeron 15.4in
        "f2659098f38a17e014343c77fd9affaaaea2f84b",  # Computer Laptop
        "61304cfe7c7c7c57463ac6297b6e02cf622cbf75",  # Sony Vaio
        "588ddc58a77c05c0a8d92232d30b6b197663dbdf",  # HP Pavilion
        "cd743d72e27ab591706521942259495acd8a819a",  # Toshiba L300D
        "c0fa43b54bd550bd1cdb93ef4d97b57ce585fc19",  # Apple MacBook Pro 17"
        "8cad457393cf850f77738f647a8c5e445176d881",  # Apple MacBook Pro 15.4"
    ],
}


def _load_object_catalog(metadata_csv: Path) -> Dict[str, List[str]]:
    """Return ``{clean_category: [template_id, ...]}`` from the small-object
    pickable catalog (``object_categories_filtered.csv``), augmented with
    the curated extras in ``_EXTRA_CATALOG``."""
    cat_to_ids: Dict[str, List[str]] = defaultdict(list)
    available = _available_object_handles(metadata_csv)
    with metadata_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = row.get("clean_category", "").strip().lower()
            tid = row.get("id", "").strip()
            if cat and tid and (not available or tid in available):
                cat_to_ids[cat].append(tid)
    # Merge extras (skip ones that are already present and skip missing assets).
    for cat, extra_ids in _EXTRA_CATALOG.items():
        existing = set(cat_to_ids.get(cat, []))
        for tid in extra_ids:
            if tid in existing:
                continue
            if available and tid not in available:
                continue
            cat_to_ids[cat].append(tid)
    return cat_to_ids


def _template_to_category(furniture_csv: Path) -> Dict[str, str]:
    """Return ``{template_id: main_category}`` from the full HSSD furniture CSV.

    Uses ``fpmodels-with-decomposed.csv`` (column ``main_category``) which
    covers the ~20k furniture/asset templates that appear in scene files.
    The smaller ``object_categories_filtered.csv`` only covers ~2k pickable
    small objects and would miss the actual room furniture.
    """
    out: Dict[str, str] = {}
    with furniture_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = (row.get("main_category") or "").strip().lower()
            tid = (row.get("id") or "").strip()
            if cat and tid:
                out[tid] = cat
    return out


def _instance_template_id(inst: Dict[str, Any]) -> str:
    """Return the bare template id used for category lookup."""
    return str(inst.get("template_name", "")).split("/")[-1].strip()


def _point_in_polygon(x: float, z: float, poly_xz: List[Tuple[float, float]]) -> bool:
    """Standard ray casting (XZ plane). Polygon is a closed loop of (X, Z)."""
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


def _load_rooms(semantic_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """Return ``{room_id: [(X, Z), ...]}`` for every region in the file.

    The room_id is ``"<name>__<idx>"`` to disambiguate duplicates (HSSD
    allows e.g. two ``bedroom`` regions in the same scene).
    """
    with semantic_path.open("r", encoding="utf-8") as f:
        sem = json.load(f)
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


def _room_for_xz(
    x: float, z: float, rooms: Dict[str, List[Tuple[float, float]]]
) -> Optional[str]:
    for room_id, poly in rooms.items():
        if _point_in_polygon(x, z, poly):
            return room_id
    return None


def _load_spec(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError:
            print(
                f"PyYAML not installed; please pass a JSON spec or `pip install pyyaml`",
                file=sys.stderr,
            )
            sys.exit(2)
        return yaml.safe_load(raw)
    return json.loads(raw)


def _identity_quaternion() -> List[float]:
    """Habitat-Sim quaternion order: (w, x, y, z), identity rotation."""
    return [1.0, 0.0, 0.0, 0.0]


def _matches_room_hint(detected_room: Optional[str], hint: Optional[str]) -> bool:
    if not hint:
        return True
    if not detected_room:
        return False
    hint_lower = hint.strip().lower()
    base = detected_room.split("__")[0]
    return hint_lower == base or hint_lower in detected_room


# ── Spatial deconfliction for placements that share a host ──────────────────
#
# Multiple spec entries (bottle, kettle, toaster, ...) often request the same
# furniture+room (counter+kitchen). The deterministic candidate sort would
# pick the SAME first counter for all of them, stacking objects at identical
# (x, fy+height, z). Habitat then collapses them visually: only the topmost
# template renders cleanly, the others are occluded inside it. YOLOE never
# sees the buried instances.
#
# Fix: track host usage across the whole placement run so that:
#   1. fresh hosts (never used) are preferred over already-used ones.
#   2. when a host MUST be reused (no other candidates), apply a small
#      deterministic XZ offset on the surface so the new object stands
#      next to the previous one instead of inside it.
#
# Offsets are chosen on a 35 cm grid for furniture (most HSSD counters and
# tables span >1 m, so up to ~5 distinct objects fit comfortably). Floor
# placements use a smaller 30 cm grid: a wider step would push the placed
# object far from the room centroid where the navigator stops, and the
# robot would not see it during the verification scan.

_FURNITURE_OFFSET_STEP_M = 0.30
_FLOOR_OFFSET_STEP_M = 0.30


# Approximate half-extent (m) of the host's TOP face per furniture category.
# Used to validate that the placement offset keeps the object body fully on
# the support surface. These are conservative — when in doubt the placer
# prefers to skip the host instead of letting an object overhang.
_FURNITURE_TOP_HALF_XZ_M: Dict[str, float] = {
    "counter":     0.55,   # counters are long (~2m) but ~0.6m deep; conservative
    "table":       0.55,
    "desk":        0.50,
    "shelf":       0.30,   # narrow shelves; many native items already on top
    "shelving":    0.30,
    "cabinet":     0.45,
    "sofa":        0.55,
    "bed":         0.75,
    "chair":       0.22,
    "stool":       0.18,
    "refrigerator": 0.35,
    "oven":        0.40,
    "sink":        0.30,
}
_DEFAULT_TOP_HALF_M = 0.35


# Approximate half-footprint (m) of the placed object body. The placer
# checks that |offset| + obj_half_extent ≤ host_top_half so the object body
# fully rests on the support face.
_OBJECT_FOOTPRINT_HALF_M: Dict[str, float] = {
    "book":        0.13,
    "books":       0.13,
    "laptop":      0.20,
    "notebook":    0.13,
    "computer":    0.20,
    "bottle":      0.05,
    "mug":         0.05,
    "cup":         0.05,
    "toaster":     0.13,
    "teapot":      0.10,
    "kettle":      0.10,
    "plate":       0.12,
    "bowl":        0.10,
}
_DEFAULT_OBJ_HALF_M = 0.10


def _fits_on_top(furn_cat: str, obj_cat: str, dx: float, dz: float) -> bool:
    """True if an object placed at (dx, dz) offset from host translation
    keeps its full body inside the host's top face."""
    top_half = _FURNITURE_TOP_HALF_XZ_M.get(furn_cat, _DEFAULT_TOP_HALF_M)
    obj_half = _OBJECT_FOOTPRINT_HALF_M.get(obj_cat, _DEFAULT_OBJ_HALF_M)
    margin = top_half - obj_half
    if margin <= 0:
        # Object body is wider than estimated host top — let the caller skip.
        return False
    return abs(dx) <= margin and abs(dz) <= margin


def _grid_offset(n: int, step: float) -> Tuple[float, float]:
    """Return a deterministic (dx, dz) offset for the n-th re-use of a host.

    Pattern (origin first, then four cardinal at +step, then four diagonals,
    then expand to 2*step ring, ...):
      n=0 → (0, 0)         centred
      n=1 → (+s, 0)        +X
      n=2 → (-s, 0)        -X
      n=3 → ( 0,+s)        +Z
      n=4 → ( 0,-s)        -Z
      n=5 → (+s,+s)        diagonal
      n=6 → (-s,-s)
      n=7 → (+s,-s)
      n=8 → (-s,+s)
      n=9+ → (±2s, 0) ring (rare; spec count would have to exceed 9 reuses)
    """
    pattern = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2),
    ]
    if n < len(pattern):
        ix, iz = pattern[n]
    else:
        # Spiral fallback for very dense reuse: never expected in practice.
        ring = 1 + ((n - len(pattern)) // 4)
        side = (n - len(pattern)) % 4
        ix, iz = [(ring, 0), (0, ring), (-ring, 0), (0, -ring)][side]
    return (ix * step, iz * step)


def _candidate_host_key(
    candidate: Any,
    furn_cat: str,
    room_id: Optional[str],
) -> Tuple[Any, ...]:
    """Stable key used to detect host re-use across spec entries."""
    if furn_cat == "floor":
        # Floor candidates come back as (translation_list, room_id).
        x, _y, z = (round(float(v), 2) for v in candidate)
        return ("floor", room_id or "?", x, z)
    tid = _instance_template_id(candidate)
    fx, _fy, fz = (round(float(v), 2) for v in candidate.get("translation", [0, 0, 0]))
    return (tid, fx, fz)


def _reorder_candidates_fresh_first(
    candidates: List[Tuple[Any, Optional[str]]],
    used_host_keys: set,
    furn_cat: str,
) -> List[Tuple[Any, Optional[str]]]:
    """Move never-used hosts to the front while preserving relative order."""
    fresh: List[Tuple[Any, Optional[str]]] = []
    reused: List[Tuple[Any, Optional[str]]] = []
    for cand, room_id in candidates:
        key = _candidate_host_key(cand, furn_cat, room_id)
        (reused if key in used_host_keys else fresh).append((cand, room_id))
    return fresh + reused


def _select_furniture_candidates(
    instances: List[Dict[str, Any]],
    template_to_cat: Dict[str, str],
    target_category: str,
    room_hint: Optional[str],
    rooms: Dict[str, List[Tuple[float, float]]],
) -> List[Tuple[Dict[str, Any], Optional[str]]]:
    target = _normalize_furniture_label(target_category)
    out: List[Tuple[Dict[str, Any], Optional[str]]] = []
    for inst in instances:
        tid = _instance_template_id(inst)
        cat = _normalize_furniture_label(template_to_cat.get(tid, ""))
        if cat != target:
            continue
        x, _y, z = (float(v) for v in inst.get("translation", [0.0, 0.0, 0.0]))
        room_id = _room_for_xz(x, z, rooms)
        if not _matches_room_hint(room_id, room_hint):
            continue
        out.append((inst, room_id))
    out.sort(key=lambda pair: (
        float(pair[0]["translation"][0]),
        float(pair[0]["translation"][2]),
    ))
    return out


def _polygon_centroid(poly_xz: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not poly_xz:
        return None
    xs = [p[0] for p in poly_xz]
    zs = [p[1] for p in poly_xz]
    return (sum(xs) / len(xs), sum(zs) / len(zs))


def _select_floor_candidates(
    room_hint: Optional[str],
    rooms: Dict[str, List[Tuple[float, float]]],
) -> List[Tuple[List[float], str]]:
    out: List[Tuple[List[float], str]] = []
    for room_id, poly in sorted(rooms.items()):
        if not _matches_room_hint(room_id, room_hint):
            continue
        centroid = _polygon_centroid(poly)
        if centroid is None:
            continue
        x, z = centroid
        out.append(([float(x), 0.02, float(z)], room_id))
    return out


def _build_object_instance(
    template_id: str,
    translation: List[float],
) -> Dict[str, Any]:
    return {
        "template_name": template_id,
        "translation": [float(v) for v in translation],
        "rotation": _identity_quaternion(),
        "non_uniform_scale": [1.0, 1.0, 1.0],
        "motion_type": "STATIC",
    }


def _instance_already_present(
    instances: List[Dict[str, Any]],
    template_id: str,
    translation: List[float],
    tol: float = 1e-4,
) -> bool:
    for inst in instances:
        if str(inst.get("template_name", "")).split("/")[-1].strip() != template_id:
            continue
        current = [float(v) for v in inst.get("translation", [0.0, 0.0, 0.0])]
        if len(current) != 3:
            continue
        if all(abs(a - b) <= tol for a, b in zip(current, translation)):
            return True
    return False


def _any_instance_within(
    instances: List[Dict[str, Any]],
    target_xyz: Tuple[float, float, float],
    *,
    radius_xz: float = 0.20,
    y_tol: float = 0.30,
) -> bool:
    """Generic-overlap guard. Returns True if any existing scene instance
    sits within ``radius_xz`` (XZ plane) AND within ``y_tol`` vertically.

    This catches collisions that ``_category_already_near`` misses — e.g.
    placing a mug on a counter where there's already a native plate or a
    salt shaker. The Y filter prevents flagging stacked objects as
    overlaps (an item under the table is not in the way of one on top).
    """
    tx, ty, tz = float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])
    r2 = radius_xz * radius_xz
    for inst in instances:
        tr = inst.get("translation", None)
        if not tr or len(tr) != 3:
            continue
        ix, iy, iz = float(tr[0]), float(tr[1]), float(tr[2])
        if abs(iy - ty) > y_tol:
            continue
        dx = ix - tx
        dz = iz - tz
        if dx * dx + dz * dz <= r2:
            return True
    return False


def _category_already_near(
    instances: List[Dict[str, Any]],
    template_to_cat: Dict[str, str],
    target_cat: str,
    target_xz: Tuple[float, float],
    radius: float = 0.6,
) -> bool:
    """True if any instance of ``target_cat`` (or its YOLOE-confusable
    aliases) is already within *radius* metres of ``target_xz``.

    Used by the placer to avoid stacking a new placement on top of a
    native HSSD instance of the same kind — e.g. our placed ``laptop``
    landing exactly on the same desk as the scene's pre-existing laptop.
    """
    target_cat = (target_cat or "").strip().lower()
    if not target_cat:
        return False
    aliases = {target_cat}
    # Common asset/category aliases — keep conservative.
    _alias_table = {
        "laptop": {"laptop", "notebook", "computer"},
        "book": {"book", "books", "notebook"},
        "mug": {"mug", "cup"},
        "cup": {"mug", "cup"},
        "bottle": {"bottle"},
        "teapot": {"teapot", "kettle"},
        "kettle": {"teapot", "kettle"},
        "toaster": {"toaster"},
    }
    aliases |= _alias_table.get(target_cat, set())
    tx, tz = float(target_xz[0]), float(target_xz[1])
    r2 = float(radius) * float(radius)
    for inst in instances:
        tid = str(inst.get("template_name", "")).split("/")[-1].strip()
        cat = template_to_cat.get(tid, "").strip().lower()
        if cat not in aliases:
            continue
        tr = inst.get("translation", [0.0, 0.0, 0.0])
        if len(tr) < 3:
            continue
        dx = float(tr[0]) - tx
        dz = float(tr[2]) - tz
        if dx * dx + dz * dz <= r2:
            return True
    return False


def _safe_template_for_category(
    catalog: Dict[str, List[str]], category: str
) -> Optional[str]:
    """Pick a deterministic template id for a given small-object category."""
    cat = category.strip().lower()
    options = catalog.get(cat, [])
    if not options:
        return None
    # Prefer "human-readable" template ids (e.g. ``Book_5``) over hashes
    # for easier audit; fall back to alphabetical if none.
    readable = [t for t in options if not all(c in "0123456789abcdef" for c in t.lower())]
    pool = readable or options
    pool = sorted(pool)
    return pool[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, type=Path,
                   help="Path to the HSSD *.scene_instance.json to augment.")
    p.add_argument("--spec", required=True, type=Path,
                   help="YAML or JSON file describing the placements.")
    p.add_argument("--out-scene", type=Path, default=None,
                   help="Where to write the augmented scene file. Defaults "
                        "to the original path (after backing it up).")
    p.add_argument("--out-jsonl", required=True, type=Path,
                   help="JSONL file to write evaluation queries to.")
    p.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("/workspace/data/versioned_data/hssd-hab/metadata/object_categories_filtered.csv"),
        help="Path to HSSD's object_categories_filtered.csv (small/pickable "
             "objects — used to pick which template to insert).",
    )
    p.add_argument(
        "--furniture-csv",
        type=Path,
        default=Path("/workspace/data/versioned_data/hssd-hab/metadata/fpmodels-with-decomposed.csv"),
        help="Path to HSSD's fpmodels-with-decomposed.csv (full furniture "
             "catalog — used to resolve the category of existing instances).",
    )
    p.add_argument("--semantic-config", type=Path, default=None,
                   help="Path to the *.semantic_config.json. Defaults to the "
                        "sibling file under .../semantics/scenes/.")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip the *.before_placements.json backup (useful when "
                        "writing to --out-scene only).")
    args = p.parse_args()

    if not args.scene.exists():
        print(f"Scene not found: {args.scene}", file=sys.stderr)
        sys.exit(2)
    if not args.metadata_csv.exists():
        print(f"Metadata CSV not found: {args.metadata_csv}", file=sys.stderr)
        sys.exit(2)

    # Locate semantic_config if not given.
    semantic_path = args.semantic_config
    if semantic_path is None:
        # HSSD layout: scenes/<scene>.scene_instance.json
        # paired with semantics/scenes/<scene>.semantic_config.json
        scene_id = args.scene.name.replace(".scene_instance.json", "")
        semantic_path = args.scene.parent.parent / "semantics" / "scenes" / f"{scene_id}.semantic_config.json"
    if not semantic_path.exists():
        print(f"Semantic config not found: {semantic_path}", file=sys.stderr)
        sys.exit(2)

    spec = _load_spec(args.spec)
    placements = spec.get("placements") or []
    if not placements:
        print(f"Spec contains no 'placements' entries: {args.spec}", file=sys.stderr)
        sys.exit(2)

    if not args.furniture_csv.exists():
        print(f"Furniture CSV not found: {args.furniture_csv}", file=sys.stderr)
        sys.exit(2)
    catalog = _load_object_catalog(args.metadata_csv)
    template_to_cat = _template_to_category(args.furniture_csv)
    rooms = _load_rooms(semantic_path)
    print(f"Loaded {len(rooms)} rooms from {semantic_path.name}")
    print(f"Loaded {len(catalog)} object categories from metadata CSV")

    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    instances: List[Dict[str, Any]] = scene.setdefault("object_instances", [])
    articulated_instances: List[Dict[str, Any]] = scene.get("articulated_object_instances", [])
    host_instances: List[Dict[str, Any]] = list(instances) + list(articulated_instances)
    n_existing = len(instances)
    print(
        f"Scene has {n_existing} existing object instances "
        f"(hosts available: {len(host_instances)})"
    )

    queries: List[Dict[str, Any]] = []
    inserted: List[Dict[str, Any]] = []
    next_query_idx = 1

    scene_id_for_jsonl = spec.get("scene_id", 0)
    scene_name_for_jsonl = spec.get(
        "scene_name", args.scene.name.replace(".scene_instance.json", "_0")
    )

    # Track host re-use across the whole placement pass to prevent stacking.
    used_host_keys: set = set()
    host_reuse_count: Dict[Tuple[Any, ...], int] = {}

    for entry in placements:
        obj_cat = str(entry.get("object", "")).strip().lower()
        query_target = str(entry.get("query_target", obj_cat)).strip().lower()
        furn_cat = str(entry.get("furniture", "")).strip().lower()
        room_hint = entry.get("room_hint")
        count = int(entry.get("count", 1))
        tags = list(entry.get("tags") or ["small_object"])

        if not obj_cat or not furn_cat or not query_target:
            print(f"  [skip] entry missing object/furniture: {entry}", file=sys.stderr)
            continue

        template_id = _safe_template_for_category(catalog, obj_cat)
        if not template_id:
            print(f"  [skip] no HSSD template for object '{obj_cat}'", file=sys.stderr)
            continue

        if furn_cat == "floor":
            candidates = _select_floor_candidates(room_hint, rooms)
        else:
            candidates = _select_furniture_candidates(
                host_instances, template_to_cat, furn_cat, room_hint, rooms
            )
        if not candidates:
            hint = f" in room '{room_hint}'" if room_hint else ""
            print(
                f"  [skip] no '{furn_cat}' furniture instances found{hint} "
                f"for '{obj_cat}'",
                file=sys.stderr,
            )
            continue

        # Prefer hosts not yet used by previous placements in this pass.
        candidates = _reorder_candidates_fresh_first(
            candidates, used_host_keys, furn_cat
        )

        # Avoid stacking onto a host that already carries a native HSSD
        # instance of the same target category (e.g. our placed laptop
        # landing on a desk that already has the scene's pre-existing
        # laptop). The check is alias-aware (laptop/notebook/computer,
        # mug/cup, ...) and uses a 0.6 m XZ radius around the host
        # translation.
        def _candidate_clear(cand) -> bool:
            if furn_cat == "floor":
                xz = (float(cand[0]), float(cand[2]))
            else:
                tr = cand.get("translation", [0.0, 0.0, 0.0])
                xz = (float(tr[0]), float(tr[2]))
            return not _category_already_near(
                instances, template_to_cat, query_target, xz, radius=0.6
            )

        clear = [(c, r) for (c, r) in candidates if _candidate_clear(c)]
        if clear:
            candidates = clear
        else:
            print(
                f"  [overlap] every candidate host for '{obj_cat}' already "
                f"holds a native '{query_target}' nearby — placing on the first "
                f"anyway with grid offset",
                file=sys.stderr,
            )

        # Iterate through ALL candidates (not a fixed slice) until we've
        # placed `count` objects. Hosts that fail the top-face / overhang
        # check via _picked=False below will `continue` so the next host
        # in the list gets a chance — otherwise a single bad host (e.g.
        # an over-narrow shelf) could leave the placement count unmet.
        _placed_for_entry = 0
        _target_count = max(0, count)
        height = _FURNITURE_HEIGHT_M.get(furn_cat, _DEFAULT_HEIGHT_M)
        for candidate, room_id in candidates:
            if _placed_for_entry >= _target_count:
                break
            host_key = _candidate_host_key(candidate, furn_cat, room_id)
            n_reuse = host_reuse_count.get(host_key, 0)
            offset_step = (
                _FLOOR_OFFSET_STEP_M if furn_cat == "floor" else _FURNITURE_OFFSET_STEP_M
            )

            if furn_cat == "floor":
                fx, fy, fz = (float(v) for v in candidate)
            else:
                fx, fy, fz = (float(v) for v in candidate["translation"])

            # Choose an offset that:
            #   (1) keeps the placement inside the room polygon, AND
            #   (2) keeps the object body fully on the host top face, AND
            #   (3) BIASES toward the room interior (front edge of the
            #       host) so the robot sees the object from the open side
            #       it approaches from instead of behind a wall.
            #
            # The bias vector points from the host translation toward the
            # room polygon centroid, capped to ~60% of the available
            # margin so the reuse pattern still has room to spread.
            poly = rooms.get(room_id) if room_id else None
            bias_x, bias_z = 0.0, 0.0
            if furn_cat != "floor" and poly:
                _room_c = _polygon_centroid(poly)
                if _room_c is not None:
                    _vx = _room_c[0] - fx
                    _vz = _room_c[1] - fz
                    _norm = (_vx * _vx + _vz * _vz) ** 0.5
                    if _norm > 1e-3:
                        _top_half = _FURNITURE_TOP_HALF_XZ_M.get(
                            furn_cat, _DEFAULT_TOP_HALF_M
                        )
                        _obj_half = _OBJECT_FOOTPRINT_HALF_M.get(
                            obj_cat, _DEFAULT_OBJ_HALF_M
                        )
                        _bias_cap = max(0.0, 0.6 * (_top_half - _obj_half))
                        _bias_mag = min(0.25, _bias_cap)
                        bias_x = (_vx / _norm) * _bias_mag
                        bias_z = (_vz / _norm) * _bias_mag

            offset_x, offset_z = _grid_offset(n_reuse, offset_step)
            _picked = False
            _placement_y = (fy if furn_cat == "floor"
                            else fy + _FURNITURE_HEIGHT_M.get(furn_cat, _DEFAULT_HEIGHT_M))
            for k in range(n_reuse, n_reuse + 13):
                _gx, _gz = _grid_offset(k, offset_step)
                cand_dx = bias_x + _gx
                cand_dz = bias_z + _gz
                if poly and not _point_in_polygon(fx + cand_dx, fz + cand_dz, poly):
                    continue
                if furn_cat != "floor" and not _fits_on_top(furn_cat, obj_cat, cand_dx, cand_dz):
                    continue
                # Generic overlap guard: ensure no existing instance (native
                # or earlier placement in this run) sits at the same XZ +
                # similar Y. Without this we'd happily land a mug on top of
                # a native plate when only the category-aware check ran.
                cand_xyz = (fx + cand_dx, _placement_y, fz + cand_dz)
                if _any_instance_within(instances, cand_xyz, radius_xz=0.20, y_tol=0.30):
                    continue
                offset_x, offset_z = cand_dx, cand_dz
                n_reuse = k
                _picked = True
                break
            if furn_cat != "floor" and not _picked:
                # Host top face cannot accommodate this object body even at
                # the smallest offsets — skip this host so the object isn't
                # left overhanging or floating outside the support surface.
                print(
                    f"  [skip-host] '{obj_cat}' would overhang on '{furn_cat}' "
                    f"top (no offset keeps body on support); trying next host",
                    file=sys.stderr,
                )
                continue

            host_reuse_count[host_key] = n_reuse + 1
            used_host_keys.add(host_key)

            if furn_cat == "floor":
                placement = [fx + offset_x, fy, fz + offset_z]
                host_template = None
                host_translation = None
            else:
                furn_inst = candidate
                placement = [fx + offset_x, fy + height, fz + offset_z]
                host_template = _instance_template_id(furn_inst)
                host_translation = [fx, fy, fz]
            if n_reuse > 0:
                print(
                    f"  [offset] '{obj_cat}' reusing host (n={n_reuse}); "
                    f"applying offset ({offset_x:+.2f}, {offset_z:+.2f}) m"
                )
            if _instance_already_present(instances, template_id, placement):
                print(
                    f"  [skip] '{obj_cat}' already present at {placement[:1]+placement[2:]} "
                    f"(template='{template_id}')"
                )
                continue
            new_inst = _build_object_instance(template_id, placement)
            inserted.append(new_inst)
            instances.append(new_inst)
            _placed_for_entry += 1

            base_room = (room_id or "unknown").split("__")[0]
            qid = f"q{next_query_idx:03d}"
            next_query_idx += 1
            query_entry = {
                "id": qid,
                "scene_id": scene_id_for_jsonl,
                "scene_name": scene_name_for_jsonl,
                "query": query_target,
                "query_type": "object",
                "target_label": query_target,
                "expected_rooms": [base_room],
                "expected_room_polygons": (
                    [{"label": base_room, "instance_idx": int(room_id.split("__")[1])}]
                    if room_id else []
                ),
                "tags": tags + ["placed", f"on_{furn_cat}", f"inserted_{obj_cat}"],
                "placement": {
                    "template_id": template_id,
                    "inserted_object": obj_cat,
                    "query_target": query_target,
                    "translation": placement,
                    "host_furniture_template": host_template,
                    "host_furniture_translation": host_translation,
                    "host_room_id": room_id,
                },
            }
            queries.append(query_entry)
            print(
                f"  [+] placed '{obj_cat}' as query '{query_target}' on "
                f"'{furn_cat}' at {placement[:1]+placement[2:]} (room='{base_room}')"
            )

        if _placed_for_entry < _target_count:
            print(
                f"  [under-placed] '{obj_cat}' on '{furn_cat}': "
                f"requested {_target_count}, placed {_placed_for_entry} "
                f"(remaining hosts could not fit the object body on top)",
                file=sys.stderr,
            )

    if not inserted:
        print("No placements applied. Aborting without writing files.", file=sys.stderr)
        sys.exit(3)

    # Write augmented scene.
    out_scene = args.out_scene or args.scene
    if out_scene == args.scene and not args.no_backup:
        backup = args.scene.with_suffix(".before_placements.json")
        if not backup.exists():
            backup.write_text(args.scene.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"Backed up original scene to {backup}")
        else:
            print(f"Backup already exists at {backup}; not overwritten")

    out_scene.parent.mkdir(parents=True, exist_ok=True)
    out_scene.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    print(f"Wrote augmented scene to {out_scene} "
          f"({n_existing} → {len(instances)} instances)")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, sort_keys=True))
            f.write("\n")
    print(f"Wrote {len(queries)} queries to {args.out_jsonl}")


if __name__ == "__main__":
    main()
