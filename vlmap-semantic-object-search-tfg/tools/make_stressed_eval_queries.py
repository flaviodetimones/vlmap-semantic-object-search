#!/usr/bin/env python3
"""
Derive stressed evaluation JSONLs from the normalized source batches.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = REPO_ROOT / "tools" / "eval_queries"
DEFAULT_OUT_HEATMAP = DEFAULT_SOURCES / "stressed_heatmap"
DEFAULT_OUT_ORCHESTRATOR = DEFAULT_SOURCES / "stressed_orchestrator"
DEFAULT_SEED = 24042026
MIN_VIABLE = 15


def _iter_source_jsonls(sources: Path) -> Iterable[Path]:
    for path in sorted(sources.glob("*.jsonl")):
        if path.is_file():
            yield path


def _normalize_scene_names(raw_scene_names: str | None, sources: Path) -> List[str]:
    if raw_scene_names:
        return [chunk.strip() for chunk in raw_scene_names.split(",") if chunk.strip()]
    return [path.stem for path in _iter_source_jsonls(sources)]


def _load_source_entries(path: Path) -> List[Tuple[str, dict]]:
    entries: List[Tuple[str, dict]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            entries.append((raw, json.loads(raw)))
    return entries


def _has_any_tag(payload: dict, tags: set[str]) -> bool:
    query_tags = set(str(tag) for tag in payload.get("tags") or [])
    return bool(query_tags & tags)


def select_heatmap_entries(entries: List[Tuple[str, dict]]) -> List[Tuple[str, dict]]:
    wanted = {"multi_instance", "ambiguous_room", "explicit_instance"}
    return [(raw, payload) for raw, payload in entries if _has_any_tag(payload, wanted)]


def select_orchestrator_entries(entries: List[Tuple[str, dict]]) -> List[Tuple[str, dict]]:
    out: List[Tuple[str, dict]] = []
    for raw, payload in entries:
        query_type = str(payload.get("query_type") or "").strip().lower()
        tags = set(str(tag) for tag in payload.get("tags") or [])
        if query_type == "room_object" and {"ambiguous_room", "multi_instance"} & tags:
            out.append((raw, payload))
            continue
        if query_type == "object" and {"ambiguous_room", "multi_instance"} <= tags:
            out.append((raw, payload))
    return out


def _stable_shuffle(entries: List[Tuple[str, dict]], seed: int) -> List[Tuple[str, dict]]:
    items = list(entries)
    rng = random.Random(seed)
    rng.shuffle(items)
    return items


def _write_entries(entries: List[Tuple[str, dict]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(raw + "\n" for raw, _ in entries)
    out_path.write_text(text, encoding="utf-8")


def _warn_if_sparse(scene_name: str, batch_name: str, count: int) -> None:
    if count < MIN_VIABLE:
        print(
            f"[stress][warn] {scene_name} {batch_name} produced {count} queries "
            f"(<{MIN_VIABLE} minimum viable)",
            file=sys.stderr,
        )


def build_stressed_batches(
    *,
    sources: Path,
    out_heatmap: Path,
    out_orchestrator: Path,
    scene_names: List[str],
    seed: int,
) -> List[dict]:
    summaries: List[dict] = []
    for idx, scene_name in enumerate(scene_names):
        src_path = sources / f"{scene_name}.jsonl"
        if not src_path.exists():
            raise SystemExit(f"Missing source JSONL: {src_path}")
        entries = _load_source_entries(src_path)
        heatmap = _stable_shuffle(select_heatmap_entries(entries), seed + idx)
        orchestrator = _stable_shuffle(select_orchestrator_entries(entries), seed + 10_000 + idx)

        _write_entries(heatmap, out_heatmap / f"{scene_name}.jsonl")
        _write_entries(orchestrator, out_orchestrator / f"{scene_name}.jsonl")

        _warn_if_sparse(scene_name, "heatmap", len(heatmap))
        _warn_if_sparse(scene_name, "orchestrator", len(orchestrator))

        print(
            f"[stress] {scene_name} heatmap={len(heatmap)} orchestrator={len(orchestrator)} "
            f"source={len(entries)}"
        )
        summaries.append(
            {
                "scene_name": scene_name,
                "source_queries": len(entries),
                "heatmap_queries": len(heatmap),
                "orchestrator_queries": len(orchestrator),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--out-heatmap", type=Path, default=DEFAULT_OUT_HEATMAP)
    parser.add_argument("--out-orchestrator", type=Path, default=DEFAULT_OUT_ORCHESTRATOR)
    parser.add_argument("--scene-names", default=None,
                        help="Optional comma-separated scene names. Defaults to all source JSONLs.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    scene_names = _normalize_scene_names(args.scene_names, args.sources)
    if not scene_names:
        raise SystemExit(f"No source JSONLs found in {args.sources}")
    build_stressed_batches(
        sources=args.sources,
        out_heatmap=args.out_heatmap,
        out_orchestrator=args.out_orchestrator,
        scene_names=scene_names,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
