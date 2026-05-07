#!/usr/bin/env python3
"""Convert a LabelMe room annotation into a VLMaps room_map for a scene.

This wrapper is intentionally repo-local so the workflow can be driven from the
project menu without forcing edits in the existing VLMaps application scripts.
It reads the LabelMe JSON from the shared annotation workspace and writes the
generated room_map into the real scene directory used by the pipeline.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path("/workspace")
APP_DIR = REPO_ROOT / "third_party" / "vlmaps" / "application"
ANNOTATION_ROOT = REPO_ROOT / "annotations" / "room_labels"
DEFAULT_HSSD_CFG = REPO_ROOT / "data" / "versioned_data" / "hssd-hab" / "hssd-hab.scene_dataset_config.json"


def _dataset_scene_root(dataset_type: str) -> Path:
    if dataset_type == "hssd":
        return REPO_ROOT / "data" / "vlmaps_dataset_hssd"
    if dataset_type == "mp3d":
        return REPO_ROOT / "data" / "vlmaps_dataset"
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def _scene_dir_from_id(dataset_type: str, scene_id: int) -> Path:
    root = _dataset_scene_root(dataset_type)
    scene_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if scene_id < 0 or scene_id >= len(scene_dirs):
        raise IndexError(f"scene_id {scene_id} out of range [0, {len(scene_dirs) - 1}]")
    return scene_dirs[scene_id]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a LabelMe room annotation into a VLMaps room_map.")
    parser.add_argument("--dataset-type", choices=["hssd", "mp3d"], default="hssd")
    parser.add_argument("--scene-id", type=int, required=True)
    parser.add_argument("--json", type=Path, default=None, help="Override LabelMe JSON path.")
    parser.add_argument("--min-region-size", type=int, default=50)
    parser.add_argument(
        "--voronoi-max-distance-cells",
        type=int,
        default=50,
        help="Distancia máxima para asignar muebles/celdas ocupadas a la región LabelMe más cercana.",
    )
    parser.add_argument("--scene-dataset-config-file", type=Path, default=DEFAULT_HSSD_CFG)
    args = parser.parse_args()

    scene_dir = _scene_dir_from_id(args.dataset_type, args.scene_id)
    annotation_dir = ANNOTATION_ROOT / args.dataset_type / scene_dir.name
    json_candidates = [
        annotation_dir / "room_labels.json",
        annotation_dir / "topdown_labeled.json",
    ]
    json_path = args.json or next((p for p in json_candidates if p.exists()), json_candidates[0])

    if not json_path.exists():
        raise SystemExit(f"No se encontró el JSON de LabelMe: {json_path}")

    room_map_dir = scene_dir / "room_map"
    room_map_dir.mkdir(parents=True, exist_ok=True)
    scene_json_copy = room_map_dir / "room_labels.json"
    shutil.copy2(json_path, scene_json_copy)

    cmd = [
        sys.executable,
        str(APP_DIR / "labelme_to_room_map.py"),
        "--json",
        str(scene_json_copy),
        "--scene",
        str(scene_dir),
        "--min-region-size",
        str(args.min_region_size),
        "--voronoi-max-distance-cells",
        str(args.voronoi_max_distance_cells),
        "--no-preview",
    ]
    subprocess.run(cmd, cwd=str(APP_DIR.parent), check=True)

    print("")
    print("Conversión completada")
    print(f"  Escena         : {scene_dir.name}")
    print(f"  JSON usado     : {json_path}")
    print(f"  JSON copiado   : {scene_json_copy}")
    print(f"  room_map final : {room_map_dir}")


if __name__ == "__main__":
    main()
