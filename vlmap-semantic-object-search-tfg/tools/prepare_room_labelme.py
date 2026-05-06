#!/usr/bin/env python3
"""Prepare a scene for manual room-zone annotation with LabelMe.

This wrapper keeps the existing VLMaps utilities untouched and creates a
repo-local annotation workspace under:

  /workspace/annotations/room_labels/<dataset>/<scene_name>/

It ensures the top-down images exist, copies the relevant assets there and
writes a small manifest with the exact paths the user needs for LabelMe.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path("/workspace")
APP_DIR = REPO_ROOT / "third_party" / "vlmaps" / "application"
DEFAULT_HSSD_CFG = REPO_ROOT / "data" / "versioned_data" / "hssd-hab" / "hssd-hab.scene_dataset_config.json"
ANNOTATION_ROOT = REPO_ROOT / "annotations" / "room_labels"


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


def _run_generate(dataset_type: str, data_paths: str, scene_id: int, scene_dataset_config_file: Path | None) -> None:
    cmd = [
        sys.executable,
        str(APP_DIR / "generate_obstacle_map_png.py"),
        f"data_paths={data_paths}",
        f"scene_id={scene_id}",
    ]
    if dataset_type == "hssd":
        cmd.append("dataset_type=hssd")
        cfg = scene_dataset_config_file or DEFAULT_HSSD_CFG
        cmd.append(f"scene_dataset_config_file={cfg}")
    subprocess.run(cmd, cwd=str(APP_DIR.parent), check=True)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def _make_writable_tree(root: Path) -> None:
    """Make the annotation workspace writable from both host and container."""
    if not root.exists():
        return
    os.chmod(root, 0o777)
    for path in root.iterdir():
        if path.is_dir():
            _make_writable_tree(path)
        else:
            os.chmod(path, 0o666)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LabelMe room-zone annotation assets.")
    parser.add_argument("--dataset-type", choices=["hssd", "mp3d"], default="hssd")
    parser.add_argument("--data-paths", default=None, help="Override Hydra data_paths (default: hssd for HSSD, docker for MP3D).")
    parser.add_argument("--scene-id", type=int, required=True)
    parser.add_argument("--scene-dataset-config-file", type=Path, default=None)
    parser.add_argument("--host-workspace", default="/home/mario/tfg/vlmap-semantic-object-search-tfg")
    parser.add_argument("--regenerate", action="store_true", help="Force regeneration of top-down assets.")
    args = parser.parse_args()

    data_paths = args.data_paths or ("hssd" if args.dataset_type == "hssd" else "docker")
    scene_dir = _scene_dir_from_id(args.dataset_type, args.scene_id)
    annotation_dir = ANNOTATION_ROOT / args.dataset_type / scene_dir.name
    annotation_dir.mkdir(parents=True, exist_ok=True)

    topdown_labeled = scene_dir / "topdown_labeled.png"
    if args.regenerate or not topdown_labeled.exists():
        _run_generate(
            dataset_type=args.dataset_type,
            data_paths=data_paths,
            scene_id=args.scene_id,
            scene_dataset_config_file=args.scene_dataset_config_file,
        )

    for name in ("topdown_labeled.png", "topdown_rgb.png", "obstacle_map.png"):
        _copy_if_exists(scene_dir / name, annotation_dir / name)

    labelme_json = annotation_dir / "room_labels.json"
    manifest = {
        "dataset_type": args.dataset_type,
        "scene_id": args.scene_id,
        "scene_name": scene_dir.name,
        "scene_dir_container": str(scene_dir),
        "annotation_dir_container": str(annotation_dir),
        "annotation_dir_host": str(Path(args.host_workspace) / "annotations" / "room_labels" / args.dataset_type / scene_dir.name),
        "topdown_image_container": str(annotation_dir / "topdown_labeled.png"),
        "topdown_image_host": str(Path(args.host_workspace) / "annotations" / "room_labels" / args.dataset_type / scene_dir.name / "topdown_labeled.png"),
        "labelme_json_container": str(labelme_json),
        "labelme_json_host": str(Path(args.host_workspace) / "annotations" / "room_labels" / args.dataset_type / scene_dir.name / "room_labels.json"),
        "scene_dataset_config_file": str(args.scene_dataset_config_file or DEFAULT_HSSD_CFG) if args.dataset_type == "hssd" else "",
        "instructions": [
            "Marca solo la zona navegable útil de cada habitación.",
            "Si dos habitaciones abiertas se mezclan, sepáralas manualmente.",
            "Usa polígonos grandes y limpios; evita perfilar muebles o paredes con demasiado detalle.",
            "Puedes usar varias regiones con la misma etiqueta si hay zonas desconectadas.",
        ],
        "suggested_labels": [
            "living room",
            "bedroom",
            "kitchen",
            "bathroom",
            "office",
            "hallway",
            "dining room",
            "laundry room",
            "closet",
            "garage",
        ],
        "host_labelme_command": f"labelme \"{Path(args.host_workspace) / 'annotations' / 'room_labels' / args.dataset_type / scene_dir.name / 'topdown_labeled.png'}\" -O \"{Path(args.host_workspace) / 'annotations' / 'room_labels' / args.dataset_type / scene_dir.name / 'room_labels.json'}\" --autosave --nodata",
    }
    (annotation_dir / "labelme_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _make_writable_tree(annotation_dir)

    print("")
    print("Preparación de LabelMe completada")
    print(f"  Escena           : {scene_dir.name}")
    print(f"  Anotación        : {annotation_dir}")
    print(f"  Imagen principal : {annotation_dir / 'topdown_labeled.png'}")
    print(f"  JSON esperado    : {labelme_json}")
    print("")
    print("Consejo:")
    print("  Marca solo la parte navegable útil de cada habitación.")
    print("  Si un espacio abierto mezcla dos zonas, sepáralas manualmente.")
    print("")
    print("Comando en el host:")
    print(f"  {manifest['host_labelme_command']}")


if __name__ == "__main__":
    main()
