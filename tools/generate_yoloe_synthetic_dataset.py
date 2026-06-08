#!/usr/bin/env python3
"""Generate a small YOLO-style synthetic segmentation dataset.

The generator uses Habitat only to render isolated object cutouts with exact
semantic masks. It then composites those cutouts over previously captured RGB
backgrounds and writes YOLO segmentation labels from mask-derived contours.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import habitat_sim
import magnum as mn
import numpy as np
from PIL import Image, ImageDraw


DEFAULT_SCENE_DATASET_CONFIG = (
    "/workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json"
)
DEFAULT_MANIFEST = "/workspace/configs/yoloe_synthetic_object_manifest.json"


@dataclass(frozen=True)
class RenderedCutout:
    class_name: str
    handle: str
    image: Image.Image
    alpha: Image.Image


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _make_render_sim(scene_dataset_config: str, width: int, height: int):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = scene_dataset_config
    sim_cfg.scene_id = "NONE"
    sim_cfg.enable_physics = False

    color = habitat_sim.CameraSensorSpec()
    color.uuid = "color"
    color.sensor_type = habitat_sim.SensorType.COLOR
    color.resolution = [height, width]
    color.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    semantic = habitat_sim.CameraSensorSpec()
    semantic.uuid = "semantic"
    semantic.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic.resolution = [height, width]
    semantic.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [color, semantic]
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def _resolve_template_handle(sim: habitat_sim.Simulator, raw_handle: str) -> Optional[str]:
    mgr = sim.get_object_template_manager()
    matches = mgr.get_template_handles(raw_handle)
    if not matches:
        return None
    # Prefer the exact object config match when Habitat returns multiple paths.
    suffix = f"{raw_handle}.object_config.json"
    for match in matches:
        if match.endswith(suffix):
            return match
    return matches[0]


def _rotation_for_class(rng: random.Random, class_name: str) -> Tuple[float, float]:
    yaw = rng.uniform(0.0, 2.0 * math.pi)
    if class_name == "ball":
        plane = rng.uniform(0.0, 360.0)
    elif class_name == "plush toy":
        plane = rng.uniform(-90.0, 90.0)
    elif class_name == "bottle" and rng.random() < 0.15:
        plane = rng.choice([-90.0, 90.0]) + rng.uniform(-12.0, 12.0)
    else:
        plane = rng.uniform(-15.0, 15.0)
    return yaw, plane


def _crop_from_mask(rgb: np.ndarray, semantic: np.ndarray, sem_id: int) -> Optional[Tuple[Image.Image, Image.Image]]:
    mask = semantic == sem_id
    if int(mask.sum()) < 20:
        return None
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad = 8
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(rgb.shape[1], x1 + pad)
    y1 = min(rgb.shape[0], y1 + pad)
    cut_rgb = rgb[y0:y1, x0:x1, :3]
    cut_mask = (mask[y0:y1, x0:x1].astype(np.uint8) * 255)
    return Image.fromarray(cut_rgb, "RGB"), Image.fromarray(cut_mask, "L")


def _render_cutout(
    sim: habitat_sim.Simulator,
    rng: random.Random,
    class_name: str,
    handle: str,
    semantic_id: int,
) -> Optional[RenderedCutout]:
    full_handle = _resolve_template_handle(sim, handle)
    if full_handle is None:
        print(f"[synthetic] missing object template: {handle}")
        return None

    rom = sim.get_rigid_object_manager()
    obj = rom.add_object_by_template_handle(full_handle)
    if obj is None:
        print(f"[synthetic] failed to add object: {handle}")
        return None

    try:
        yaw, plane_rot = _rotation_for_class(rng, class_name)
        try:
            obj.rotation = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3.y_axis())
        except Exception:
            pass
        obj.translation = np.array([0.0, 0.0, -1.2], dtype=np.float32)
        obj.semantic_id = semantic_id
        obs = sim.get_sensor_observations()
        cropped = _crop_from_mask(obs["color"], obs["semantic"], semantic_id)
        if cropped is None:
            return None
        rgb_img, alpha = cropped
        if abs(plane_rot) > 0.5:
            rgba = rgb_img.copy()
            rgba.putalpha(alpha)
            rgba = rgba.rotate(plane_rot, expand=True, resample=Image.Resampling.BICUBIC)
            alpha = rgba.getchannel("A")
            rgb_img = rgba.convert("RGB")
        return RenderedCutout(class_name=class_name, handle=handle, image=rgb_img, alpha=alpha)
    finally:
        try:
            rom.remove_object_by_id(obj.object_id)
        except Exception:
            pass


def _build_cutout_bank(
    manifest: Dict[str, Any],
    scene_dataset_config: str,
    rng: random.Random,
    render_width: int,
    render_height: int,
    variants_per_object: int,
) -> Dict[str, List[RenderedCutout]]:
    sim = _make_render_sim(scene_dataset_config, render_width, render_height)
    bank: Dict[str, List[RenderedCutout]] = {name: [] for name in manifest["class_names"]}
    sem_id = 7
    try:
        for class_name in manifest["class_names"]:
            for item in manifest["objects"][class_name]:
                handle = item["handle"]
                for _ in range(variants_per_object):
                    cut = _render_cutout(sim, rng, class_name, handle, sem_id)
                    if cut is not None:
                        bank[class_name].append(cut)
        return bank
    finally:
        sim.close()


def _read_backgrounds_one(backgrounds_dir: Path) -> List[Path]:
    image_dir = backgrounds_dir / "images"
    if not image_dir.exists():
        image_dir = backgrounds_dir
    paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not paths:
        raise FileNotFoundError(f"no background images found under {backgrounds_dir}")
    return paths


def _read_backgrounds(backgrounds_dirs: Iterable[Path]) -> List[Path]:
    paths: List[Path] = []
    seen: set[Path] = set()
    for backgrounds_dir in backgrounds_dirs:
        for path in _read_backgrounds_one(backgrounds_dir):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    if not paths:
        raise FileNotFoundError("no background images found in any --backgrounds-dir")
    return paths


def _split_for_index(idx: int, total: int) -> str:
    frac = idx / max(total, 1)
    if frac < 0.70:
        return "train"
    if frac < 0.90:
        return "val"
    return "test"


def _relative_width_range(manifest: Dict[str, Any], class_name: str) -> Tuple[float, float]:
    raw = manifest["generation_policy"]["relative_box_width_ranges"][class_name]
    return float(raw[0]), float(raw[1])


def _resize_cutout(
    rng: random.Random,
    cutout: RenderedCutout,
    manifest: Dict[str, Any],
    canvas_w: int,
    canvas_h: int,
) -> Tuple[Image.Image, Image.Image]:
    min_rel, max_rel = _relative_width_range(manifest, cutout.class_name)
    target_w = int(canvas_w * rng.uniform(min_rel, max_rel))
    scale = target_w / max(cutout.image.width, 1)
    target_h = max(4, int(cutout.image.height * scale))
    if target_h > int(canvas_h * 0.75):
        scale = (canvas_h * 0.75) / max(cutout.image.height, 1)
        target_w = max(4, int(cutout.image.width * scale))
        target_h = max(4, int(cutout.image.height * scale))
    img = cutout.image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    alpha = cutout.alpha.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return img, alpha


def _sample_position(
    rng: random.Random,
    canvas_w: int,
    canvas_h: int,
    obj_w: int,
    obj_h: int,
    partial_fraction: float,
    lower_fraction: float,
) -> Tuple[int, int]:
    allow_partial = rng.random() < partial_fraction
    margin_x = int(obj_w * 0.25) if allow_partial else 0
    margin_y = int(obj_h * 0.20) if allow_partial else 0
    x = rng.randint(-margin_x, max(-margin_x, canvas_w - obj_w + margin_x))
    if rng.random() < lower_fraction:
        y_min = int(canvas_h * 0.35)
    else:
        y_min = 0
    y = rng.randint(max(-margin_y, y_min), max(max(-margin_y, y_min), canvas_h - obj_h + margin_y))
    return x, y


def _bbox_from_alpha(alpha: Image.Image, x: int, y: int, canvas_w: int, canvas_h: int) -> Optional[Tuple[int, int, int, int]]:
    arr = np.asarray(alpha) > 12
    if int(arr.sum()) < 20:
        return None
    ys, xs = np.where(arr)
    x0 = max(0, int(xs.min()) + x)
    x1 = min(canvas_w, int(xs.max()) + 1 + x)
    y0 = max(0, int(ys.min()) + y)
    y1 = min(canvas_h, int(ys.max()) + 1 + y)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, y0, x1, y1


def _visible_mask_from_alpha(alpha: Image.Image, x: int, y: int, canvas_w: int, canvas_h: int) -> np.ndarray:
    arr = (np.asarray(alpha) > 12).astype(np.uint8)
    full = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    src_h, src_w = arr.shape[:2]
    dst_x0 = max(0, x)
    dst_y0 = max(0, y)
    dst_x1 = min(canvas_w, x + src_w)
    dst_y1 = min(canvas_h, y + src_h)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return full
    src_x0 = dst_x0 - x
    src_y0 = dst_y0 - y
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    full[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return full


def _segment_points_from_mask(mask: np.ndarray, width: int, height: int) -> Optional[List[Tuple[float, float]]]:
    if int(mask.sum()) < 20:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20:
        return None

    peri = cv2.arcLength(contour, True)
    epsilon = max(1.0, 0.003 * peri)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    while len(approx) > 80:
        epsilon *= 1.5
        approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        return None

    points: List[Tuple[float, float]] = []
    for pt in approx.reshape(-1, 2):
        px = min(max(float(pt[0]) / width, 0.0), 1.0)
        py = min(max(float(pt[1]) / height, 0.0), 1.0)
        points.append((px, py))
    return points


def _yolo_segment_line(class_id: int, points: List[Tuple[float, float]]) -> str:
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
    return f"{class_id} {coords}"


def _ensure_dataset_dirs(out_dir: Path) -> None:
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def _write_data_yaml(out_dir: Path, class_names: List[str]) -> None:
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    text = (
        f"path: {out_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"{names}\n"
    )
    (out_dir / "data.yaml").write_text(text, encoding="utf-8")


def _draw_review_mosaic(
    samples: List[Tuple[Path, List[Tuple[str, Tuple[int, int, int, int]]]]],
    out_path: Path,
) -> None:
    cells: List[Image.Image] = []
    for image_path, boxes in samples[:25]:
        img = Image.open(image_path).convert("RGB")
        src_w, src_h = img.size
        img.thumbnail((220, 165))
        sx = img.width / max(src_w, 1)
        sy = img.height / max(src_h, 1)
        draw = ImageDraw.Draw(img)
        for name, (x0, y0, x1, y1) in boxes:
            box = (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))
            draw.rectangle(box, outline=(0, 255, 0), width=2)
            draw.text((box[0] + 2, max(0, box[1] - 12)), name, fill=(0, 255, 0))
        canvas = Image.new("RGB", (220, 185), "white")
        canvas.paste(img, ((220 - img.width) // 2, 0))
        ImageDraw.Draw(canvas).text((4, 168), image_path.name, fill=(0, 0, 0))
        cells.append(canvas)

    cols = 5
    rows = math.ceil(len(cells) / cols)
    sheet = Image.new("RGB", (cols * 220, rows * 185), (240, 240, 240))
    for i, cell in enumerate(cells):
        sheet.paste(cell, ((i % cols) * 220, (i // cols) * 185))
    sheet.save(out_path)


def generate_dataset(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    manifest = _load_json(args.manifest)
    class_names = list(manifest["class_names"])
    class_to_id = {name: i for i, name in enumerate(class_names)}
    bg_paths = _read_backgrounds(args.backgrounds_dir)
    canvas_w = int(args.width)
    canvas_h = int(args.height)
    print(f"[synthetic] background directories: {len(args.backgrounds_dir)}")
    print(f"[synthetic] background images: {len(bg_paths)}")
    print(f"[synthetic] output size: {canvas_w}x{canvas_h}")

    out_dir = args.out_dir
    _ensure_dataset_dirs(out_dir)
    _write_data_yaml(out_dir, class_names)

    print("[synthetic] rendering object cutouts...")
    bank = _build_cutout_bank(
        manifest,
        args.scene_dataset_config,
        rng,
        args.render_width,
        args.render_height,
        args.variants_per_object,
    )
    for class_name, cutouts in bank.items():
        print(f"[synthetic] cutouts {class_name}: {len(cutouts)}")
        if not cutouts:
            raise RuntimeError(f"no usable cutouts for class '{class_name}'")

    policy = manifest["generation_policy"]
    negative_fraction = float(policy.get("negative_image_fraction", 0.2))
    max_objects = int(policy.get("max_objects_per_image", 3))
    partial_fraction = float(policy.get("placement", {}).get("allow_partial_cutoff_fraction", 0.1))
    lower_fraction = float(policy.get("placement", {}).get("prefer_lower_image_fraction", 0.7))

    order = list(range(args.images))
    rng.shuffle(order)
    review: List[Tuple[Path, List[Tuple[str, Tuple[int, int, int, int]]]]] = []
    metadata_path = out_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as meta:
        for out_idx, source_idx in enumerate(order):
            split = _split_for_index(out_idx, args.images)
            bg = Image.open(bg_paths[source_idx % len(bg_paths)]).convert("RGB").resize((canvas_w, canvas_h), Image.Resampling.BICUBIC)
            labels: List[str] = []
            boxes_for_review: List[Tuple[str, Tuple[int, int, int, int]]] = []
            records: List[Dict[str, Any]] = []

            if rng.random() >= negative_fraction:
                n_objects = rng.randint(1, max_objects)
                for _ in range(n_objects):
                    class_name = rng.choice(class_names)
                    cutout = rng.choice(bank[class_name])
                    img, alpha = _resize_cutout(rng, cutout, manifest, canvas_w, canvas_h)
                    x, y = _sample_position(rng, canvas_w, canvas_h, img.width, img.height, partial_fraction, lower_fraction)
                    bbox = _bbox_from_alpha(alpha, x, y, canvas_w, canvas_h)
                    if bbox is None:
                        continue
                    visible_mask = _visible_mask_from_alpha(alpha, x, y, canvas_w, canvas_h)
                    segment = _segment_points_from_mask(visible_mask, canvas_w, canvas_h)
                    if segment is None:
                        continue
                    bg.paste(img, (x, y), alpha)
                    class_id = class_to_id[class_name]
                    labels.append(_yolo_segment_line(class_id, segment))
                    boxes_for_review.append((class_name, bbox))
                    records.append({
                        "class": class_name,
                        "handle": cutout.handle,
                        "bbox_xyxy": list(bbox),
                        "segment_points": len(segment),
                    })

            stem = f"{out_idx:06d}"
            img_path = out_dir / "images" / split / f"{stem}.jpg"
            label_path = out_dir / "labels" / split / f"{stem}.txt"
            bg.save(img_path, quality=92)
            label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            meta.write(json.dumps({
                "image": str(img_path.relative_to(out_dir)),
                "label": str(label_path.relative_to(out_dir)),
                "background": str(bg_paths[source_idx % len(bg_paths)]),
                "objects": records,
            }, sort_keys=True) + "\n")
            if len(review) < 25:
                review.append((img_path, boxes_for_review))

    _draw_review_mosaic(review, out_dir / "review_mosaic.jpg")
    print(f"[synthetic] wrote dataset: {out_dir}")
    print(f"[synthetic] review: {out_dir / 'review_mosaic.jpg'}")
    print(f"[synthetic] yaml: {out_dir / 'data.yaml'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backgrounds-dir",
        type=Path,
        action="append",
        required=True,
        help="Background directory. Can be provided multiple times. If it contains an images/ child, that child is used.",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST))
    p.add_argument("--scene-dataset-config", default=DEFAULT_SCENE_DATASET_CONFIG)
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--width", type=int, default=640, help="Final composited image width.")
    p.add_argument("--height", type=int, default=480, help="Final composited image height.")
    p.add_argument("--render-width", type=int, default=640)
    p.add_argument("--render-height", type=int, default=480)
    p.add_argument("--variants-per-object", type=int, default=5)
    return p.parse_args()


def main() -> None:
    generate_dataset(parse_args())


if __name__ == "__main__":
    main()
