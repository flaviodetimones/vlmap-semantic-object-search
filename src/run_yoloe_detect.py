"""
run_yoloe_detect.py  (Step 4)
=============================
Run YOLOE open-vocabulary detection on a scene's RGB frames.

For each sampled frame, runs YOLOE with text prompts and saves all
detections (bounding boxes, confidence, class name) as JSON.

Usage (inside Docker):
  python src/run_yoloe_detect.py data_paths=docker scene_id=0 \
      +queries='["chair","table","sofa","bed","toilet","sink","tv_monitor"]' \
      +sample_rate=10

Output:
  data/vlmaps_dataset/<scene>/yoloe_detections.json
"""

import json
import os
import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# YOLOE wrapper
# ─────────────────────────────────────────────────────────────────────────────

def load_yoloe(model_name="yoloe-11l-seg.pt"):
    """Load YOLOE model via ultralytics."""
    from ultralytics import YOLO
    model = YOLO(model_name)
    return model


def detect_frame(model, image, queries, conf_thresh=0.25):
    """
    Run YOLOE on a single image with text prompts.

    Returns list of dicts: {bbox_xyxy, confidence, class_name}
    """
    model.set_classes(queries)
    results = model.predict(image, conf=conf_thresh, verbose=False)

    detections = []
    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            for i in range(len(r.boxes)):
                bbox = r.boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(r.boxes.conf[i].cpu().numpy())
                cls_idx = int(r.boxes.cls[i].cpu().numpy())
                cls_name = queries[cls_idx] if cls_idx < len(queries) else "unknown"
                detections.append({
                    "bbox_xyxy": [round(x, 1) for x in bbox],
                    "confidence": round(conf, 4),
                    "class_name": cls_name,
                })
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(
    version_base=None,
    config_path="../third_party/vlmaps/config",
    config_name="object_goal_navigation_cfg.yaml",
)
def main(config: DictConfig) -> None:
    queries     = list(config.get("queries", ["chair", "table", "sofa", "bed", "toilet"]))
    sample_rate = int(config.get("sample_rate", 10))
    conf_thresh = float(config.get("conf_thresh", 0.25))
    model_name  = config.get("model", "yoloe-11l-seg.pt")

    # ── Resolve scene directory ──────────────────────────────────────────
    data_dir = Path(config.data_paths.vlmaps_data_dir)
    scene_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if config.scene_id >= len(scene_dirs):
        print(f"ERROR: scene_id={config.scene_id} but only {len(scene_dirs)} scenes found.")
        sys.exit(1)
    scene_dir = scene_dirs[config.scene_id]
    rgb_dir = scene_dir / "rgb"

    print(f"\n{'='*60}")
    print(f"  YOLOE Detection — scene {config.scene_id} ({scene_dir.name})")
    print(f"  Queries: {queries}")
    print(f"  Model: {model_name}  Sample rate: 1/{sample_rate}  Conf: {conf_thresh}")
    print(f"{'='*60}\n")

    # ── List RGB frames ──────────────────────────────────────────────────
    rgb_files = sorted(rgb_dir.glob("*.png"))
    if not rgb_files:
        print(f"ERROR: No RGB frames found in {rgb_dir}")
        sys.exit(1)
    sampled = rgb_files[::sample_rate]
    print(f"Total frames: {len(rgb_files)}  Sampled: {len(sampled)}\n")

    # ── Load model ───────────────────────────────────────────────────────
    print("Loading YOLOE model...")
    model = load_yoloe(model_name)
    print("Model loaded.\n")

    # ── Run detection ────────────────────────────────────────────────────
    all_detections = []
    total_dets = 0
    pbar = tqdm(sampled, desc="Detecting")
    for rgb_path in pbar:
        frame_id = int(rgb_path.stem)
        img = cv2.imread(str(rgb_path))
        if img is None:
            continue

        dets = detect_frame(model, img, queries, conf_thresh)
        for d in dets:
            d["frame_id"] = frame_id
        all_detections.extend(dets)
        total_dets += len(dets)
        pbar.set_postfix(detections=total_dets)

        # Show detections on frame
        if dets:
            vis = img.copy()
            for d in dets:
                x1, y1, x2, y2 = [int(v) for v in d["bbox_xyxy"]]
                label = f"{d['class_name']} {d['confidence']:.2f}"
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("YOLOE detections", vis)
            cv2.waitKey(1)

    # ── Summary ──────────────────────────────────────────────────────────
    det_counts = {}
    for d in all_detections:
        det_counts[d["class_name"]] = det_counts.get(d["class_name"], 0) + 1

    print(f"\nTotal detections: {total_dets}")
    for cls, cnt in sorted(det_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:20s}: {cnt}")

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = scene_dir / "yoloe_detections.json"
    payload = {
        "metadata": {
            "scene": scene_dir.name,
            "model": model_name,
            "queries": queries,
            "sample_rate": sample_rate,
            "conf_thresh": conf_thresh,
            "total_frames": len(rgb_files),
            "sampled_frames": len(sampled),
            "total_detections": total_dets,
            "detection_counts": det_counts,
        },
        "detections": all_detections,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {out_path}")

    # ── Top-down detection map ────────────────────────────────────────
    from vlmaps.utils.mapping_utils import cvt_pose_vec2tf, base_pos2grid_id_3d
    poses_path = scene_dir / "poses.txt"
    if poses_path.exists() and all_detections:
        poses = np.loadtxt(str(poses_path))
        gs = 1000
        cs = 0.05

        # Load obstacle map as background
        vlmap_dir = None
        for sub in sorted(scene_dir.iterdir()):
            if sub.is_dir() and sub.name.startswith("vlmap"):
                vlmap_dir = sub
                break
        if vlmap_dir and (vlmap_dir / "vlmaps.h5df").exists():
            import h5py
            with h5py.File(str(vlmap_dir / "vlmaps.h5df"), "r") as f:
                grid_pos = f["grid_pos"][:]
            # Build obstacle map
            canvas = np.zeros((gs, gs, 3), dtype=np.uint8)
            canvas[:] = 40  # dark gray background
            for pos in grid_pos:
                r, c, _ = pos
                if 0 <= r < gs and 0 <= c < gs:
                    canvas[r, c] = (80, 80, 80)  # building footprint
        else:
            canvas = np.full((gs, gs, 3), 40, dtype=np.uint8)

        # Color per class
        class_colors = {}
        color_list = [
            (0, 255, 0), (0, 165, 255), (255, 0, 255),
            (0, 255, 255), (255, 255, 0), (255, 100, 100),
            (100, 255, 100), (100, 100, 255),
        ]
        for i, q in enumerate(queries):
            class_colors[q] = color_list[i % len(color_list)]

        # Plot each detection at its camera position
        for d in all_detections:
            fid = d["frame_id"]
            if fid >= len(poses):
                continue
            tf = cvt_pose_vec2tf(poses[fid])
            pos = tf[:3, 3]
            row, col, _ = base_pos2grid_id_3d(gs, cs, pos[0], pos[1], pos[2])
            row, col = int(np.clip(row, 0, gs - 1)), int(np.clip(col, 0, gs - 1))
            color = class_colors.get(d["class_name"], (0, 255, 0))
            cv2.circle(canvas, (col, row), 3, color, -1)

        # Legend
        y_off = 20
        for cls, color in class_colors.items():
            cnt = det_counts.get(cls, 0)
            if cnt > 0:
                cv2.circle(canvas, (15, y_off), 5, color, -1)
                cv2.putText(canvas, f"{cls} ({cnt})", (28, y_off + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                y_off += 22

        # Save and show
        topdown_path = scene_dir / "yoloe_topdown.png"
        cv2.imwrite(str(topdown_path), canvas)
        print(f"Top-down map saved to {topdown_path}")
        cv2.imshow("YOLOE top-down detections", canvas)
        print("Press any key to close.")
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
