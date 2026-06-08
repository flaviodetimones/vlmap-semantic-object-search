#!/usr/bin/env python3
"""Fine-tune YOLOE on a synthetic YOLO segmentation dataset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True,
                   help="Path to data.yaml or to a dataset directory containing data.yaml.")
    p.add_argument("--weights", default="/workspace/yoloe-11l-seg.pt",
                   help="Base YOLOE .pt weights.")
    p.add_argument("--project", type=Path, default=Path("/shared/yoloe_finetune"))
    p.add_argument("--name", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default=None,
                   help="Ultralytics device string. Defaults to 0 when CUDA is available, otherwise cpu.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--freeze", type=int, default=0,
                   help="Number of early layers to freeze. 0 leaves all layers trainable.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = args.data / "data.yaml" if args.data.is_dir() else args.data
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")

    device = args.device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    name = args.name or f"yoloe_synth_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config = {
        "created_at": datetime.now().isoformat(),
        "data": str(data_yaml),
        "weights": str(weights),
        "project": str(args.project),
        "name": name,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "workers": args.workers,
        "patience": args.patience,
        "freeze": args.freeze,
    }
    print(json.dumps(config, indent=2))
    if args.dry_run:
        return

    model = YOLO(str(weights))
    result = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(args.project),
        name=name,
        patience=args.patience,
        freeze=args.freeze,
        amp=True,
    )
    run_dir = Path(result.save_dir)
    (run_dir / "finetune_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[train] run: {run_dir}")
    print(f"[train] best weights: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
