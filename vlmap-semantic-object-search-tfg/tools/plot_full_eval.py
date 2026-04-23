#!/usr/bin/env python3
"""
Generate compact plots from the cross-scene aggregate CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MAIN_METRICS = [
    ("sr", "SR"),
    ("object_sr", "Object SR"),
    ("cfr", "CFR"),
    ("ct2r", "CT2R"),
    ("rooms_before_success", "Rooms Before Success"),
    ("wrong_visits", "Wrong Visits"),
    ("mean_pose_updates", "Mean Pose Updates"),
    ("early_stop_rate", "Early Stop Rate"),
]


def _load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k, v in list(row.items()):
            if k in {"method"}:
                continue
            row[k] = float(v) if v not in {"", "—"} else float("nan")
    return rows


def _method_style(method: str) -> tuple[str, str]:
    color = "#4E79A7" if method.startswith("Ob") else "#E15759"
    hatch = "//" if method.endswith("Hp") else ""
    return color, hatch


def _plot_bar_metric(rows: list[dict], metric: str, title: str, out_path: Path) -> None:
    methods = [r["method"] for r in rows]
    means = [r[f"{metric}_mean"] for r in rows]
    stds = [r[f"{metric}_std"] for r in rows]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=4)
    for bar, method in zip(bars, methods):
        color, hatch = _method_style(method)
        bar.set_color(color)
        bar.set_hatch(hatch)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_heatmap(rows: list[dict], out_path: Path) -> None:
    metrics = [m for m, _ in MAIN_METRICS]
    labels = [label for _, label in MAIN_METRICS]
    methods = [r["method"] for r in rows]
    arr = np.array([[r[f"{metric}_mean"] for metric in metrics] for r in rows], dtype=float)

    # Normalize column-wise so the heatmap is readable across heterogeneous metrics.
    norm = arr.copy()
    for j in range(norm.shape[1]):
        col = norm[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            continue
        lo = np.nanmin(col[finite])
        hi = np.nanmax(col[finite])
        if hi - lo > 1e-9:
            norm[:, j] = (col - lo) / (hi - lo)
        else:
            norm[:, j] = 0.5
        if metrics[j] in {"rooms_before_success", "wrong_visits", "mean_pose_updates", "early_stop_rate"}:
            norm[:, j] = 1.0 - norm[:, j]

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(norm, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Full 2x2 evaluation summary heatmap")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-scene-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = _load_rows(args.cross_scene_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for metric, title in MAIN_METRICS:
        _plot_bar_metric(rows, metric, title, args.out_dir / f"{metric}.png")
    _plot_heatmap(rows, args.out_dir / "summary_heatmap.png")


if __name__ == "__main__":
    main()
