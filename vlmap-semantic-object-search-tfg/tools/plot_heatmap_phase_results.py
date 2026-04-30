#!/usr/bin/env python3
"""Generate compact tables and plots for a heatmap phase run."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


METRICS = [
    ("hit@1", True),
    ("hit@5", True),
    ("mass_in_expected_ratio", True),
    ("iou_topmass50", True),
    ("wrong_room_mass_ratio", False),
    ("n_components", False),
]


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _to_float(value: str) -> float:
    return float(value) if value not in {"", "nan", "NaN", "None"} else math.nan


def _fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _winner(raw: float, clean: float, higher_better: bool) -> str:
    if math.isnan(raw) and math.isnan(clean):
        return "-"
    if math.isnan(raw):
        return "postprocessed"
    if math.isnan(clean):
        return "raw"
    if abs(raw - clean) < 1e-12:
        return "tie"
    if higher_better:
        return "postprocessed" if clean > raw else "raw"
    return "postprocessed" if clean < raw else "raw"


def _write_delta_table(rows: list[dict[str, str]], out_dir: Path) -> None:
    csv_out = out_dir / "heatmap_delta_summary.csv"
    md_out = out_dir / "heatmap_delta_summary.md"
    header = [
        "scene_name",
        "n_queries",
        "delta_hit@1",
        "delta_hit@5",
        "delta_mass_in_expected_ratio",
        "delta_iou_topmass50",
        "delta_wrong_room_mass_ratio",
        "delta_n_components",
    ]
    table_rows = []
    for row in rows:
        out = {
            "scene_name": row["scene_name"],
            "n_queries": row["n_queries"],
        }
        for metric, _ in METRICS:
            raw = _to_float(row[f"raw_{metric}"])
            clean = _to_float(row[f"clean_{metric}"])
            out[f"delta_{metric}"] = clean - raw
        table_rows.append(out)

    with csv_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in table_rows:
            writer.writerow(row)

    lines = [
        "# Heatmap Delta Summary",
        "",
        "| Scene | N | Δ hit@1 | Δ hit@5 | Δ mass_in_expected_ratio | Δ iou_topmass50 | Δ wrong_room_mass_ratio | Δ n_components |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in table_rows:
        lines.append(
            "| {scene_name} | {n_queries} | {d1} | {d5} | {dm} | {di} | {dw} | {dn} |".format(
                scene_name=row["scene_name"],
                n_queries=row["n_queries"],
                d1=_fmt(row["delta_hit@1"]),
                d5=_fmt(row["delta_hit@5"]),
                dm=_fmt(row["delta_mass_in_expected_ratio"]),
                di=_fmt(row["delta_iou_topmass50"]),
                dw=_fmt(row["delta_wrong_room_mass_ratio"]),
                dn=_fmt(row["delta_n_components"]),
            )
        )
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_scene_table(rows: list[dict[str, str]], out_dir: Path) -> None:
    md_out = out_dir / "heatmap_scene_comparison.md"
    blocks = ["# Heatmap Scene Comparison", ""]
    for scene_row in rows:
        scene = scene_row["scene_name"]
        n_queries = scene_row["n_queries"]
        blocks.append(f"## {scene}")
        blocks.append("")
        blocks.append(f"- Queries: `{n_queries}`")
        blocks.append("")
        blocks.append("| Metric | Raw | Postprocessed | Winner |")
        blocks.append("| --- | ---: | ---: | --- |")
        for metric, higher_better in METRICS:
            raw = _to_float(scene_row[f"raw_{metric}"])
            clean = _to_float(scene_row[f"clean_{metric}"])
            blocks.append(
                f"| {metric} | {_fmt(raw)} | {_fmt(clean)} | {_winner(raw, clean, higher_better)} |"
            )
        blocks.append("")
    md_out.write_text("\n".join(blocks), encoding="utf-8")


def _grouped_bars(rows: list[dict[str, str]], metric: str, out_dir: Path) -> None:
    scenes = [row["scene_name"] for row in rows]
    raw_vals = [_to_float(row[f"raw_{metric}"]) for row in rows]
    clean_vals = [_to_float(row[f"clean_{metric}"]) for row in rows]
    xs = list(range(len(scenes)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([x - width / 2 for x in xs], raw_vals, width=width, label="raw", color="#8aa29e")
    ax.bar([x + width / 2 for x in xs], clean_vals, width=width, label="postprocessed", color="#2f6690")
    ax.set_xticks(xs)
    ax.set_xticklabels(scenes, rotation=12, ha="right")
    ax.set_title(f"Heatmap Phase — {metric}")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / f"bar_{metric.replace('@', 'at_')}.png", dpi=180)
    plt.close(fig)


def _delta_bars(rows: list[dict[str, str]], out_dir: Path) -> None:
    metrics = ["hit@1", "mass_in_expected_ratio", "wrong_room_mass_ratio"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.8))
    scenes = [row["scene_name"] for row in rows]
    for ax, metric in zip(axes, metrics):
        deltas = [
            _to_float(row[f"clean_{metric}"]) - _to_float(row[f"raw_{metric}"])
            for row in rows
        ]
        colors = ["#2f6690" if val >= 0 else "#b56576" for val in deltas]
        ax.bar(scenes, deltas, color=colors)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(f"Δ {metric}")
        ax.tick_params(axis="x", rotation=12)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Heatmap Phase — Delta Bars (postprocessed - raw)")
    fig.tight_layout()
    fig.savefig(out_dir / "delta_bars_core.png", dpi=180)
    plt.close(fig)


def _delta_lines(rows: list[dict[str, str]], out_dir: Path) -> None:
    scenes = [row["scene_name"] for row in rows]
    line_metrics = [
        ("hit@1", "#2f6690"),
        ("mass_in_expected_ratio", "#d17b0f"),
        ("wrong_room_mass_ratio", "#b56576"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric, color in line_metrics:
        deltas = [
            _to_float(row[f"clean_{metric}"]) - _to_float(row[f"raw_{metric}"])
            for row in rows
        ]
        ax.plot(scenes, deltas, marker="o", linewidth=2, label=metric, color=color)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Heatmap Phase — Delta Lines (postprocessed - raw)")
    ax.set_ylabel("Delta")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "delta_lines_core.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Heatmap phase run directory containing aggregate_cross_scenes.csv")
    parser.add_argument(
        "--out-dir",
        help="Output directory for tables and plots (default: <run_dir>/plots)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    csv_path = run_dir / "aggregate_cross_scenes.csv"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(csv_path)
    _write_delta_table(rows, out_dir)
    _write_scene_table(rows, out_dir)
    for metric, _ in METRICS:
        _grouped_bars(rows, metric, out_dir)
    _delta_bars(rows, out_dir)
    _delta_lines(rows, out_dir)

    print(f"[plot] wrote tables and plots to {out_dir}")


if __name__ == "__main__":
    main()
