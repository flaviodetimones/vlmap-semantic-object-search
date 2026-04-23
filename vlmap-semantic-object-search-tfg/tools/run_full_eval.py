#!/usr/bin/env python3
"""
Run the full online 2x2 evaluation matrix and generate per-scene aggregates,
cross-scene aggregates, plots and a simple HTML report.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.aggregate_full_eval import build_cross_scene_rows
from tools.eval_methods import (
    METHOD_SPECS,
    default_queries_path,
    parse_scene_ids,
    scene_name_from_id,
)


def _resolve_queries_path(raw_queries: str | None, scene_name: str) -> Path:
    if not raw_queries:
        return default_queries_path(scene_name)
    path = Path(raw_queries)
    if path.is_dir():
        return path / f"{scene_name}.jsonl"
    return path


def _run_command(cmd: List[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True)
    return proc.returncode


def _selected_method_specs(raw_methods: str | None) -> List[dict]:
    if not raw_methods:
        return list(METHOD_SPECS)
    keys = [chunk.strip() for chunk in str(raw_methods).split(",") if chunk.strip()]
    known = {spec["key"]: spec for spec in METHOD_SPECS}
    bad = [key for key in keys if key not in known]
    if bad:
        raise SystemExit(f"Unknown method keys: {', '.join(bad)}")
    return [known[key] for key in keys]


def _scene_ids_from_config(path: Path) -> List[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "scene_ids" in payload:
        return [int(x) for x in payload["scene_ids"]]
    if "scenes" in payload:
        return [int(item["scene_id"]) for item in payload["scenes"]]
    raise SystemExit(f"{path} does not contain scene_ids or scenes[].scene_id")


def _write_report_html(out_root: Path, plot_paths: Dict[str, str]) -> Path:
    def read_text(path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def csv_to_html(path: Path) -> str:
        if not path.exists():
            return "<p><em>Missing CSV.</em></p>"
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return "<p><em>No rows.</em></p>"
        headers = list(rows[0].keys())
        parts = ["<table><thead><tr>"]
        parts.extend(f"<th>{h}</th>" for h in headers)
        parts.append("</tr></thead><tbody>")
        for row in rows:
            parts.append("<tr>")
            parts.extend(f"<td>{row.get(h, '')}</td>" for h in headers)
            parts.append("</tr>")
        parts.append("</tbody></table>")
        return "".join(parts)

    report_path = out_root / "report.html"
    cross_csv = out_root / "pipeline_full" / "aggregate_cross_scenes.csv"
    cross_md = out_root / "pipeline_full" / "aggregate_cross_scenes.md"
    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Full 2x2 evaluation report</title>",
        "<style>body{font-family:sans-serif;margin:24px;max-width:1400px} img{max-width:100%;border:1px solid #ddd} table{border-collapse:collapse;margin:12px 0 20px 0} td,th{border:1px solid #ddd;padding:6px 8px} pre{white-space:pre-wrap;background:#fafafa;padding:12px;border:1px solid #eee;overflow:auto}</style>",
        "</head><body>",
        "<h1>Full 2x2 evaluation report</h1>",
        "<ul>",
        "<li><a href='config.json'>config.json</a></li>",
        "<li><a href='pipeline_full/aggregate_cross_scenes.md'>aggregate_cross_scenes.md</a></li>",
        "</ul>",
    ]
    html += [
        "<h2>Cross-scene summary</h2>",
        csv_to_html(cross_csv),
    ]
    cross_md_text = read_text(cross_md)
    if cross_md_text:
        html += [
            "<h3>Cross-scene markdown</h3>",
            f"<pre>{cross_md_text}</pre>",
        ]
    if plot_paths:
        html.append("<h2>Plots</h2>")
        for title, rel_path in plot_paths.items():
            html.append(f"<h3>{title}</h3><img src='{rel_path}' alt='{title}'>")
    html.append("<h2>Per scene outputs</h2>")
    pipeline_root = out_root / "pipeline_full"
    if pipeline_root.exists():
        for scene_dir in sorted([p for p in pipeline_root.iterdir() if p.is_dir()]):
            rel_scene = f"pipeline_full/{scene_dir.name}"
            html += [
                f"<h3>{scene_dir.name}</h3>",
                f"<p><a href='{rel_scene}/compare_full.md'>compare_full.md</a> · "
                f"<a href='{rel_scene}/aggregate_metrics.md'>aggregate_metrics.md</a></p>",
                csv_to_html(scene_dir / "aggregate_metrics.csv"),
            ]
            agg_md = read_text(scene_dir / "aggregate_metrics.md")
            if agg_md:
                html.append(f"<details><summary>aggregate_metrics.md</summary><pre>{agg_md}</pre></details>")
    html.append("</body></html>")
    report_path.write_text("\n".join(html) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-ids", default=None,
                        help="Comma-separated scene ids.")
    parser.add_argument("--scenes-from-config", type=Path, default=None,
                        help="Optional config.json path from a previous run; extracts scene ids.")
    parser.add_argument("--queries", default=None,
                        help="Optional JSONL path (single scene) or directory holding {scene_name}.jsonl.")
    parser.add_argument("--dataset-type", default="hssd")
    parser.add_argument("--data-paths", default="hssd")
    parser.add_argument(
        "--scene-dataset-config-file",
        default="/workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json",
    )
    parser.add_argument("--policy-mode", choices=["heuristic", "hybrid", "llm"], default="hybrid")
    parser.add_argument("--yoloe-conf-thresh", type=float, default=0.30)
    parser.add_argument("--per-query-timeout", type=int, default=180)
    parser.add_argument("--methods", default=None,
                        help="Optional comma-separated subset of method keys.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.scenes_from_config:
        scene_ids = _scene_ids_from_config(args.scenes_from_config)
    elif args.scene_ids:
        scene_ids = parse_scene_ids(args.scene_ids)
    else:
        raise SystemExit("Provide --scene-ids or --scenes-from-config")

    if args.queries and len(scene_ids) > 1:
        qpath = Path(args.queries)
        if qpath.exists() and qpath.is_file():
            raise SystemExit("With multiple scene ids, --queries must be a directory containing {scene_name}.jsonl files.")

    selected_specs = _selected_method_specs(args.methods)
    args.out.mkdir(parents=True, exist_ok=True)

    run_config = {
        "created_at": datetime.now().isoformat(),
        "scene_ids": scene_ids,
        "dataset_type": args.dataset_type,
        "data_paths": args.data_paths,
        "scene_dataset_config_file": args.scene_dataset_config_file,
        "policy_mode": args.policy_mode,
        "yoloe_conf_thresh": args.yoloe_conf_thresh,
        "per_query_timeout": args.per_query_timeout,
        "resume": args.resume,
        "methods": selected_specs,
        "scenes": [],
    }

    queries_root = args.out / "queries"
    pipeline_root = args.out / "pipeline_full"
    plots_root = args.out / "plots"
    queries_root.mkdir(exist_ok=True)
    pipeline_root.mkdir(exist_ok=True)
    plots_root.mkdir(exist_ok=True)

    cross_scene_inputs: Dict[str, List[dict]] = {}

    for scene_id in scene_ids:
        scene_name = scene_name_from_id(scene_id, args.dataset_type, args.data_paths)
        if not scene_name:
            raise SystemExit(f"Could not resolve scene_name for scene_id={scene_id}")

        queries_path = _resolve_queries_path(args.queries, scene_name)
        if not queries_path.exists():
            raise SystemExit(f"Query JSONL not found for scene '{scene_name}': {queries_path}")

        copied_queries = queries_root / f"{scene_name}.jsonl"
        shutil.copy2(queries_path, copied_queries)

        scene_out = pipeline_root / scene_name
        scene_out.mkdir(parents=True, exist_ok=True)
        method_manifests: Dict[str, str] = {}

        for spec in selected_specs:
            method_out = scene_out / spec["key"]
            manifest_path = method_out / "manifest.json"
            if args.resume and manifest_path.exists() and not args.force:
                print(f"[resume] Skipping existing {scene_name}/{spec['key']}")
                method_manifests[spec["key"]] = str(manifest_path)
                continue

            method_out.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(REPO_ROOT / "tools" / "run_nav_eval.py"),
                "--queries", str(copied_queries),
                "--entrypoint", spec["entrypoint"],
                "--heatmap-mode", spec["heatmap_mode"],
                "--yoloe-conf-thresh", str(args.yoloe_conf_thresh),
                "--scene-id", str(scene_id),
                "--scene-name", scene_name,
                "--dataset-type", args.dataset_type,
                "--scene-dataset-config-file", args.scene_dataset_config_file,
                "--data-paths", args.data_paths,
                "--per-query-timeout", str(args.per_query_timeout),
                "--out", str(method_out),
            ]
            if spec["entrypoint"] == "executor":
                cmd.extend(["--policy-mode", args.policy_mode])
            rc = _run_command(cmd)
            if not manifest_path.exists():
                raise SystemExit(
                    f"Method {spec['key']} did not produce manifest.json (rc={rc})"
                )
            method_manifests[spec["key"]] = str(manifest_path)

        agg_cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "aggregate_full_eval.py"),
            "--out-csv", str(scene_out / "compare_full.csv"),
            "--out-md", str(scene_out / "compare_full.md"),
            "--aggregate-csv", str(scene_out / "aggregate_metrics.csv"),
            "--aggregate-md", str(scene_out / "aggregate_metrics.md"),
        ]
        for key in [spec["key"] for spec in selected_specs if spec["key"] in method_manifests]:
            agg_cmd.extend(["--manifest", f"{key}={method_manifests[key]}"])
        _run_command(agg_cmd)

        agg_rows = []
        agg_csv = scene_out / "aggregate_metrics.csv"
        if agg_csv.exists():
            with agg_csv.open("r", encoding="utf-8") as f:
                agg_rows = list(csv.DictReader(f))
        for row in agg_rows:
            if row.get("slice_kind") == "global":
                cross_scene_inputs.setdefault(row["method"], []).append({
                    k: (float(v) if k not in {"method", "slice_kind", "slice_value"} and v not in {"", "—"} else v)
                    for k, v in row.items()
                })

        run_config["scenes"].append({
            "scene_id": scene_id,
            "scene_name": scene_name,
            "queries_jsonl": str(copied_queries),
            "method_manifests": method_manifests,
            "outputs": {
                "compare_full_csv": str(scene_out / "compare_full.csv"),
                "compare_full_md": str(scene_out / "compare_full.md"),
                "aggregate_metrics_csv": str(scene_out / "aggregate_metrics.csv"),
                "aggregate_metrics_md": str(scene_out / "aggregate_metrics.md"),
            },
        })

    cross_rows = build_cross_scene_rows(cross_scene_inputs)
    cross_csv = pipeline_root / "aggregate_cross_scenes.csv"
    cross_md = pipeline_root / "aggregate_cross_scenes.md"
    if cross_rows:
        from tools.aggregate_full_eval import _write_csv, _write_cross_scene_md
        _write_csv(cross_rows, cross_csv)
        _write_cross_scene_md(cross_rows, cross_md)

    plot_paths = {}
    if cross_rows:
        plot_cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "plot_full_eval.py"),
            "--cross-scene-csv", str(cross_csv),
            "--out-dir", str(plots_root),
        ]
        _run_command(plot_cmd)
        for png in sorted(plots_root.glob("*.png")):
            plot_paths[png.stem] = str(png.relative_to(args.out))

    config_path = args.out / "config.json"
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    report_path = _write_report_html(args.out, plot_paths)
    print(f"Wrote {config_path}")
    print(f"Open: {report_path}")


if __name__ == "__main__":
    main()
