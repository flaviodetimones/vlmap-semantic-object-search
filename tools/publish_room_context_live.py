#!/usr/bin/env python3
"""Publish live semantic room context from tfg-sim to tfg-ros via rosbridge.

Symmetric to ``send_ros_semantic_goal.py``: instead of pushing a navigation
goal, this pushes the semantic room layout (rooms + centroids + priors) so the
ROS side can resolve ``actual_room`` from the robot's real final pose instead
of echoing the requested room (Phase 3).

The payload is a latched JSON ``std_msgs/String`` on ``/vlmap/room_context``,
matching ``vlmap_semantic_server.dynamic_context.load_dynamic_room_context``.

Three ways to build the context:

  --demo
      A small built-in 3-room context (kitchen/office/bedroom) whose
      centroids match simple smoke-test goals. Coordinates are already in
      the ROS /map frame.

  --context-file PATH
      Publish a ready JSON file verbatim (after validating it parses).

  --scene ID
      Build from the HSSD scene's room_map/regions.json. Grid centroids are
      converted to metres with --resolution/--origin-x/--origin-y. Note: the
      result is only physically meaningful once tfg-ros navigates inside the
      same map frame (Phase 4+); for the Phase 3 contract smoke prefer --demo.

A latched topic stays available to late subscribers only while the publisher
is alive, so by default this tool keeps the rosbridge connection open until
Ctrl+C. Use --once to publish and exit immediately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(
    0, str(ROOT / "ros_ws" / "src" / "vlmap_semantic_server" / "src")
)

from tfg_nav_contracts.navigation_backend import RoomContextPublisher  # noqa: E402

# Optional: validate locally before publishing (no ROS runtime needed).
try:
    from vlmap_semantic_server.dynamic_context import load_dynamic_room_context
except Exception:  # pragma: no cover
    load_dynamic_room_context = None


_DEMO_CONTEXT = {
    "rooms": [
        {"room_id": "kitchen", "x": 0.0, "y": 0.0, "aliases": ["cocina"]},
        {"room_id": "office", "x": 2.5, "y": 0.0, "aliases": ["oficina", "study"]},
        {"room_id": "bedroom", "x": 0.0, "y": 3.0, "aliases": ["dormitorio"]},
        {"room_id": "living room", "x": -2.5, "y": 0.0, "aliases": ["salon", "living_room"]},
    ],
    "priors": {
        "laptop": [["office", 0.7], ["bedroom", 0.3]],
        "mug": [["kitchen", 0.8], ["office", 0.2]],
        "bottle": [["kitchen", 1.0]],
        "ball": [["living room", 0.6], ["bedroom", 0.4]],
        "teddy bear": [["bedroom", 0.9]],
    },
}


def _build_from_scene(
    scene_id: str, resolution: float, origin_x: float, origin_y: float
) -> dict:
    """Build a context dict from a scene's room_map/regions.json (grid cells)."""
    base = ROOT.parent / "data" / "vlmaps_dataset_hssd"
    # Resolve the scene directory: accept an explicit name or a numeric index.
    candidates = sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []
    scene_dir = None
    if scene_id.isdigit() and candidates:
        idx = int(scene_id)
        if 0 <= idx < len(candidates):
            scene_dir = candidates[idx]
    if scene_dir is None:
        named = base / scene_id
        scene_dir = named if named.exists() else None
    if scene_dir is None:
        raise SystemExit(f"could not resolve scene '{scene_id}' under {base}")

    regions_path = scene_dir / "room_map" / "regions.json"
    if not regions_path.exists():
        raise SystemExit(f"no regions.json for scene at {regions_path}")
    data = json.loads(regions_path.read_text(encoding="utf-8"))
    regions = data.get("regions", {})

    rooms = []
    for label, instances in regions.items():
        if not instances:
            continue
        # Largest-area instance is the representative centroid for the room.
        best = max(instances, key=lambda r: float(r.get("area", 0.0)))
        row, col = best["centroid"]
        rooms.append(
            {
                "room_id": str(label),
                "row": float(row),
                "col": float(col),
                "metadata": {"scene": scene_dir.name, "area": best.get("area")},
            }
        )
    payload = {"rooms": rooms, "_grid": {
        "resolution": resolution, "origin_x": origin_x, "origin_y": origin_y}}
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="tfg-ros", help="rosbridge host")
    p.add_argument("--port", type=int, default=9090, help="rosbridge websocket port")
    p.add_argument("--topic", default="/vlmap/room_context", help="latched context topic")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--demo", action="store_true", help="publish the built-in demo context")
    src.add_argument("--context-file", help="publish a ready JSON context file verbatim")
    src.add_argument("--scene", help="build context from a HSSD scene id/name (grid centroids)")
    p.add_argument("--resolution", type=float, default=0.05, help="grid resolution m/cell (--scene)")
    p.add_argument("--origin-x", type=float, default=0.0, help="grid origin x in metres (--scene)")
    p.add_argument("--origin-y", type=float, default=0.0, help="grid origin y in metres (--scene)")
    p.add_argument("--once", action="store_true", help="publish and exit (do not keep latched)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.demo:
        payload = dict(_DEMO_CONTEXT)
    elif args.context_file:
        payload = json.loads(Path(args.context_file).read_text(encoding="utf-8"))
    else:
        payload = _build_from_scene(args.scene, args.resolution, args.origin_x, args.origin_y)

    rooms = payload.get("rooms", [])
    priors = payload.get("priors")
    maps = payload.get("maps")

    # Local validation so we fail fast before hitting rosbridge.
    if load_dynamic_room_context is not None:
        grid = payload.get("_grid", {})
        load_dynamic_room_context(
            json.dumps(payload),
            resolution=grid.get("resolution", args.resolution),
            origin_x=grid.get("origin_x", args.origin_x),
            origin_y=grid.get("origin_y", args.origin_y),
        )

    publisher = RoomContextPublisher(host=args.host, port=args.port, context_topic=args.topic)
    data = publisher.publish(rooms, priors=priors, maps=maps)
    room_ids = [r.get("room_id") or r.get("label") for r in rooms]
    print(f"published {len(rooms)} rooms to {args.topic}: {room_ids}")
    print(json.dumps(json.loads(data), indent=2, ensure_ascii=False))

    if args.once:
        return
    print("keeping latched context alive (Ctrl+C to stop)...")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
