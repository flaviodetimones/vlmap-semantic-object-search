#!/usr/bin/env python3
"""Fetch and patch the public Toyota HSR description packages for ROS Noetic.

The TRI repositories are intentionally kept outside this git repository under
``ros_ws/src/external``. This helper makes the setup reproducible without
vendoring large mesh assets or third-party git history.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


REPOS = {
    "hsr_meshes": "https://github.com/ToyotaResearchInstitute/hsr_meshes.git",
    "hsr_description": "https://github.com/ToyotaResearchInstitute/hsr_description.git",
}


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _clone_or_update(dst: Path, url: str, *, depth: int) -> None:
    if (dst / ".git").exists():
        _run(["git", "fetch", "--depth", str(depth), "origin"], cwd=dst)
        _run(["git", "checkout", "origin/master"], cwd=dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", str(depth), url, str(dst)])


def _patch_hsr_description_for_noetic(hsr_description_dir: Path) -> None:
    common = hsr_description_dir / "urdf" / "common.xacro"
    text = common.read_text(encoding="utf-8")
    replacements = {
        "<macro ": "<xacro:macro ",
        "</macro>": "</xacro:macro>",
        '<transmission name="${prefix}_transmission">': '<transmission name="${joint}_transmission">',
    }
    patched = text
    for old, new in replacements.items():
        patched = patched.replace(old, new)
    if patched != text:
        common.write_text(patched, encoding="utf-8")
        print(f"Patched Noetic-compatible xacro syntax in {common}")
    else:
        print(f"No xacro compatibility patch needed for {common}")

    for xacro_file in hsr_description_dir.rglob("*.xacro"):
        text = xacro_file.read_text(encoding="utf-8")
        patched = text.replace("<insert_block ", "<xacro:insert_block ")
        patched = patched.replace("</insert_block>", "</xacro:insert_block>")
        if patched != text:
            xacro_file.write_text(patched, encoding="utf-8")
            print(f"Patched xacro insert_block syntax in {xacro_file}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "ros_ws",
        help="Catkin workspace root. Defaults to this repo's ros_ws.",
    )
    parser.add_argument("--depth", type=int, default=1)
    args = parser.parse_args()

    external = args.workspace / "src" / "external"
    for name, url in REPOS.items():
        _clone_or_update(external / name, url, depth=args.depth)
    _patch_hsr_description_for_noetic(external / "hsr_description")

    print("\nNext validation:")
    print("  docker exec tfg-ros bash -lc 'cd /ros_ws && catkin build'")
    print("  docker exec tfg-ros bash -lc 'source /opt/ros/noetic/setup.bash && source /ros_ws/devel/setup.bash && rosrun vlmap_bringup hsr_stack_check'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
