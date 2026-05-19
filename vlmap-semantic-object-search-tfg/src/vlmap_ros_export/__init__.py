"""Export VLMap artifacts in formats consumed by the ROS containers.

This package lives outside ``third_party/vlmaps`` on purpose: it is the
``tfg-sim`` side of the file-backed contract used by ``vlmap_semantic_server``
in ``tfg-ros`` (see ``project_migration_to_ros.tex``, bitácora 2026-05-13).
"""

from .heatmap_dumper import (
    dump_heatmap_npy,
    pool_voxel_scores_to_2d,
)

__all__ = [
    "dump_heatmap_npy",
    "pool_voxel_scores_to_2d",
]
