"""vlmap_semantic_server — semantic queries against the VLMap.

Sprint 1: stub. Sprint 2 will wrap:
  - third_party/vlmaps/vlmaps/map/vlmap.py  (VLMap.load + scoring)
  - third_party/vlmaps/vlmaps/utils/room_provider.py  (room polygons / centroids)
  - third_party/vlmaps/vlmaps/utils/room_priors.py  (object → room priors)
"""

from .centroid_voronoi import (
    UNASSIGNED_ROOM_ID,
    RoomCentroid,
    VoronoiRoomProvider,
    build_voronoi_room_map,
    dump_room_centroids,
    load_room_centroids,
)
from .dynamic_context import DynamicRoomContext, DynamicRoomEntry, load_dynamic_room_context
from .heatmap import extract_top_candidates, load_heatmap_array, normalize_to_occupancy

__all__ = [
    "UNASSIGNED_ROOM_ID",
    "RoomCentroid",
    "VoronoiRoomProvider",
    "DynamicRoomContext",
    "DynamicRoomEntry",
    "build_voronoi_room_map",
    "dump_room_centroids",
    "extract_top_candidates",
    "load_dynamic_room_context",
    "load_heatmap_array",
    "load_room_centroids",
    "normalize_to_occupancy",
]
