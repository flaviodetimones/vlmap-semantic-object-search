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

__all__ = [
    "UNASSIGNED_ROOM_ID",
    "RoomCentroid",
    "VoronoiRoomProvider",
    "build_voronoi_room_map",
    "dump_room_centroids",
    "load_room_centroids",
]
