"""Offline tests for the ROS centroid/Voronoi room labeler.

These tests intentionally avoid ROS, Habitat and the VLMaps submodule.
They validate the first reusable Sprint 2 block in isolation.
"""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros_ws" / "src" / "vlmap_semantic_server" / "src"))

from vlmap_semantic_server import (
    UNASSIGNED_ROOM_ID,
    RoomCentroid,
    VoronoiRoomProvider,
    build_voronoi_room_map,
    dump_room_centroids,
    load_room_centroids,
)


def test_build_voronoi_room_map_assigns_only_free_cells():
    occ = np.array(
        [
            [100, 0, 0, 0, 100],
            [100, 0, 0, 0, 100],
            [100, 0, -1, 0, 100],
            [100, 0, 0, 0, 100],
            [100, 100, 100, 100, 100],
        ],
        dtype=np.int32,
    )
    centroids = [
        RoomCentroid("kitchen", row=1, col=1),
        RoomCentroid("office", row=1, col=3),
    ]

    room_map, labels, regions = build_voronoi_room_map(occ, centroids)

    assert labels == ["kitchen", "office"]
    assert room_map[0, 0] == UNASSIGNED_ROOM_ID
    assert room_map[2, 2] == UNASSIGNED_ROOM_ID
    assert labels[room_map[1, 1]] == "kitchen"
    assert labels[room_map[1, 3]] == "office"
    assert labels[room_map[3, 1]] == "kitchen"
    assert labels[room_map[3, 3]] == "office"
    assert regions[0]["area"] > 0
    assert regions[1]["area"] > 0


def test_voronoi_ties_break_by_input_order():
    occ = np.zeros((3, 3), dtype=np.int32)
    centroids = [
        RoomCentroid("left", row=1, col=0),
        RoomCentroid("right", row=1, col=2),
    ]
    room_map, labels, _ = build_voronoi_room_map(occ, centroids)

    # Cell (1,1) is equidistant to both centroids; first centroid wins.
    assert labels[room_map[1, 1]] == "left"


def test_provider_exposes_room_queries():
    occ = np.zeros((4, 4), dtype=np.int32)
    provider = VoronoiRoomProvider(
        occ,
        [
            RoomCentroid("Kitchen", row=0, col=0),
            RoomCentroid("Office", row=3, col=3),
        ],
    )

    assert provider.is_available() is True
    assert provider.list_rooms() == ["Kitchen", "Office"]
    assert provider.resolve_room_name("kitchen") == "Kitchen"
    assert provider.resolve_room_name("OFFICE") == "Office"
    assert provider.get_room_centroid("kitchen") == (0.0, 0.0)
    assert provider.get_room_at_cell(0, 1) == "Kitchen"
    assert provider.get_room_at_cell(3, 2) == "Office"
    assert provider.get_room_at_cell(-1, 0) is None


def test_centroid_json_round_trip(tmp_path):
    path = tmp_path / "room_centroids.json"
    centroids = [
        RoomCentroid("kitchen", 10, 20, metadata={"source": "rviz"}),
        RoomCentroid("bedroom", 30.5, 40.0),
    ]

    dump_room_centroids(path, centroids)
    restored = load_room_centroids(path)

    assert restored == centroids
