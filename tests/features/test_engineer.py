"""Tests for feature engineering."""

import numpy as np
import pytest
from src.data.loader import TripData
from src.features.candidates import Candidate, CandidateSet
from src.features.engineer import (
    compute_node_degrees,
    compute_arc_lengths,
    compute_adjacency,
    extract_candidate_features,
    build_feature_matrix,
)


def create_mock_trip() -> TripData:
    """Create a simple mock trip for testing."""
    # 4 nodes in a square
    nodes = np.array([
        [8.5, 49.8],  # 0
        [8.6, 49.8],  # 1
        [8.6, 49.9],  # 2
        [8.5, 49.9],  # 3
    ])
    
    # 4 arcs forming a loop
    arcs = np.array([
        [0, 1],  # 0: bottom edge
        [1, 2],  # 1: right edge
        [2, 3],  # 2: top edge
        [3, 0],  # 3: left edge
    ])
    
    # 3 GPS observations along bottom edge
    track = np.array([
        [8.52, 49.8, 1.0],
        [8.55, 49.8, 2.0],
        [8.58, 49.8, 3.0],
    ])
    
    # Ground truth: arc 0
    route = np.array([0, 0, 0])
    
    return TripData(
        trip_id="test",
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route
    )


def test_compute_node_degrees():
    arcs = np.array([
        [0, 1],
        [1, 2],
        [2, 0],
        [0, 2],  # Multiple outgoing from 0
    ])
    
    in_deg, out_deg = compute_node_degrees(arcs, 3)
    
    assert out_deg[0] == 2  # 0 -> 1, 0 -> 2
    assert out_deg[1] == 1  # 1 -> 2
    assert out_deg[2] == 1  # 2 -> 0
    
    assert in_deg[0] == 1   # 2 -> 0
    assert in_deg[1] == 1   # 0 -> 1
    assert in_deg[2] == 2   # 1 -> 2, 0 -> 2


def test_compute_arc_lengths():
    trip = create_mock_trip()
    lengths = compute_arc_lengths(trip.nodes, trip.arcs)
    
    assert len(lengths) == 4
    # All edges should be ~111km (1 degree at this latitude)
    for l in lengths:
        assert 100000 < l < 120000


def test_compute_adjacency():
    trip = create_mock_trip()
    adj = compute_adjacency(trip.arcs, len(trip.arcs))
    
    # Arc 0 (0->1) connects to arc 1 (1->2)
    assert 1 in adj[0]
    # Arc 1 (1->2) connects to arc 2 (2->3)
    assert 2 in adj[1]
    # Arc 2 (2->3) connects to arc 3 (3->0)
    assert 3 in adj[2]
    # Arc 3 (3->0) connects to arc 0 (0->1)
    assert 0 in adj[3]


def test_extract_candidate_features():
    trip = create_mock_trip()
    
    arc_lengths = compute_arc_lengths(trip.nodes, trip.arcs)
    in_deg, out_deg = compute_node_degrees(trip.arcs, len(trip.nodes))
    adj = compute_adjacency(trip.arcs, len(trip.arcs))
    
    # Candidate on arc 0 (correct)
    cand = Candidate(arc_id=0, distance=10.0, along_track_dist=5000.0, bearing_diff=5.0, midpoint_dist=10000.0)
    cand_set = CandidateSet(0, 8.52, 49.8, 1.0, [cand])
    
    fv = extract_candidate_features(cand_set, trip, arc_lengths, in_deg, out_deg, adj)
    
    assert fv.features.shape == (11,)
    assert len(fv.feature_names) == 11
    assert fv.features[0] == 10.0  # perp_distance
    assert fv.features[10] == 0.0  # is_continuation (no prev)


def test_build_feature_matrix():
    trip = create_mock_trip()
    
    # Create candidates for each observation
    candidate_sets = []
    for i in range(3):
        candidates = [
            Candidate(arc_id=0, distance=10.0, along_track_dist=5000.0, bearing_diff=5.0, midpoint_dist=10000.0),
            Candidate(arc_id=1, distance=100.0, along_track_dist=1000.0, bearing_diff=90.0, midpoint_dist=15000.0),
            Candidate(arc_id=2, distance=200.0, along_track_dist=2000.0, bearing_diff=180.0, midpoint_dist=20000.0),
        ]
        candidate_sets.append(CandidateSet(i, trip.track[i,0], trip.track[i,1], trip.track[i,2], candidates))
    
    X, y, feature_names, mapping = build_feature_matrix(candidate_sets, trip, trip.route)
    
    # 3 observations * 3 candidates = 9 rows
    assert X.shape == (9, 11)
    assert y.shape == (9,)
    assert len(feature_names) == 11
    assert len(mapping) == 9
    
    # Labels: first candidate (arc 0) should be 1 for all observations
    assert y[0] == 1   # obs 0, cand 0
    assert y[1] == 0   # obs 0, cand 1
    assert y[2] == 0   # obs 0, cand 2
    assert y[3] == 1   # obs 1, cand 0
    assert y[4] == 0   # obs 1, cand 1
    assert y[5] == 0   # obs 1, cand 2
    assert y[6] == 1   # obs 2, cand 0
    assert y[7] == 0   # obs 2, cand 1
    assert y[8] == 0   # obs 2, cand 2
    
    # Check mapping
    assert mapping[0] == (0, 0)
    assert mapping[3] == (1, 0)
    assert mapping[6] == (2, 0)


def test_no_nan_features():
    trip = create_mock_trip()
    
    candidate_sets = []
    for i in range(3):
        candidates = [
            Candidate(arc_id=0, distance=10.0, along_track_dist=5000.0, bearing_diff=5.0, midpoint_dist=10000.0),
            Candidate(arc_id=1, distance=100.0, along_track_dist=1000.0, bearing_diff=90.0, midpoint_dist=15000.0),
        ]
        candidate_sets.append(CandidateSet(i, trip.track[i,0], trip.track[i,1], trip.track[i,2], candidates))
    
    X, y, _, _ = build_feature_matrix(candidate_sets, trip, trip.route)
    
    assert not np.any(np.isnan(X))
    assert not np.any(np.isinf(X))
    assert np.all(np.isfinite(X))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])