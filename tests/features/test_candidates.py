"""Tests for candidate generation."""

import numpy as np
import pytest
from src.features.candidates import (
    haversine_distance,
    bearing,
    perpendicular_distance_to_segment,
    compute_arc_geometry,
    Candidate,
    CandidateSet,
)


def test_haversine_distance():
    # Distance between two known points
    # Approximate: 1 degree lat ~= 111km at equator
    dist = haversine_distance(0, 0, 0, 1)
    assert 110000 < dist < 112000  # ~111km
    
    # Same point
    dist = haversine_distance(8.5, 49.8, 8.5, 49.8)
    assert dist == 0


def test_bearing():
    # North
    brng = bearing(0, 0, 0, 1)
    assert abs(brng - 0) < 1e-10
    
    # East
    brng = bearing(0, 0, 1, 0)
    assert abs(brng - 90) < 1e-10
    
    # South
    brng = bearing(0, 0, 0, -1)
    assert abs(brng - 180) < 1e-10
    
    # West
    brng = bearing(0, 0, -1, 0)
    assert abs(brng - 270) < 1e-10


def test_perpendicular_distance_to_segment():
    # Point on segment
    dist, along = perpendicular_distance_to_segment(0, 0, -1, 0, 1, 0)
    assert abs(dist) < 1e-10
    
    # Point perpendicular to segment midpoint
    dist, along = perpendicular_distance_to_segment(0, 1, -1, 0, 1, 0)
    assert abs(dist - 1.0) < 1e-10
    assert abs(along) < 1e-10
    
    # Point beyond segment end
    dist, along = perpendicular_distance_to_segment(2, 0, -1, 0, 1, 0)
    assert abs(dist - 1.0) < 1e-10  # Distance to endpoint
    assert abs(along - 2.0) < 1e-10  # Along track from start
    
    # Degenerate segment (point)
    dist, along = perpendicular_distance_to_segment(1, 1, 0, 0, 0, 0)
    assert abs(dist - np.sqrt(2)) < 1e-10
    assert along == 0


def test_compute_arc_geometry():
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    arcs = np.array([
        [0, 1],  # Arc 0: (0,0) -> (1,0)
        [1, 2],  # Arc 1: (1,0) -> (1,1)
        [2, 3],  # Arc 2: (1,1) -> (0,1)
        [3, 0],  # Arc 3: (0,1) -> (0,0)
    ])
    
    midpoints, start_points, end_points = compute_arc_geometry(nodes, arcs)
    
    assert midpoints.shape == (4, 2)
    assert start_points.shape == (4, 2)
    assert end_points.shape == (4, 2)
    
    # Check arc 0
    assert np.allclose(start_points[0], [0.0, 0.0])
    assert np.allclose(end_points[0], [1.0, 0.0])
    assert np.allclose(midpoints[0], [0.5, 0.0])


def test_candidate_sorting():
    c1 = Candidate(arc_id=1, distance=10.0, along_track_dist=5.0, bearing_diff=30.0, midpoint_dist=12.0)
    c2 = Candidate(arc_id=2, distance=5.0, along_track_dist=3.0, bearing_diff=10.0, midpoint_dist=8.0)
    c3 = Candidate(arc_id=3, distance=15.0, along_track_dist=8.0, bearing_diff=45.0, midpoint_dist=18.0)
    
    candidates = [c1, c2, c3]
    candidates.sort()
    
    assert candidates[0].arc_id == 2  # Smallest distance
    assert candidates[1].arc_id == 1
    assert candidates[2].arc_id == 3


def test_candidate_set_miss():
    candidates = [
        Candidate(arc_id=1, distance=10.0, along_track_dist=5.0, bearing_diff=30.0, midpoint_dist=12.0),
        Candidate(arc_id=2, distance=5.0, along_track_dist=3.0, bearing_diff=10.0, midpoint_dist=8.0),
    ]
    
    # Ground truth in candidates
    cs1 = CandidateSet(0, 0, 0, 0, candidates, ground_truth_arc=2)
    assert not cs1.miss
    
    # Ground truth not in candidates
    cs2 = CandidateSet(0, 0, 0, 0, candidates, ground_truth_arc=5)
    assert cs2.miss
    
    # No ground truth
    cs3 = CandidateSet(0, 0, 0, 0, candidates, ground_truth_arc=None)
    assert not cs3.miss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])