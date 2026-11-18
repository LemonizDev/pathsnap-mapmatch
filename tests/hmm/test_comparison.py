"""Tests for HMM comparison utility."""

import numpy as np
import pytest

from src.data.loader import TripData
from src.features.candidates import Candidate, CandidateSet
from src.hmm.comparison import (
    ComparisonConfig,
    ComparisonResult,
    compare_methods,
    compute_route_continuity,
    compute_top1_accuracy,
)
from src.hmm.model_interface import MockModelInterface


class TestComputeRouteContinuity:
    def test_connected_route(self):
        """Test continuity on fully connected route."""
        arcs = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
        ])
        route = np.array([0, 1, 2])
        
        cont = compute_route_continuity(route, arcs)
        assert cont == 1.0
    
    def test_disconnected_route(self):
        """Test continuity on disconnected route."""
        arcs = np.array([
            [0, 1],
            [1, 2],
            [3, 4],  # Disconnected
        ])
        route = np.array([0, 1, 2])  # Two transitions: 0->1 (connected), 1->2 (connected)
        # Actually 0->1: arc 0 ends at 1, arc 1 starts at 1 -> connected
        # 1->2: arc 1 ends at 2, arc 2 starts at 3 -> disconnected!
        
        cont = compute_route_continuity(route, arcs)
        assert cont == 1.0 / 2.0  # 1 connected out of 2 transitions
    
    def test_single_arc(self):
        """Test continuity with single arc."""
        arcs = np.array([[0, 1]])
        route = np.array([0])
        
        cont = compute_route_continuity(route, arcs)
        assert cont == 1.0
    
    def test_empty_route(self):
        """Test continuity with empty route."""
        arcs = np.array([[0, 1]])
        route = np.array([])
        
        cont = compute_route_continuity(route, arcs)
        assert cont == 1.0
    
    def test_invalid_arc_ids(self):
        """Test handling of invalid arc IDs."""
        arcs = np.array([[0, 1], [1, 2]])
        route = np.array([0, 5, 1])  # Arc 5 doesn't exist
        
        cont = compute_route_continuity(route, arcs)
        # Only valid transitions counted
        assert cont == 0.0  # 0->5 invalid, 5->1 invalid


class TestComputeTop1Accuracy:
    def test_perfect_match(self):
        """Test accuracy when all predictions match."""
        candidate_sets = [
            CandidateSet(0, 0, 0, 0, [Candidate(arc_id=1, distance=1, along_track_dist=1, bearing_diff=1, midpoint_dist=1)]),
            CandidateSet(1, 0, 0, 1, [Candidate(arc_id=2, distance=1, along_track_dist=1, bearing_diff=1, midpoint_dist=1)]),
        ]
        gt_route = np.array([1, 2])
        pred_route = np.array([1, 2])
        
        acc = compute_top1_accuracy(pred_route, gt_route, candidate_sets)
        assert acc == 1.0
    
    def test_partial_match(self):
        """Test accuracy with partial matches."""
        candidate_sets = [
            CandidateSet(0, 0, 0, 0, [Candidate(arc_id=1, distance=1, along_track_dist=1, bearing_diff=1, midpoint_dist=1)]),
            CandidateSet(1, 0, 0, 1, [Candidate(arc_id=2, distance=1, along_track_dist=1, bearing_diff=1, midpoint_dist=1)]),
            CandidateSet(2, 0, 0, 2, [Candidate(arc_id=3, distance=1, along_track_dist=1, bearing_diff=1, midpoint_dist=1)]),
        ]
        gt_route = np.array([1, 2, 3])
        pred_route = np.array([1, 3, 3])  # 2nd wrong
        
        acc = compute_top1_accuracy(pred_route, gt_route, candidate_sets)
        assert acc == 2.0 / 3.0
    
    def test_different_lengths(self):
        """Test accuracy with different length arrays."""
        candidate_sets = [
            CandidateSet(0, 0, 0, 0, [Candidate(arc_id=1, distance=1, along_track_dist=1, bearing_diff=1, midpoint_dist=1)]),
        ]
        gt_route = np.array([1, 2, 3])
        pred_route = np.array([1])  # Shorter
        
        acc = compute_top1_accuracy(pred_route, gt_route, candidate_sets)
        assert acc == 1.0  # Only compares min length


class MockCandidateSet:
    """Simple mock for CandidateSet."""
    def __init__(self, candidates):
        self.candidates = candidates


class TestCompareMethods:
    def setup_method(self):
        """Create a test trip."""
        self.nodes = np.array([
            [8.5, 49.8], [8.6, 49.8], [8.6, 49.9], [8.5, 49.9],
        ])
        self.arcs = np.array([
            [0, 1],  # arc 0
            [1, 2],  # arc 1
            [2, 3],  # arc 2
            [3, 0],  # arc 3
        ])
        self.track = np.array([
            [8.52, 49.8, 1.0],
            [8.55, 49.8, 2.0],
            [8.58, 49.8, 3.0],
        ])
        self.route = np.array([0, 0, 0])  # Ground truth: arc 0
        
        self.trip = TripData(
            trip_id="test_trip",
            track=self.track,
            nodes=self.nodes,
            arcs=self.arcs,
            route=self.route,
        )
    
    def test_compare_methods_runs(self):
        """Test that comparison runs without error."""
        model = MockModelInterface()
        config = ComparisonConfig(k=2, radius_meters=100.0)
        
        result = compare_methods(self.trip, model, config)
        
        assert isinstance(result, ComparisonResult)
        assert result.trip_id == "test_trip"
        assert result.n_observations == 3
        assert len(result.classifier_route) == 3
        assert len(result.hmm_route) == 3
        assert result.ground_truth_route is not None
        assert "top1_accuracy" in result.improvement
        assert "route_continuity" in result.improvement
    
    def test_compare_methods_metrics_range(self):
        """Test that metrics are in valid range."""
        model = MockModelInterface()
        config = ComparisonConfig(k=2, radius_meters=100.0)
        
        result = compare_methods(self.trip, model, config)
        
        # All metrics should be between 0 and 1
        assert 0 <= result.classifier_metrics.top1_accuracy <= 1
        assert 0 <= result.hmm_metrics.top1_accuracy <= 1
        assert 0 <= result.classifier_metrics.route_continuity <= 1
        assert 0 <= result.hmm_metrics.route_continuity <= 1
    
    def test_compare_methods_routes_valid(self):
        """Test that predicted routes contain valid arc IDs."""
        model = MockModelInterface()
        config = ComparisonConfig(k=2, radius_meters=100.0)
        
        result = compare_methods(self.trip, model, config)
        
        # All predicted arcs should be from candidate sets (or -1)
        for arc_id in result.classifier_route:
            assert arc_id >= -1
        for arc_id in result.hmm_route:
            assert arc_id >= -1
    
    def test_result_to_dict(self):
        """Test serialization to dict."""
        model = MockModelInterface()
        config = ComparisonConfig(k=2, radius_meters=100.0)
        
        result = compare_methods(self.trip, model, config)
        d = result.to_dict()
        
        assert d["trip_id"] == "test_trip"
        assert "classifier" in d
        assert "hmm" in d
        assert "improvement" in d
        assert "classifier_route" in d
        assert "hmm_route" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])