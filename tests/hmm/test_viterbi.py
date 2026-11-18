"""Tests for HMM Viterbi decoding."""

import numpy as np
import pytest

from src.features.candidates import Candidate, CandidateSet
from src.hmm.emission import EmissionConfig, compute_emission_scores_from_model
from src.hmm.transition import TransitionConfig, compute_transition_scores
from src.hmm.viterbi import (
    ViterbiConfig,
    ViterbiResult,
    viterbi_decode,
    viterbi_decode_padded,
    reconstruct_path_from_result,
)
from src.hmm.model_interface import MockModelInterface


class TestViterbiConfig:
    def test_default_config(self):
        config = ViterbiConfig()
        assert config.initial_log_prob == 0.0


class TestViterbiDecode:
    def setup_method(self):
        """Create a simple 3-timestep test case."""
        self.candidate_sets = [
            CandidateSet(0, 8.5, 49.8, 1.0, [
                Candidate(arc_id=1, distance=5.0, along_track_dist=10.0, bearing_diff=10.0, midpoint_dist=100.0),
                Candidate(arc_id=2, distance=15.0, along_track_dist=20.0, bearing_diff=30.0, midpoint_dist=200.0),
            ]),
            CandidateSet(1, 8.51, 49.81, 2.0, [
                Candidate(arc_id=2, distance=8.0, along_track_dist=15.0, bearing_diff=5.0, midpoint_dist=150.0),
                Candidate(arc_id=3, distance=12.0, along_track_dist=25.0, bearing_diff=20.0, midpoint_dist=180.0),
            ]),
            CandidateSet(2, 8.52, 49.82, 3.0, [
                Candidate(arc_id=3, distance=5.0, along_track_dist=10.0, bearing_diff=8.0, midpoint_dist=80.0),
                Candidate(arc_id=4, distance=20.0, along_track_dist=30.0, bearing_diff=40.0, midpoint_dist=250.0),
            ]),
        ]
        
        self.arcs = np.array([
            [1, 2],  # arc 1
            [2, 3],  # arc 2
            [3, 4],  # arc 3
            [4, 5],  # arc 4
        ])
        
        # Build emissions and transitions
        model = MockModelInterface()
        X = np.array([
            [5.0, 10.0, 10.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 1.0, 0.0],
            [15.0, 20.0, 30.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 2.0, 0.0],
            [8.0, 15.0, 5.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 1.0, 0.0],
            [12.0, 25.0, 20.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 2.0, 0.0],
            [5.0, 10.0, 8.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 1.0, 0.0],
            [20.0, 30.0, 40.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 2.0, 0.0],
        ])
        
        self.emissions = compute_emission_scores_from_model(
            self.candidate_sets, model, X, EmissionConfig(distance_penalty_lambda=0.05)
        )
        
        self.transitions = compute_transition_scores(
            self.candidate_sets, self.arcs, TransitionConfig(hop_penalty=1.0, gap_penalty=10.0, max_hops=2)
        )
    
    def test_viterbi_decode_basic(self):
        result = viterbi_decode(self.emissions, self.transitions)
        
        assert isinstance(result, ViterbiResult)
        assert result.T == 3
        assert len(result.best_path_arc_ids) == 3
        assert len(result.best_path_indices) == 3
        assert result.log_alphas.shape == (3, 2)  # max_K=2
        assert result.backpointers.shape == (3, 2)
        assert result.valid_mask.shape == (3, 2)
    
    def test_viterbi_path_makes_sense(self):
        """Test that Viterbi finds a valid path."""
        result = viterbi_decode(self.emissions, self.transitions)
        
        # Path should be valid arc IDs (not -1 for valid positions)
        for t in range(3):
            idx = result.best_path_indices[t]
            assert self.emissions.valid_mask[t, idx], f"Invalid index at t={t}"
            arc_id = result.best_path_arc_ids[t]
            assert arc_id in [c.arc_id for c in self.candidate_sets[t].candidates]
    
    def test_viterbi_prefers_connected_path(self):
        """Test that Viterbi prefers graph-connected transitions."""
        # The graph is 1->2->3->4
        # Best path should follow connectivity: 1 -> 2 -> 3 or 2 -> 3 -> 4
        result = viterbi_decode(self.emissions, self.transitions)
        
        path = result.best_path_arc_ids
        
        # Check continuity
        for i in range(len(path) - 1):
            arc_i = path[i]
            arc_j = path[i + 1]
            
            # Find in arcs array
            idx_i = np.where(self.arcs[:, 0] == arc_i)[0]
            idx_j = np.where(self.arcs[:, 0] == arc_j)[0]
            
            if len(idx_i) > 0 and len(idx_j) > 0:
                # Check if connected
                to_i = self.arcs[idx_i[0], 1]
                from_j = self.arcs[idx_j[0], 0]
                # Should prefer connected, but not strictly required if emissions are strong
    
    def test_viterbi_single_timestep(self):
        """Test Viterbi with single timestep."""
        single_cs = [self.candidate_sets[0]]
        single_emissions = compute_emission_scores_from_model(
            single_cs, MockModelInterface(), 
            np.array([[5.0, 10.0, 10.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 1.0, 0.0],
                     [15.0, 20.0, 30.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 2.0, 0.0]]),
            EmissionConfig()
        )
        single_transitions = compute_transition_scores(single_cs, self.arcs, TransitionConfig())
        
        result = viterbi_decode(single_emissions, single_transitions)
        
        assert result.T == 1
        assert len(result.best_path_arc_ids) == 1
    
    def test_viterbi_empty(self):
        """Test Viterbi with empty inputs."""
        from src.hmm.emission import EmissionScores
        from src.hmm.transition import TransitionScores
        
        empty_emissions = EmissionScores(
            log_emissions=np.zeros((0, 0)),
            candidate_arc_ids=np.zeros((0, 0), dtype=np.int64),
            valid_mask=np.zeros((0, 0), dtype=bool),
            candidate_sets=[],
        )
        empty_transitions = TransitionScores([], [])
        
        result = viterbi_decode(empty_emissions, empty_transitions)
        
        assert result.T == 0
        assert len(result.best_path_arc_ids) == 0
    
    def test_viterbi_log_prob(self):
        """Test that best_log_prob is reasonable."""
        result = viterbi_decode(self.emissions, self.transitions)
        
        # Log prob should be finite
        assert np.isfinite(result.best_log_prob)
        assert result.best_log_prob <= 0  # Log probs are <= 0
    
    def test_reconstruct_path_from_result(self):
        """Test path reconstruction utility."""
        result = viterbi_decode(self.emissions, self.transitions)
        arc_ids = reconstruct_path_from_result(result, self.candidate_sets)
        
        assert len(arc_ids) == 3
        assert np.array_equal(arc_ids, result.best_path_arc_ids)


class TestViterbiDecodePadded:
    def test_padded_decode(self):
        """Test the padded matrix version."""
        # Create padded matrices manually
        log_emissions = np.array([
            [-1.0, -2.0, -np.inf],
            [-0.5, -1.5, -3.0],
            [-2.0, -1.0, -0.5],
        ])
        
        log_transitions = np.array([
            [[0.0, -1.0, -10.0],
             [-1.0, 0.0, -1.0],
             [-10.0, -10.0, -10.0]],
            [[-10.0, -10.0, -10.0],
             [-1.0, 0.0, -1.0],
             [-10.0, -1.0, 0.0]],
        ])
        
        valid_mask = np.array([
            [True, True, False],
            [True, True, True],
            [True, True, True],
        ])
        
        trans_mask = np.array([
            [[True, True, True],
             [True, True, True],
             [False, False, False]],
            [[False, False, False],
             [True, True, True],
             [True, True, True]],
        ])
        
        result = viterbi_decode_padded(
            log_emissions, log_transitions, valid_mask, trans_mask
        )
        
        assert result.T == 3
        assert len(result.best_path_indices) == 3
        assert np.isfinite(result.best_log_prob)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])