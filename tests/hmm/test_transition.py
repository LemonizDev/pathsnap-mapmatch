"""Tests for HMM transition scoring."""

import numpy as np
import pytest

from src.features.candidates import Candidate, CandidateSet
from src.hmm.transition import (
    TransitionConfig,
    TransitionScores,
    compute_hop_distances,
    compute_transition_scores,
    compute_transition_matrix_for_viterbi,
)


class TestTransitionConfig:
    def test_default_config(self):
        config = TransitionConfig()
        
        assert config.hop_penalty == 1.0
        assert config.gap_penalty == 10.0
        assert config.max_hops == 3
        assert config.stay_bonus == 0.0
    
    def test_invalid_config(self):
        with pytest.raises(AssertionError):
            TransitionConfig(hop_penalty=-1.0)
        
        with pytest.raises(AssertionError):
            TransitionConfig(gap_penalty=-1.0)
        
        with pytest.raises(AssertionError):
            TransitionConfig(max_hops=0)
        
        with pytest.raises(AssertionError):
            TransitionConfig(hop_penalty=2.0, max_hops=3, gap_penalty=5.0)  # gap < hop*max


class TestComputeHopDistances:
    def test_linear_graph(self):
        """Test BFS on a simple linear graph."""
        # 0 -> 1 -> 2 -> 3
        arcs = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
        ])
        
        hop_dist = compute_hop_distances(arcs, max_hops=2)
        
        # From arc 0: 0 hops to self, 1 hop to arc 1, 2 hops to arc 2
        assert hop_dist[0][0] == 0
        assert hop_dist[0][1] == 1
        assert hop_dist[0][2] == 2
        
        # From arc 1: 0 to self, 1 to arc 2, 1 to arc 0? No, directed!
        # Actually arcs are directed: arc 0 is 0->1, arc 1 is 1->2
        # So from arc 0 (ends at 1), can go to arc 1 (starts at 1)
        # From arc 1 (ends at 2), can go to arc 2 (starts at 2)
        assert hop_dist[1][1] == 0
        assert hop_dist[1][2] == 1
        assert 0 not in hop_dist[1]  # Can't go backwards
    
    def test_loop_graph(self):
        """Test BFS on a loop."""
        # 0 -> 1 -> 2 -> 0
        arcs = np.array([
            [0, 1],
            [1, 2],
            [2, 0],
        ])
        
        hop_dist = compute_hop_distances(arcs, max_hops=3)
        
        # From arc 0: 0->1->2->0
        assert hop_dist[0][0] == 0
        assert hop_dist[0][1] == 1
        assert hop_dist[0][2] == 2
        # With max_hops=3, should reach back to 0 in 3 hops
        # But we stop at max_hops, so 0 not reached again
    
    def test_disconnected_graph(self):
        """Test with disconnected components."""
        # 0 -> 1, and 2 -> 3 (two separate components)
        arcs = np.array([
            [0, 1],
            [2, 3],
        ])
        
        hop_dist = compute_hop_distances(arcs, max_hops=2)
        
        # Arc 0 can only reach itself and arc 1? No, arc 0 ends at 1, 
        # but no arc starts at 1. So only self.
        assert hop_dist[0] == {0: 0}
        assert hop_dist[1] == {1: 0}


class TestComputeTransitionScores:
    def setup_method(self):
        """Create test candidate sets and arcs."""
        self.arcs = np.array([
            [0, 1],  # arc 0
            [1, 2],  # arc 1
            [2, 3],  # arc 2
            [3, 0],  # arc 3 (loops back)
        ])
        
        self.candidates_1 = [
            Candidate(arc_id=0, distance=5.0, along_track_dist=10.0, bearing_diff=10.0, midpoint_dist=100.0),
            Candidate(arc_id=1, distance=15.0, along_track_dist=20.0, bearing_diff=30.0, midpoint_dist=200.0),
        ]
        self.candidates_2 = [
            Candidate(arc_id=1, distance=8.0, along_track_dist=15.0, bearing_diff=5.0, midpoint_dist=150.0),
            Candidate(arc_id=2, distance=12.0, along_track_dist=25.0, bearing_diff=20.0, midpoint_dist=180.0),
            Candidate(arc_id=3, distance=30.0, along_track_dist=10.0, bearing_diff=60.0, midpoint_dist=300.0),
        ]
        
        self.candidate_sets = [
            CandidateSet(0, 8.5, 49.8, 1.0, self.candidates_1),
            CandidateSet(1, 8.51, 49.81, 2.0, self.candidates_2),
        ]
    
    def test_transition_scores_shape(self):
        config = TransitionConfig(hop_penalty=1.0, gap_penalty=10.0, max_hops=2)
        trans = compute_transition_scores(self.candidate_sets, self.arcs, config)
        
        assert len(trans) == 1  # T-1 transitions
        assert trans.log_transitions[0].shape == (2, 3)  # K_0=2, K_1=3
        assert trans.valid_mask[0].shape == (2, 3)
    
    def test_same_arc_transition(self):
        """Test staying on same arc gives stay_bonus (0 by default)."""
        config = TransitionConfig(hop_penalty=1.0, gap_penalty=10.0, max_hops=2, stay_bonus=0.0)
        trans = compute_transition_scores(self.candidate_sets, self.arcs, config)
        
        # Arc 1 appears in both timesteps
        # At t=0, arc 1 is at index 1
        # At t=1, arc 1 is at index 0
        # So transition[1, 0] should be stay_bonus = 0
        assert trans.log_transitions[0][1, 0] == 0.0
    
    def test_connected_transition(self):
        """Test connected arcs get hop penalty."""
        config = TransitionConfig(hop_penalty=2.0, gap_penalty=10.0, max_hops=2)
        trans = compute_transition_scores(self.candidate_sets, self.arcs, config)
        
        # Arc 0 (0->1) connects to arc 1 (1->2): 1 hop
        # Arc 0 at t=0 index 0, arc 1 at t=1 index 0
        assert trans.log_transitions[0][0, 0] == -2.0  # -hop_penalty * 1
        
        # Arc 0 connects to arc 2 in 2 hops (0->1->2)
        # Arc 0 at t=0 index 0, arc 2 at t=1 index 1
        assert trans.log_transitions[0][0, 1] == -4.0  # -hop_penalty * 2
    
    def test_disconnected_transition(self):
        """Test disconnected arcs get gap penalty."""
        config = TransitionConfig(hop_penalty=1.0, gap_penalty=10.0, max_hops=1)  # max_hops=1
        trans = compute_transition_scores(self.candidate_sets, self.arcs, config)
        
        # Arc 1 (1->2) does NOT connect to arc 3 (3->0) within 1 hop
        # Arc 1 at t=0 index 1, arc 3 at t=1 index 2
        assert trans.log_transitions[0][1, 2] == -10.0  # -gap_penalty
    
    def test_valid_mask(self):
        config = TransitionConfig()
        trans = compute_transition_scores(self.candidate_sets, self.arcs, config)
        
        # All pairs should be valid
        assert np.all(trans.valid_mask[0])
    
    def test_single_timestep(self):
        """Test with only one timestep."""
        config = TransitionConfig()
        trans = compute_transition_scores([self.candidate_sets[0]], self.arcs, config)
        
        assert len(trans) == 0
        assert trans.log_transitions == []
    
    def test_compute_transition_matrix_for_viterbi(self):
        """Test padded transition matrix for Viterbi."""
        config = TransitionConfig(hop_penalty=1.0, gap_penalty=10.0, max_hops=2)
        
        trans_padded, mask_padded = compute_transition_matrix_for_viterbi(
            self.candidate_sets, self.arcs, config
        )
        
        # T-1=1, max_K=3
        assert trans_padded.shape == (1, 3, 3)
        assert mask_padded.shape == (1, 3, 3)
        
        # Valid entries should match
        assert mask_padded[0, 0, 0] == True
        assert mask_padded[0, 1, 2] == True
        
        # Invalid (padded) should be False
        assert mask_padded[0, 2, 0] == False  # K_curr=2, K_next=3


class TestGetTransitionMatrix:
    def setup_method(self):
        """Create test candidate sets and arcs."""
        self.arcs = np.array([
            [0, 1],  # arc 0
            [1, 2],  # arc 1
            [2, 3],  # arc 2
            [3, 0],  # arc 3 (loops back)
        ])
        
        self.candidates_1 = [
            Candidate(arc_id=0, distance=5.0, along_track_dist=10.0, bearing_diff=10.0, midpoint_dist=100.0),
            Candidate(arc_id=1, distance=15.0, along_track_dist=20.0, bearing_diff=30.0, midpoint_dist=200.0),
        ]
        self.candidates_2 = [
            Candidate(arc_id=1, distance=8.0, along_track_dist=15.0, bearing_diff=5.0, midpoint_dist=150.0),
            Candidate(arc_id=2, distance=12.0, along_track_dist=25.0, bearing_diff=20.0, midpoint_dist=180.0),
            Candidate(arc_id=3, distance=30.0, along_track_dist=10.0, bearing_diff=60.0, midpoint_dist=300.0),
        ]
        
        self.candidate_sets = [
            CandidateSet(0, 8.5, 49.8, 1.0, self.candidates_1),
            CandidateSet(1, 8.51, 49.81, 2.0, self.candidates_2),
        ]
    
    def test_get_transition_matrix(self):
        config = TransitionConfig()
        trans = compute_transition_scores(self.candidate_sets, self.arcs, config)
        
        t_matrix, t_mask = trans.log_transitions[0], trans.valid_mask[0]
        
        assert t_matrix.shape == (2, 3)
        assert t_mask.shape == (2, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])