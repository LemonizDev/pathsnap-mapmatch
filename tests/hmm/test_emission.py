"""Tests for HMM emission scoring."""

import numpy as np
import pytest

from src.features.candidates import Candidate, CandidateSet
from src.hmm.emission import (
    EmissionConfig,
    EmissionScores,
    compute_emission_scores,
    compute_emission_scores_from_model,
    get_emission_for_timestep,
)
from src.hmm.model_interface import MockModelInterface


class TestEmissionConfig:
    def test_default_config(self):
        config = EmissionConfig()
        
        assert config.distance_penalty_lambda == 0.05
        assert config.min_log_prob == -100.0
        assert config.max_log_prob == 0.0
    
    def test_custom_config(self):
        config = EmissionConfig(
            distance_penalty_lambda=0.1,
            min_log_prob=-50.0,
            max_log_prob=-1.0,
        )
        
        assert config.distance_penalty_lambda == 0.1
        assert config.min_log_prob == -50.0
        assert config.max_log_prob == -1.0
    
    def test_invalid_lambda(self):
        with pytest.raises(AssertionError):
            EmissionConfig(distance_penalty_lambda=-1.0)


class TestComputeEmissionScores:
    def setup_method(self):
        """Create test candidate sets and model probabilities."""
        self.candidates_1 = [
            Candidate(arc_id=1, distance=5.0, along_track_dist=10.0, bearing_diff=10.0, midpoint_dist=100.0),
            Candidate(arc_id=2, distance=15.0, along_track_dist=20.0, bearing_diff=30.0, midpoint_dist=200.0),
            Candidate(arc_id=3, distance=50.0, along_track_dist=5.0, bearing_diff=90.0, midpoint_dist=500.0),
        ]
        self.candidates_2 = [
            Candidate(arc_id=2, distance=8.0, along_track_dist=15.0, bearing_diff=5.0, midpoint_dist=150.0),
            Candidate(arc_id=3, distance=12.0, along_track_dist=25.0, bearing_diff=20.0, midpoint_dist=180.0),
            Candidate(arc_id=4, distance=30.0, along_track_dist=10.0, bearing_diff=60.0, midpoint_dist=300.0),
        ]
        
        self.candidate_sets = [
            CandidateSet(0, 8.5, 49.8, 1.0, self.candidates_1),
            CandidateSet(1, 8.51, 49.81, 2.0, self.candidates_2),
        ]
        
        # Model probabilities
        self.model_probs = MockModelInterface().get_probabilities(
            np.array([
                [5.0, 10.0, 10.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 1.0, 0.0],
                [15.0, 20.0, 30.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 2.0, 0.0],
                [50.0, 5.0, 90.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 3.0, 0.0],
                [8.0, 15.0, 5.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 1.0, 0.0],
                [12.0, 25.0, 20.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 2.0, 0.0],
                [30.0, 10.0, 60.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 3.0, 0.0],
            ]),
            np.array([1, 2, 3, 2, 3, 4]),
        )
    
    def test_compute_emission_scores_shape(self):
        config = EmissionConfig(distance_penalty_lambda=0.05)
        emissions = compute_emission_scores(self.candidate_sets, self.model_probs, config)
        
        assert emissions.T == 2
        assert emissions.max_K == 3
        assert emissions.log_emissions.shape == (2, 3)
        assert emissions.candidate_arc_ids.shape == (2, 3)
        assert emissions.valid_mask.shape == (2, 3)
    
    def test_emission_formula(self):
        """Test emission = model_log_prob - lambda * distance."""
        config = EmissionConfig(distance_penalty_lambda=0.05)
        emissions = compute_emission_scores(self.candidate_sets, self.model_probs, config)
        
        # For candidate 0 at t=0: arc_id=1, distance=5.0
        # model_log_prob for arc 1 - 0.05 * 5.0
        arc1_logprob = self.model_probs.log_probs[0]  # arc_id=1 is first
        expected = arc1_logprob - 0.05 * 5.0
        
        actual = emissions.log_emissions[0, 0]
        assert np.isclose(actual, expected), f"Expected {expected}, got {actual}"
    
    def test_emission_clamping(self):
        """Test that emissions are clamped to min/max."""
        config = EmissionConfig(
            distance_penalty_lambda=100.0,  # Huge penalty
            min_log_prob=-50.0,
            max_log_prob=-1.0,
        )
        emissions = compute_emission_scores(self.candidate_sets, self.model_probs, config)
        
        # All should be clamped
        assert np.all(emissions.log_emissions[emissions.valid_mask] >= -50.0)
        assert np.all(emissions.log_emissions[emissions.valid_mask] <= -1.0)
    
    def test_valid_mask(self):
        config = EmissionConfig()
        emissions = compute_emission_scores(self.candidate_sets, self.model_probs, config)
        
        # All candidates should be valid
        assert np.all(emissions.valid_mask[0, :3])  # First timestep has 3 candidates
        assert np.all(emissions.valid_mask[1, :3])  # Second timestep has 3 candidates
        
        # Check arc IDs
        assert emissions.candidate_arc_ids[0, 0] == 1
        assert emissions.candidate_arc_ids[0, 1] == 2
        assert emissions.candidate_arc_ids[0, 2] == 3
        assert emissions.candidate_arc_ids[1, 0] == 2
        assert emissions.candidate_arc_ids[1, 1] == 3
        assert emissions.candidate_arc_ids[1, 2] == 4
    
    def test_missing_arc_in_model(self):
        """Test handling when arc not in model output."""
        # Model probs only for arcs 1, 2
        model_probs = MockModelInterface().get_probabilities(
            np.array([
                [5.0, 10.0, 10.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 1.0, 0.0],
                [15.0, 20.0, 30.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 2.0, 0.0],
            ]),
            np.array([1, 2]),
        )
        
        # Candidate set has arc 3 which is not in model
        candidates = [Candidate(arc_id=3, distance=10.0, along_track_dist=10.0, bearing_diff=10.0, midpoint_dist=100.0)]
        candidate_sets = [CandidateSet(0, 8.5, 49.8, 1.0, candidates)]
        
        config = EmissionConfig(min_log_prob=-100.0)
        emissions = compute_emission_scores(candidate_sets, model_probs, config)
        
        # Should use min_log_prob for missing arc
        assert emissions.log_emissions[0, 0] == -100.0
    
    def test_get_emission_for_timestep(self):
        config = EmissionConfig()
        emissions = compute_emission_scores(self.candidate_sets, self.model_probs, config)
        
        log_em, arc_ids, valid_idx = get_emission_for_timestep(emissions, 0)
        
        assert len(log_em) == 3
        assert len(arc_ids) == 3
        assert np.array_equal(valid_idx, [0, 1, 2])
    
    def test_empty_candidate_sets(self):
        config = EmissionConfig()
        emissions = compute_emission_scores([], MockModelInterface().get_probabilities(
            np.zeros((0, 11)), np.zeros(0, dtype=np.int64)
        ), config)
        
        assert emissions.T == 0
        assert emissions.max_K == 0


class TestComputeEmissionScoresFromModel:
    def test_end_to_end(self):
        """Test full pipeline from model to emissions."""
        candidate_sets = [
            CandidateSet(0, 8.5, 49.8, 1.0, [
                Candidate(arc_id=1, distance=5.0, along_track_dist=10.0, bearing_diff=10.0, midpoint_dist=100.0),
                Candidate(arc_id=2, distance=15.0, along_track_dist=20.0, bearing_diff=30.0, midpoint_dist=200.0),
            ]),
            CandidateSet(1, 8.51, 49.81, 2.0, [
                Candidate(arc_id=2, distance=8.0, along_track_dist=15.0, bearing_diff=5.0, midpoint_dist=150.0),
                Candidate(arc_id=3, distance=12.0, along_track_dist=25.0, bearing_diff=20.0, midpoint_dist=180.0),
            ]),
        ]
        
        model = MockModelInterface()
        X = np.array([
            [5.0, 10.0, 10.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 1.0, 0.0],
            [15.0, 20.0, 30.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 2.0, 0.0],
            [8.0, 15.0, 5.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 1.0, 0.0],
            [12.0, 25.0, 20.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 2.0, 0.0],
        ])
        
        config = EmissionConfig(distance_penalty_lambda=0.05)
        emissions = compute_emission_scores_from_model(candidate_sets, model, X, config)
        
        assert emissions.T == 2
        assert emissions.max_K == 2
        assert np.all(emissions.valid_mask)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])