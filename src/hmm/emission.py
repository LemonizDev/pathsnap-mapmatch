"""
Emission scoring for PathSnap HMM.

Computes log P(observation_t | candidate_arc) using:
    log P(obs_t | arc) = model_log_prob(arc) - lambda * distance(gps_t, arc)

Where:
- model_log_prob: Calibrated log-probability from classifier
- distance: Perpendicular distance from GPS to arc segment (meters)
- lambda: Distance penalty weight (tunable hyperparameter)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.features.candidates import Candidate, CandidateSet
from src.hmm.model_interface import ModelInterface, ModelProbabilities


@dataclass
class EmissionConfig:
    """Configuration for emission scoring."""
    distance_penalty_lambda: float = 0.05  # Weight for distance penalty (per meter)
    min_log_prob: float = -100.0  # Clamp minimum log-probability
    max_log_prob: float = 0.0     # Clamp maximum log-probability (log(1) = 0)
    
    def __post_init__(self):
        assert self.distance_penalty_lambda >= 0, "distance_penalty_lambda must be >= 0"
        assert self.min_log_prob <= self.max_log_prob, "min_log_prob must be <= max_log_prob"


@dataclass
class EmissionScores:
    """
    Emission log-probabilities for all candidates at all timesteps.
    
    Attributes:
        log_emissions: Shape (T, max_K) - log P(obs_t | candidate_k)
                       Padded with min_log_prob for missing candidates
        candidate_arc_ids: Shape (T, max_K) - Arc ID for each candidate
        valid_mask: Shape (T, max_K) - True where candidate exists
        candidate_sets: Original CandidateSet objects for reference
    """
    log_emissions: np.ndarray
    candidate_arc_ids: np.ndarray
    valid_mask: np.ndarray
    candidate_sets: List[CandidateSet]
    
    @property
    def T(self) -> int:
        return self.log_emissions.shape[0]
    
    @property
    def max_K(self) -> int:
        return self.log_emissions.shape[1]


def compute_emission_scores(
    candidate_sets: List[CandidateSet],
    model_probs: ModelProbabilities,
    config: EmissionConfig,
) -> EmissionScores:
    """
    Compute emission log-probabilities for all candidates across all timesteps.
    
    Formula: log P(obs_t | arc) = model_log_prob(arc) - lambda * perp_distance
    
    Args:
        candidate_sets: List of CandidateSet, one per observation
        model_probs: ModelProbabilities from ModelInterface.get_probabilities()
        config: EmissionConfig with lambda and clamping bounds
        
    Returns:
        EmissionScores with padded arrays for variable-size candidate sets
    """
    T = len(candidate_sets)
    max_K = max(len(cs.candidates) for cs in candidate_sets) if candidate_sets else 0
    
    if max_K == 0:
        return EmissionScores(
            log_emissions=np.zeros((0, 0), dtype=np.float64),
            candidate_arc_ids=np.zeros((0, 0), dtype=np.int64),
            valid_mask=np.zeros((0, 0), dtype=bool),
            candidate_sets=candidate_sets,
        )
    
    # Initialize with min_log_prob (effectively -inf for Viterbi)
    log_emissions = np.full((T, max_K), config.min_log_prob, dtype=np.float64)
    candidate_arc_ids = np.full((T, max_K), -1, dtype=np.int64)
    valid_mask = np.zeros((T, max_K), dtype=bool)
    
    # Create lookup from arc_id to model log-prob
    arc_to_logprob = {}
    for arc_id, logp in zip(model_probs.candidate_arc_ids, model_probs.log_probs):
        arc_to_logprob[arc_id] = logp
    
    # Fill in emissions for each timestep
    for t, cs in enumerate(candidate_sets):
        for k, cand in enumerate(cs.candidates):
            if cand.arc_id in arc_to_logprob:
                model_logp = arc_to_logprob[cand.arc_id]
            else:
                # Arc not in model output - use minimum probability
                model_logp = config.min_log_prob
            
            # Distance penalty
            distance_penalty = config.distance_penalty_lambda * cand.distance
            
            # Combined emission
            emission = model_logp - distance_penalty
            
            # Clamp
            emission = np.clip(emission, config.min_log_prob, config.max_log_prob)
            
            log_emissions[t, k] = emission
            candidate_arc_ids[t, k] = cand.arc_id
            valid_mask[t, k] = True
    
    return EmissionScores(
        log_emissions=log_emissions,
        candidate_arc_ids=candidate_arc_ids,
        valid_mask=valid_mask,
        candidate_sets=candidate_sets,
    )


def compute_emission_scores_from_model(
    candidate_sets: List[CandidateSet],
    model: ModelInterface,
    X: np.ndarray,
    config: EmissionConfig,
    batch_size: int = 50000,
) -> EmissionScores:
    """
    Convenience function: compute emissions directly from model and feature matrix.
    
    Args:
        candidate_sets: List of CandidateSet per observation
        model: ModelInterface implementation
        X: Feature matrix (n_candidates, n_features) - must match candidate ordering
        config: EmissionConfig
        batch_size: Batch size for model inference
        
    Returns:
        EmissionScores
    """
    # Get candidate arc IDs in same order as X
    candidate_arc_ids = []
    for cs in candidate_sets:
        for cand in cs.candidates:
            candidate_arc_ids.append(cand.arc_id)
    candidate_arc_ids = np.array(candidate_arc_ids, dtype=np.int64)
    
    # Get model probabilities
    model_probs = model.get_probabilities(X, candidate_arc_ids, batch_size)
    
    return compute_emission_scores(candidate_sets, model_probs, config)


def get_emission_for_timestep(
    emissions: EmissionScores,
    t: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract valid emissions for a single timestep.
    
    Returns:
        (log_emissions_valid, arc_ids_valid, valid_indices)
    """
    mask = emissions.valid_mask[t]
    return (
        emissions.log_emissions[t, mask],
        emissions.candidate_arc_ids[t, mask],
        np.where(mask)[0],
    )


if __name__ == "__main__":
    # Quick test
    from src.hmm.model_interface import MockModelInterface
    
    # Create mock candidate sets
    candidates_1 = [
        Candidate(arc_id=1, distance=5.0, along_track_dist=10.0, bearing_diff=10.0, midpoint_dist=100.0),
        Candidate(arc_id=2, distance=15.0, along_track_dist=20.0, bearing_diff=30.0, midpoint_dist=200.0),
        Candidate(arc_id=3, distance=50.0, along_track_dist=5.0, bearing_diff=90.0, midpoint_dist=500.0),
    ]
    candidates_2 = [
        Candidate(arc_id=2, distance=8.0, along_track_dist=15.0, bearing_diff=5.0, midpoint_dist=150.0),
        Candidate(arc_id=3, distance=12.0, along_track_dist=25.0, bearing_diff=20.0, midpoint_dist=180.0),
        Candidate(arc_id=4, distance=30.0, along_track_dist=10.0, bearing_diff=60.0, midpoint_dist=300.0),
    ]
    
    candidate_sets = [
        CandidateSet(0, 8.5, 49.8, 1.0, candidates_1),
        CandidateSet(1, 8.51, 49.81, 2.0, candidates_2),
    ]
    
    # Create mock model and feature matrix
    model = MockModelInterface()
    
    # Build X matrix (2 timesteps * 3 candidates = 6 rows)
    # Using simple features for test
    X = np.array([
        [5.0, 10.0, 10.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 1.0, 0.0],
        [15.0, 20.0, 30.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 2.0, 0.0],
        [50.0, 5.0, 90.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 3.0, 0.0],
        [8.0, 15.0, 5.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 1.0, 0.0],
        [12.0, 25.0, 20.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 2.0, 0.0],
        [30.0, 10.0, 60.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 3.0, 0.0],
    ])
    
    config = EmissionConfig(distance_penalty_lambda=0.05)
    emissions = compute_emission_scores_from_model(candidate_sets, model, X, config)
    
    print(f"T={emissions.T}, max_K={emissions.max_K}")
    print(f"log_emissions shape: {emissions.log_emissions.shape}")
    print(f"valid_mask:\n{emissions.valid_mask}")
    print(f"log_emissions:\n{emissions.log_emissions}")
    print("Emission test passed!")