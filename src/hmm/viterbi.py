"""
Numerically stable Viterbi decoding for PathSnap HMM.

Implements the forward-backward Viterbi algorithm in log-space with:
- Variable candidate set sizes per timestep (K_t)
- Padding with -inf for invalid transitions
- Backpointer tracking for path reconstruction
- Log-sum-exp for numerical stability (not needed for max-product Viterbi)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.features.candidates import CandidateSet
from src.hmm.emission import EmissionScores
from src.hmm.transition import TransitionScores, compute_transition_matrix_for_viterbi


@dataclass
class ViterbiConfig:
    """Configuration for Viterbi decoding."""
    initial_log_prob: float = 0.0  # Log prior for first timestep (uniform = 0)
    
    def __post_init__(self):
        pass


@dataclass
class ViterbiResult:
    """
    Result of Viterbi decoding.
    
    Attributes:
        best_path_arc_ids: Arc IDs of the decoded route (length T)
        best_path_indices: Candidate indices at each timestep (length T)
        best_log_prob: Log-probability of the best path
        log_alphas: Forward log-probabilities (T, max_K) - for debugging
        backpointers: Backpointer indices (T, max_K) - for path reconstruction
        valid_mask: Valid candidate mask (T, max_K)
    """
    best_path_arc_ids: np.ndarray
    best_path_indices: np.ndarray
    best_log_prob: float
    log_alphas: np.ndarray
    backpointers: np.ndarray
    valid_mask: np.ndarray
    
    @property
    def T(self) -> int:
        return len(self.best_path_arc_ids)


def viterbi_decode(
    emissions: EmissionScores,
    transitions: TransitionScores,
    config: Optional[ViterbiConfig] = None,
) -> ViterbiResult:
    """
    Run Viterbi decoding to find the most likely sequence of road arcs.
    
    Standard DP in log-space:
    - alpha[t, k] = log_emission[t, k] + max_j (alpha[t-1, j] + log_transition[t-1, j, k])
    - backpointer[t, k] = argmax_j (alpha[t-1, j] + log_transition[t-1, j, k])
    
    Handles variable K_t via padding with -inf.
    
    Args:
        emissions: EmissionScores from compute_emission_scores()
        transitions: TransitionScores from compute_transition_scores()
        config: ViterbiConfig
        
    Returns:
        ViterbiResult with best path and decoding info
    """
    if config is None:
        config = ViterbiConfig()
    
    T = emissions.T
    max_K = emissions.max_K
    
    if T == 0:
        return ViterbiResult(
            best_path_arc_ids=np.array([], dtype=np.int64),
            best_path_indices=np.array([], dtype=np.int64),
            best_log_prob=-np.inf,
            log_alphas=np.zeros((0, 0)),
            backpointers=np.zeros((0, 0), dtype=np.int64),
            valid_mask=np.zeros((0, 0), dtype=bool),
        )
    
    if T == 1:
        # Single timestep: just pick best emission
        valid = emissions.valid_mask[0]
        log_em = emissions.log_emissions[0]
        arc_ids = emissions.candidate_arc_ids[0]
        
        # Mask invalid
        masked_em = np.where(valid, log_em, -np.inf)
        best_idx = np.argmax(masked_em)
        
        return ViterbiResult(
            best_path_arc_ids=np.array([arc_ids[best_idx]], dtype=np.int64),
            best_path_indices=np.array([best_idx], dtype=np.int64),
            best_log_prob=float(masked_em[best_idx]),
            log_alphas=masked_em[None, :],
            backpointers=np.zeros((1, max_K), dtype=np.int64),
            valid_mask=valid[None, :],
        )
    
    # Initialize alpha (T, max_K) with -inf
    log_alphas = np.full((T, max_K), -np.inf, dtype=np.float64)
    backpointers = np.zeros((T, max_K), dtype=np.int64)
    
    # Use TransitionScores directly for variable K_t
    
    # T = 0: initial log-probs = emission + initial_prior
    log_alphas[0] = np.where(
        emissions.valid_mask[0],
        emissions.log_emissions[0] + config.initial_log_prob,
        -np.inf
    )
    backpointers[0] = -1  # No predecessor
    
    # Forward pass
    for t in range(1, T):
        # Get valid transitions from t-1 to t
        if t - 1 < len(transitions.log_transitions):
            trans_t = transitions.log_transitions[t - 1]
            trans_mask_t = transitions.valid_mask[t - 1]
            K_prev = trans_t.shape[0]
            K_curr = trans_t.shape[1]
        else:
            # Should not happen
            K_prev = max_K
            K_curr = max_K
            trans_t = np.full((K_prev, K_curr), -np.inf)
            trans_mask_t = np.zeros((K_prev, K_curr), dtype=bool)
        
        # For each current candidate k
        for k in range(K_curr):
            if not emissions.valid_mask[t, k]:
                continue
            
            # Find best predecessor j
            # alpha[t-1, j] + trans[t-1, j, k]
            # Only consider valid predecessors
            valid_prev_mask = emissions.valid_mask[t - 1, :K_prev]
            valid_prev_indices = np.where(valid_prev_mask)[0]
            
            if len(valid_prev_indices) == 0:
                log_alphas[t, k] = -np.inf
                backpointers[t, k] = -1
                continue
            
            # Compute scores for all valid predecessors
            scores = np.full(K_prev, -np.inf)
            scores[valid_prev_indices] = log_alphas[t - 1, valid_prev_indices]
            
            # Add transition scores
            trans_scores = trans_t[:, k]
            # Only valid transitions
            trans_scores = np.where(trans_mask_t[:, k], trans_scores, -np.inf)
            
            combined = scores + trans_scores
            
            if np.all(np.isneginf(combined)):
                log_alphas[t, k] = -np.inf
                backpointers[t, k] = -1
            else:
                best_j = np.argmax(combined)
                log_alphas[t, k] = emissions.log_emissions[t, k] + combined[best_j]
                backpointers[t, k] = best_j
    
    # Backward pass: find best final state
    valid_last = emissions.valid_mask[T - 1]
    if not np.any(valid_last):
        # No valid candidates at last timestep
        best_last_idx = 0
        best_log_prob = -np.inf
    else:
        last_scores = np.where(valid_last, log_alphas[T - 1], -np.inf)
        best_last_idx = np.argmax(last_scores)
        best_log_prob = float(last_scores[best_last_idx])
    
    # Reconstruct path
    best_path_indices = np.zeros(T, dtype=np.int64)
    best_path_arc_ids = np.zeros(T, dtype=np.int64)
    
    best_path_indices[T - 1] = best_last_idx
    best_path_arc_ids[T - 1] = emissions.candidate_arc_ids[T - 1, best_last_idx]
    
    for t in range(T - 2, -1, -1):
        next_idx = best_path_indices[t + 1]
        # Check if we have a valid backpointer
        if t + 1 < len(transitions.log_transitions):
            K_next = transitions.log_transitions[t].shape[1]
            if next_idx < K_next and backpointers[t + 1, next_idx] >= 0:
                best_path_indices[t] = backpointers[t + 1, next_idx]
            else:
                # Fallback: pick best valid at this timestep
                valid_t = emissions.valid_mask[t]
                if np.any(valid_t):
                    best_path_indices[t] = np.argmax(
                        np.where(valid_t, log_alphas[t], -np.inf)
                    )
                else:
                    best_path_indices[t] = 0
        else:
            valid_t = emissions.valid_mask[t]
            if np.any(valid_t):
                best_path_indices[t] = np.argmax(
                    np.where(valid_t, log_alphas[t], -np.inf)
                )
            else:
                best_path_indices[t] = 0
        
        best_path_arc_ids[t] = emissions.candidate_arc_ids[t, best_path_indices[t]]
    
    return ViterbiResult(
        best_path_arc_ids=best_path_arc_ids,
        best_path_indices=best_path_indices,
        best_log_prob=best_log_prob,
        log_alphas=log_alphas,
        backpointers=backpointers,
        valid_mask=emissions.valid_mask,
    )


def viterbi_decode_padded(
    log_emissions: np.ndarray,      # (T, max_K)
    log_transitions: np.ndarray,    # (T-1, max_K, max_K)
    valid_mask: np.ndarray,         # (T, max_K)
    trans_mask: np.ndarray,         # (T-1, max_K, max_K)
    config: Optional[ViterbiConfig] = None,
) -> ViterbiResult:
    """
    Viterbi decoding using pre-padded matrices (faster for batch processing).
    
    All arrays padded to max_K with -inf / False.
    """
    if config is None:
        config = ViterbiConfig()
    
    T, max_K = log_emissions.shape
    
    if T == 0:
        return ViterbiResult(
            best_path_arc_ids=np.array([], dtype=np.int64),
            best_path_indices=np.array([], dtype=np.int64),
            best_log_prob=-np.inf,
            log_alphas=np.zeros((0, max_K)),
            backpointers=np.zeros((0, max_K), dtype=np.int64),
            valid_mask=valid_mask,
        )
    
    # Initialize
    log_alphas = np.full((T, max_K), -np.inf, dtype=np.float64)
    backpointers = np.zeros((T, max_K), dtype=np.int64)
    
    # t = 0
    log_alphas[0] = np.where(
        valid_mask[0],
        log_emissions[0] + config.initial_log_prob,
        -np.inf
    )
    backpointers[0] = -1
    
    # Forward
    for t in range(1, T):
        # Vectorized: for each k, max over j of (alpha[t-1, j] + trans[t-1, j, k])
        # Shape: (max_K, max_K) + (max_K,) -> (max_K,)
        # Use broadcasting: alpha[t-1, :, None] + trans[t-1, :, :]
        
        prev_alpha = log_alphas[t - 1]  # (max_K,)
        trans_t = log_transitions[t - 1]  # (max_K, max_K)
        trans_mask_t = trans_mask[t - 1]  # (max_K, max_K)
        
        # Mask invalid transitions
        trans_t_masked = np.where(trans_mask_t, trans_t, -np.inf)
        
        # (max_K, max_K) + (max_K, 1) -> (max_K, max_K)
        # Wait: we want max over j (prev), so shape should be (max_K, max_K) where
        # [j, k] = prev_alpha[j] + trans_t[j, k]
        
        combined = prev_alpha[:, None] + trans_t_masked  # (max_K, max_K)
        
        # Max over j (axis 0) for each k
        best_scores = np.max(combined, axis=0)  # (max_K,)
        best_indices = np.argmax(combined, axis=0)  # (max_K,)
        
        # Add emission and mask
        log_alphas[t] = np.where(
            valid_mask[t],
            log_emissions[t] + best_scores,
            -np.inf
        )
        backpointers[t] = best_indices
    
    # Backward
    best_path_indices = np.zeros(T, dtype=np.int64)
    best_path_arc_ids = np.zeros(T, dtype=np.int64)  # Placeholder
    
    # Best final state
    last_scores = np.where(valid_mask[T - 1], log_alphas[T - 1], -np.inf)
    best_last_idx = np.argmax(last_scores)
    best_log_prob = float(last_scores[best_last_idx])
    
    best_path_indices[T - 1] = best_last_idx
    
    for t in range(T - 2, -1, -1):
        next_idx = best_path_indices[t + 1]
        if next_idx < max_K and backpointers[t + 1, next_idx] >= 0:
            best_path_indices[t] = backpointers[t + 1, next_idx]
        else:
            # Fallback
            scores_t = np.where(valid_mask[t], log_alphas[t], -np.inf)
            best_path_indices[t] = np.argmax(scores_t)
    
    return ViterbiResult(
        best_path_arc_ids=best_path_arc_ids,  # Caller must fill from candidate_sets
        best_path_indices=best_path_indices,
        best_log_prob=best_log_prob,
        log_alphas=log_alphas,
        backpointers=backpointers,
        valid_mask=valid_mask,
    )


def reconstruct_path_from_result(
    result: ViterbiResult,
    candidate_sets: List[CandidateSet],
) -> np.ndarray:
    """
    Reconstruct arc ID sequence from ViterbiResult.
    
    The padded version doesn't have arc_ids, so this fills them in.
    """
    T = result.T
    arc_ids = np.zeros(T, dtype=np.int64)
    
    for t in range(T):
        idx = result.best_path_indices[t]
        if idx < len(candidate_sets[t].candidates):
            arc_ids[t] = candidate_sets[t].candidates[idx].arc_id
        else:
            arc_ids[t] = -1
    
    return arc_ids


if __name__ == "__main__":
    # Quick test with simple synthetic data
    from src.features.candidates import Candidate, CandidateSet
    from src.hmm.emission import EmissionScores, EmissionConfig, compute_emission_scores
    from src.hmm.transition import TransitionConfig, compute_transition_scores
    from src.hmm.model_interface import MockModelInterface
    
    # Create synthetic candidate sets
    candidates_1 = [
        Candidate(arc_id=1, distance=5.0, along_track_dist=10.0, bearing_diff=10.0, midpoint_dist=100.0),
        Candidate(arc_id=2, distance=15.0, along_track_dist=20.0, bearing_diff=30.0, midpoint_dist=200.0),
    ]
    candidates_2 = [
        Candidate(arc_id=2, distance=8.0, along_track_dist=15.0, bearing_diff=5.0, midpoint_dist=150.0),
        Candidate(arc_id=3, distance=12.0, along_track_dist=25.0, bearing_diff=20.0, midpoint_dist=180.0),
    ]
    candidates_3 = [
        Candidate(arc_id=3, distance=5.0, along_track_dist=10.0, bearing_diff=8.0, midpoint_dist=80.0),
        Candidate(arc_id=4, distance=20.0, along_track_dist=30.0, bearing_diff=40.0, midpoint_dist=250.0),
    ]
    
    candidate_sets = [
        CandidateSet(0, 8.5, 49.8, 1.0, candidates_1),
        CandidateSet(1, 8.51, 49.81, 2.0, candidates_2),
        CandidateSet(2, 8.52, 49.82, 3.0, candidates_3),
    ]
    
    # Mock arcs for transition
    arcs = np.array([
        [1, 2],  # arc 1
        [2, 3],  # arc 2
        [3, 4],  # arc 3
        [4, 5],  # arc 4
    ])
    
    # Create mock model and emissions
    model = MockModelInterface()
    
    # Build feature matrix
    X_rows = []
    for cs in candidate_sets:
        for cand in cs.candidates:
            X_rows.append([
                cand.distance, cand.along_track_dist, cand.bearing_diff,
                1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 1.0, 0.0
            ])
    X = np.array(X_rows)
    
    # Compute emissions
    emission_config = EmissionConfig(distance_penalty_lambda=0.05)
    emissions = compute_emission_scores_from_model(candidate_sets, model, X, emission_config)
    
    # Compute transitions
    trans_config = TransitionConfig(hop_penalty=1.0, gap_penalty=10.0, max_hops=2)
    transitions = compute_transition_scores(candidate_sets, arcs, trans_config)
    
    # Run Viterbi
    viterbi_config = ViterbiConfig()
    result = viterbi_decode(emissions, transitions, viterbi_config)
    
    # Reconstruct arc IDs
    arc_ids = reconstruct_path_from_result(result, candidate_sets)
    
    print(f"T={result.T}")
    print(f"Best path arc IDs: {arc_ids}")
    print(f"Best path indices: {result.best_path_indices}")
    print(f"Best log prob: {result.best_log_prob:.4f}")
    print(f"Log alphas:\n{result.log_alphas}")
    print("Viterbi test passed!")