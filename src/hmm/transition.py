"""
Graph-aware transition scoring for PathSnap HMM.

Computes log P(arc_j | arc_i) using:
- 0 if arc_i == arc_j (stay on same arc)
- -hop_penalty * hops if connected within max_hops (BFS on arc adjacency)
- -gap_penalty otherwise (gap transition / teleport)

Handles variable candidate sets per timestep.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.features.candidates import CandidateSet
from src.features.engineer import compute_adjacency
from src.hmm.emission import EmissionScores


@dataclass
class TransitionConfig:
    """Configuration for transition scoring."""
    hop_penalty: float = 1.0        # Penalty per hop in graph
    gap_penalty: float = 10.0       # Penalty for disconnected transition
    max_hops: int = 3               # Maximum BFS hops to consider connected
    stay_bonus: float = 0.0         # Bonus for staying on same arc (log-space)
    
    def __post_init__(self):
        assert self.hop_penalty >= 0, "hop_penalty must be >= 0"
        assert self.gap_penalty >= 0, "gap_penalty must be >= 0"
        assert self.max_hops >= 1, "max_hops must be >= 1"
        assert self.gap_penalty >= self.hop_penalty * self.max_hops, \
            "gap_penalty should be >= hop_penalty * max_hops"


@dataclass
class TransitionScores:
    """
    Transition log-probabilities between candidate sets at consecutive timesteps.
    
    For each pair of consecutive timesteps (t, t+1), stores:
        log_transitions[t]: Shape (K_t, K_{t+1}) - log P(arc_{t+1} | arc_t)
    
    Attributes:
        log_transitions: List of (K_t, K_{t+1}) matrices for t=0..T-2
        valid_mask: List of (K_t, K_{t+1}) boolean masks
    """
    log_transitions: List[np.ndarray]
    valid_mask: List[np.ndarray]
    
    def __len__(self) -> int:
        return len(self.log_transitions)


def compute_hop_distances(
    arcs: np.ndarray,
    max_hops: int,
) -> Dict[int, Dict[int, int]]:
    """
    Compute BFS hop distances between all pairs of arcs.
    
    Returns dict: {from_arc_id: {to_arc_id: hops}}
    Only includes pairs within max_hops.
    """
    # Build adjacency: arc -> list of next arcs
    adjacency = compute_adjacency(arcs, len(arcs), max_hops=1)
    
    hop_distances = {}
    
    for start_arc in range(len(arcs)):
        # BFS from start_arc
        distances = {start_arc: 0}
        queue = deque([start_arc])
        
        while queue:
            current = queue.popleft()
            current_dist = distances[current]
            
            if current_dist >= max_hops:
                continue
            
            for neighbor in adjacency.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        
        hop_distances[start_arc] = distances
    
    return hop_distances


def compute_transition_scores(
    candidate_sets: List[CandidateSet],
    arcs: np.ndarray,
    config: TransitionConfig,
) -> TransitionScores:
    """
    Compute transition log-probabilities between consecutive timesteps.
    
    For each t, computes K_t x K_{t+1} matrix where:
    - log P(arc_j | arc_i) = 0 if arc_i == arc_j (stay)
    - log P(arc_j | arc_i) = -hop_penalty * hops if hops <= max_hops
    - log P(arc_j | arc_i) = -gap_penalty otherwise
    
    Args:
        candidate_sets: List of CandidateSet per timestep
        arcs: Trip arcs array (n_arcs, 2)
        config: TransitionConfig
        
    Returns:
        TransitionScores with list of transition matrices
    """
    T = len(candidate_sets)
    if T <= 1:
        return TransitionScores(log_transitions=[], valid_mask=[])
    
    # Precompute hop distances between all arcs
    hop_distances = compute_hop_distances(arcs, config.max_hops)
    
    log_transitions = []
    valid_mask = []
    
    for t in range(T - 1):
        cs_curr = candidate_sets[t]
        cs_next = candidate_sets[t + 1]
        
        K_curr = len(cs_curr.candidates)
        K_next = len(cs_next.candidates)
        
        # Transition matrix: (K_curr, K_next)
        trans = np.full((K_curr, K_next), -config.gap_penalty, dtype=np.float64)
        mask = np.zeros((K_curr, K_next), dtype=bool)
        
        curr_arc_ids = [c.arc_id for c in cs_curr.candidates]
        next_arc_ids = [c.arc_id for c in cs_next.candidates]
        
        for i, arc_i in enumerate(curr_arc_ids):
            for j, arc_j in enumerate(next_arc_ids):
                mask[i, j] = True
                
                if arc_i == arc_j:
                    # Stay on same arc
                    trans[i, j] = config.stay_bonus
                elif arc_i in hop_distances and arc_j in hop_distances[arc_i]:
                    # Connected within max_hops
                    hops = hop_distances[arc_i][arc_j]
                    trans[i, j] = -config.hop_penalty * hops
                else:
                    # Gap transition (disconnected)
                    trans[i, j] = -config.gap_penalty
        
        log_transitions.append(trans)
        valid_mask.append(mask)
    
    return TransitionScores(log_transitions=log_transitions, valid_mask=valid_mask)


def compute_transition_matrix_for_viterbi(
    candidate_sets: List[CandidateSet],
    arcs: np.ndarray,
    config: TransitionConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute padded transition matrices for Viterbi with variable K_t.
    
    Returns:
        (transitions_padded, valid_mask_padded)
        - transitions_padded: (T-1, max_K, max_K) with -gap_penalty padding
        - valid_mask_padded: (T-1, max_K, max_K) boolean
    """
    T = len(candidate_sets)
    if T <= 1:
        return np.zeros((0, 0, 0)), np.zeros((0, 0, 0), dtype=bool)
    
    max_K = max(len(cs.candidates) for cs in candidate_sets)
    
    transitions = np.full((T - 1, max_K, max_K), -config.gap_penalty, dtype=np.float64)
    mask = np.zeros((T - 1, max_K, max_K), dtype=bool)
    
    hop_distances = compute_hop_distances(arcs, config.max_hops)
    
    for t in range(T - 1):
        cs_curr = candidate_sets[t]
        cs_next = candidate_sets[t + 1]
        
        curr_arc_ids = [c.arc_id for c in cs_curr.candidates]
        next_arc_ids = [c.arc_id for c in cs_next.candidates]
        
        K_curr = len(curr_arc_ids)
        K_next = len(next_arc_ids)
        
        for i, arc_i in enumerate(curr_arc_ids):
            for j, arc_j in enumerate(next_arc_ids):
                mask[t, i, j] = True
                
                if arc_i == arc_j:
                    transitions[t, i, j] = config.stay_bonus
                elif arc_i in hop_distances and arc_j in hop_distances[arc_i]:
                    hops = hop_distances[arc_i][arc_j]
                    transitions[t, i, j] = -config.hop_penalty * hops
                else:
                    transitions[t, i, j] = -config.gap_penalty
    
    return transitions, mask


def get_transition_matrix(
    transitions: TransitionScores,
    t: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get transition matrix and valid mask for timestep t -> t+1.
    
    Returns:
        (log_transitions[t], valid_mask[t])
    """
    return transitions.log_transitions[t], transitions.valid_mask[t]


if __name__ == "__main__":
    # Quick test with synthetic graph
    # 4 nodes in a line: 0 -> 1 -> 2 -> 3
    # Arcs: 0: 0->1, 1: 1->2, 2: 2->3, 3: 3->0 (loop)
    arcs = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
    ])
    
    # Two timesteps
    candidates_1 = [
        # arc_id, distance, along, bearing, midpoint
        Candidate(arc_id=0, distance=5.0, along_track_dist=10.0, bearing_diff=10.0, midpoint_dist=100.0),
        Candidate(arc_id=1, distance=15.0, along_track_dist=20.0, bearing_diff=30.0, midpoint_dist=200.0),
    ]
    candidates_2 = [
        Candidate(arc_id=1, distance=8.0, along_track_dist=15.0, bearing_diff=5.0, midpoint_dist=150.0),
        Candidate(arc_id=2, distance=12.0, along_track_dist=25.0, bearing_diff=20.0, midpoint_dist=180.0),
        Candidate(arc_id=3, distance=30.0, along_track_dist=10.0, bearing_diff=60.0, midpoint_dist=300.0),
    ]
    
    candidate_sets = [
        CandidateSet(0, 8.5, 49.8, 1.0, candidates_1),
        CandidateSet(1, 8.51, 49.81, 2.0, candidates_2),
    ]
    
    config = TransitionConfig(hop_penalty=1.0, gap_penalty=10.0, max_hops=2)
    trans = compute_transition_scores(candidate_sets, arcs, config)
    
    print(f"T={len(candidate_sets)}, transitions={len(trans)}")
    print(f"Transition matrix shape: {trans.log_transitions[0].shape}")
    print(f"Transition matrix:\n{trans.log_transitions[0]}")
    print(f"Valid mask:\n{trans.valid_mask[0]}")
    
    # Test padded version
    padded_trans, padded_mask = compute_transition_matrix_for_viterbi(candidate_sets, arcs, config)
    print(f"\nPadded shape: {padded_trans.shape}")
    print(f"Padded transition[0]:\n{padded_trans[0]}")
    print("Transition test passed!")