"""
Feature engineering for PathSnap map-matching.

Extracts features for each candidate arc using only information available at inference time:
- Geometric: perpendicular distance, along-track distance, bearing difference, speed
- Topological: node degree, arc length, connectivity
- Temporal: time delta, timestamp consistency
- Context: candidate rank, previous candidate info (for transition prior)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from src.data.loader import TripData
from src.features.candidates import Candidate, CandidateSet, haversine_distance, bearing


@dataclass
class FeatureVector:
    """Feature vector for a single candidate."""
    candidate: Candidate
    features: np.ndarray  # 1D array of feature values
    feature_names: List[str]
    label: Optional[int] = None  # 1 if ground truth, 0 otherwise
    
    def to_dict(self) -> dict:
        return {name: val for name, val in zip(self.feature_names, self.features)}


def compute_node_degrees(arcs: np.ndarray, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute in-degree and out-degree for each node."""
    in_degree = np.zeros(num_nodes, dtype=np.int32)
    out_degree = np.zeros(num_nodes, dtype=np.int32)
    
    for from_node, to_node in arcs:
        out_degree[from_node] += 1
        in_degree[to_node] += 1
    
    return in_degree, out_degree


def compute_arc_lengths(nodes: np.ndarray, arcs: np.ndarray) -> np.ndarray:
    """Compute length of each arc in meters."""
    lengths = np.zeros(len(arcs), dtype=np.float64)
    for i, (from_node, to_node) in enumerate(arcs):
        lon1, lat1 = nodes[from_node]
        lon2, lat2 = nodes[to_node]
        lengths[i] = haversine_distance(lon1, lat1, lon2, lat2)
    return lengths


def compute_adjacency(arcs: np.ndarray, num_arcs: int, max_hops: int = 2) -> Dict[int, List[int]]:
    """
    Compute adjacency list for arcs (which arcs connect to which).
    Two arcs are adjacent if the end node of one equals the start node of another.
    """
    # Build node -> outgoing arcs mapping
    node_to_outgoing = {}
    for arc_id, (from_node, to_node) in enumerate(arcs):
        if from_node not in node_to_outgoing:
            node_to_outgoing[from_node] = []
        node_to_outgoing[from_node].append(arc_id)
    
    # Build adjacency
    adjacency = {i: [] for i in range(num_arcs)}
    for arc_id, (from_node, to_node) in enumerate(arcs):
        # Arcs that can follow this one
        if to_node in node_to_outgoing:
            adjacency[arc_id].extend(node_to_outgoing[to_node])
    
    return adjacency


def extract_candidate_features(
    candidate_set: CandidateSet,
    trip: TripData,
    arc_lengths: np.ndarray,
    in_degree: np.ndarray,
    out_degree: np.ndarray,
    adjacency: Dict[int, List[int]],
    prev_candidate_arc: Optional[int] = None,
    prev_timestamp: Optional[float] = None,
) -> FeatureVector:
    """
    Extract all features for a single candidate.
    
    Features (in order):
    1. perpendicular_distance (m)
    2. along_track_distance (m)
    3. bearing_difference (degrees)
    4. speed (m/s) - from time delta
    5. time_delta (s)
    6. arc_length (m)
    7. from_node_out_degree
    8. to_node_in_degree
    9. arc_connectivity (num adjacent arcs)
    10. candidate_rank (1-based, by distance)
    11. is_continuation (1 if prev_candidate_arc connects to this arc)
    """
    cand = candidate_set.candidate
    obs_idx = candidate_set.obs_idx
    
    # Get arc info
    arc_id = cand.arc_id
    from_node, to_node = trip.arcs[arc_id]
    
    # Geometric features
    perp_dist = cand.distance
    along_dist = cand.along_track_dist
    bearing_diff = cand.bearing_diff
    
    # Temporal features
    if prev_timestamp is not None and candidate_set.timestamp > prev_timestamp:
        time_delta = candidate_set.timestamp - prev_timestamp
        # Speed in m/s (using perpendicular distance as proxy for displacement)
        speed = perp_dist / max(time_delta, 0.1) if time_delta > 0 else 0.0
    else:
        time_delta = 1.0  # Default 1 second
        speed = perp_dist / time_delta
    
    # Topological features
    arc_len = arc_lengths[arc_id] if arc_id < len(arc_lengths) else 0.0
    from_deg = out_degree[from_node] if from_node < len(out_degree) else 0
    to_deg = in_degree[to_node] if to_node < len(in_degree) else 0
    connectivity = len(adjacency.get(arc_id, []))
    
    # Context features
    candidate_rank = 1  # Will be set by caller based on sorted order
    
    is_continuation = 0
    if prev_candidate_arc is not None and prev_candidate_arc in adjacency.get(arc_id, []):
        is_continuation = 1
    
    features = np.array([
        perp_dist,
        along_dist,
        bearing_diff,
        speed,
        time_delta,
        arc_len,
        float(from_deg),
        float(to_deg),
        float(connectivity),
        float(candidate_rank),
        float(is_continuation),
    ], dtype=np.float64)
    
    feature_names = [
        "perp_distance_m",
        "along_track_dist_m",
        "bearing_diff_deg",
        "speed_m_s",
        "time_delta_s",
        "arc_length_m",
        "from_node_out_degree",
        "to_node_in_degree",
        "arc_connectivity",
        "candidate_rank",
        "is_continuation",
    ]
    
    return FeatureVector(
        candidate=cand,
        features=features,
        feature_names=feature_names,
        label=None  # Set by caller
    )


def build_feature_matrix(
    candidate_sets: List[CandidateSet],
    trip: TripData,
    ground_truth_arcs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[int, int]]]:
    """
    Build feature matrix for all candidates in a trip.
    
    Returns:
        X: (n_candidates, n_features) feature matrix
        y: (n_candidates,) binary labels (1 if ground truth)
        feature_names: List of feature names
        candidate_mapping: List of (obs_idx, candidate_rank) for each row
    """
    # Precompute trip-level topology
    arc_lengths = compute_arc_lengths(trip.nodes, trip.arcs)
    in_degree, out_degree = compute_node_degrees(trip.arcs, len(trip.nodes))
    adjacency = compute_adjacency(trip.arcs, len(trip.arcs))
    
    # Align ground truth to observations
    gt_arc_per_obs = None
    if ground_truth_arcs is not None and len(ground_truth_arcs) > 0:
        n_obs = len(candidate_sets)
        route_len = len(ground_truth_arcs)
        gt_arc_per_obs = np.zeros(n_obs, dtype=np.int64)
        for i in range(n_obs):
            route_idx = min(int(i * route_len / max(1, n_obs)), route_len - 1)
            gt_arc_per_obs[i] = ground_truth_arcs[route_idx]
    
    all_features = []
    all_labels = []
    candidate_mapping = []
    feature_names = None
    
    prev_timestamp = None
    prev_candidate_arc = None
    
    for obs_idx, cand_set in enumerate(candidate_sets):
        gt_arc = gt_arc_per_obs[obs_idx] if gt_arc_per_obs is not None else None
        
        # Sort candidates by distance (they should already be sorted)
        sorted_candidates = sorted(cand_set.candidates)
        
        for rank, cand in enumerate(sorted_candidates):
            fv = extract_candidate_features(
                cand_set, trip, arc_lengths, in_degree, out_degree, adjacency,
                prev_candidate_arc=prev_candidate_arc,
                prev_timestamp=prev_timestamp,
            )
            # Update candidate rank
            fv.features[9] = float(rank + 1)
            
            # Label: 1 if this candidate is the ground truth arc
            label = 1 if (gt_arc is not None and cand.arc_id == gt_arc) else 0
            
            all_features.append(fv.features)
            all_labels.append(label)
            candidate_mapping.append((obs_idx, rank))
            
            if feature_names is None:
                feature_names = fv.feature_names
        
        # Update previous for next observation
        prev_timestamp = cand_set.timestamp
        # Use top-ranked candidate as "previous" for continuation feature
        if sorted_candidates:
            prev_candidate_arc = sorted_candidates[0].arc_id
    
    X = np.array(all_features, dtype=np.float64)
    y = np.array(all_labels, dtype=np.int64)
    
    return X, y, feature_names, candidate_mapping


def build_features_for_partition(
    trips: List[TripData],
    all_candidates: Dict[str, List[CandidateSet]],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[str, int, int]]]:
    """
    Build combined feature matrix for all trips in a partition.
    
    Returns:
        X: Combined feature matrix
        y: Combined labels
        feature_names: Feature names
        mapping: List of (trip_id, obs_idx, candidate_rank) for each row
    """
    all_X = []
    all_y = []
    all_mapping = []
    feature_names = None
    
    for trip in trips:
        candidates = all_candidates.get(trip.trip_id, [])
        if not candidates:
            continue
        
        X, y, names, mapping = build_feature_matrix(candidates, trip, trip.route)
        
        all_X.append(X)
        all_y.append(y)
        for obs_idx, rank in mapping:
            all_mapping.append((trip.trip_id, obs_idx, rank))
        
        if feature_names is None:
            feature_names = names
    
    if not all_X:
        return np.array([]), np.array([]), [], []
    
    X_combined = np.vstack(all_X)
    y_combined = np.hstack(all_y)
    
    return X_combined, y_combined, feature_names, all_mapping


def save_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    mapping: List[Tuple],
    filepath: Path,
) -> None:
    """Save features to compressed NPZ file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        filepath,
        X=X,
        y=y,
        feature_names=np.array(feature_names, dtype=object),
        mapping=np.array(mapping, dtype=object),
    )


def load_features(filepath: Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple]]:
    """Load features from NPZ file."""
    data = np.load(filepath, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    feature_names = data["feature_names"].tolist()
    mapping = data["mapping"].tolist()
    return X, y, feature_names, mapping


if __name__ == "__main__":
    # Quick test
    print("Feature engineering module loaded successfully")
    
    # Test haversine
    dist = haversine_distance(8.5, 49.8, 8.6, 49.9)
    print(f"Haversine test: {dist:.1f} meters")