"""
Classifier vs HMM comparison utility for PathSnap.

Compares:
1. Classifier-only: Top-1 candidate by model probability at each timestep
2. Classifier + HMM: Viterbi-decoded sequence

Metrics:
- Top-1 Accuracy (per observation)
- Route Continuity (% consecutive predicted arcs connected in graph)
- Sequence F1 (if ground truth available)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.loader import TripData
from src.features.candidates import CandidateSet, generate_candidates_for_trip
from src.features.engineer import build_feature_matrix, compute_adjacency
from src.hmm.emission import EmissionConfig, compute_emission_scores_from_model
from src.hmm.model_interface import ModelInterface
from src.hmm.transition import TransitionConfig, compute_transition_scores
from src.hmm.viterbi import ViterbiConfig, viterbi_decode, reconstruct_path_from_result


@dataclass
class ComparisonConfig:
    """Configuration for comparison."""
    k: int = 10
    radius_meters: float = 100.0
    emission_lambda: float = 0.05
    hop_penalty: float = 1.0
    gap_penalty: float = 10.0
    max_hops: int = 3
    stay_bonus: float = 0.0


@dataclass
class ComparisonMetrics:
    """Metrics for a single method."""
    top1_accuracy: float
    route_continuity: float
    sequence_f1: Optional[float] = None
    macro_f1: Optional[float] = None
    logloss: Optional[float] = None


@dataclass
class ComparisonResult:
    """Results comparing classifier-only vs classifier+HMM."""
    trip_id: str
    n_observations: int
    classifier_metrics: ComparisonMetrics
    hmm_metrics: ComparisonMetrics
    classifier_route: np.ndarray
    hmm_route: np.ndarray
    ground_truth_route: Optional[np.ndarray]
    improvement: Dict[str, float]  # HMM - Classifier
    
    def to_dict(self) -> dict:
        return {
            "trip_id": self.trip_id,
            "n_observations": self.n_observations,
            "classifier": {
                "top1_accuracy": self.classifier_metrics.top1_accuracy,
                "route_continuity": self.classifier_metrics.route_continuity,
                "sequence_f1": self.classifier_metrics.sequence_f1,
                "macro_f1": self.classifier_metrics.macro_f1,
                "logloss": self.classifier_metrics.logloss,
            },
            "hmm": {
                "top1_accuracy": self.hmm_metrics.top1_accuracy,
                "route_continuity": self.hmm_metrics.route_continuity,
                "sequence_f1": self.hmm_metrics.sequence_f1,
                "macro_f1": self.hmm_metrics.macro_f1,
                "logloss": self.hmm_metrics.logloss,
            },
            "classifier_route": self.classifier_route.tolist(),
            "hmm_route": self.hmm_route.tolist(),
            "ground_truth_route": self.ground_truth_route.tolist() if self.ground_truth_route is not None else None,
            "improvement": self.improvement,
        }


def compute_route_continuity(
    route: np.ndarray,
    arcs: np.ndarray,
) -> float:
    """
    Compute route continuity: fraction of consecutive arc pairs that are connected.
    
    Two arcs are connected if end_node of arc_i == start_node of arc_{i+1}.
    """
    if len(route) <= 1:
        return 1.0
    
    # Build set of valid transitions
    valid_transitions = set()
    for arc_id, (from_node, to_node) in enumerate(arcs):
        valid_transitions.add((from_node, to_node))
    
    # Check each consecutive pair
    connected = 0
    for i in range(len(route) - 1):
        arc_i = route[i]
        arc_j = route[i + 1]
        
        if arc_i < 0 or arc_j < 0 or arc_i >= len(arcs) or arc_j >= len(arcs):
            continue
            
        _, to_i = arcs[arc_i]
        from_j, _ = arcs[arc_j]
        
        if to_i == from_j:
            connected += 1
    
    return connected / (len(route) - 1)


def compute_top1_accuracy(
    predicted_route: np.ndarray,
    ground_truth_route: np.ndarray,
    candidate_sets: List[CandidateSet],
) -> float:
    """
    Compute Top-1 accuracy: fraction of observations where predicted arc matches ground truth.
    
    Aligns predicted route (per observation) to ground truth route (proportional).
    """
    if len(ground_truth_route) == 0:
        return 0.0
    
    n_obs = len(candidate_sets)
    route_len = len(ground_truth_route)
    
    # Map observations to ground truth arc indices proportionally
    gt_per_obs = np.zeros(n_obs, dtype=np.int64)
    for i in range(n_obs):
        route_idx = min(int(i * route_len / max(1, n_obs)), route_len - 1)
        gt_per_obs[i] = ground_truth_route[route_idx]
    
    # Compare
    correct = 0
    for i in range(min(n_obs, len(predicted_route))):
        if predicted_route[i] == gt_per_obs[i]:
            correct += 1
    
    return correct / n_obs if n_obs > 0 else 0.0


def compute_sequence_f1(
    predicted_route: np.ndarray,
    ground_truth_route: np.ndarray,
    candidate_sets: List[CandidateSet],
) -> float:
    """
    Compute sequence-level F1: treat each observation as a token, 
    compute F1 over the sequence.
    
    This is equivalent to per-observation accuracy for single-label case.
    """
    # For single-label (one arc per observation), F1 = accuracy
    return compute_top1_accuracy(predicted_route, ground_truth_route, candidate_sets)


def compute_logloss(
    model: ModelInterface,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """Compute log loss using model probabilities."""
    from sklearn.metrics import log_loss
    
    if not hasattr(model, "predict_proba"):
        return float("inf")
    
    try:
        y_proba = model.predict_proba(X)
        if y_proba.shape[1] == 2:
            return log_loss(y, y_proba[:, 1])
        else:
            return log_loss(y, y_proba)
    except Exception:
        return float("inf")


def run_classifier_only(
    candidate_sets: List[CandidateSet],
    model: ModelInterface,
    X: np.ndarray,
) -> Tuple[np.ndarray, ModelInterface]:
    """
    Run classifier-only prediction: pick top-probability candidate per timestep.
    
    Returns:
        (predicted_arc_ids, model)
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must have predict_proba method")
    
    # Get probabilities
    y_proba = model.predict_proba(X)
    if y_proba.shape[1] == 2:
        pos_proba = y_proba[:, 1]
    else:
        pos_proba = y_proba[:, 0]
    
    # Group by observation and pick best
    predicted_arc_ids = []
    start_idx = 0
    
    for cs in candidate_sets:
        k = len(cs.candidates)
        if k == 0:
            predicted_arc_ids.append(-1)
            continue
        
        probs = pos_proba[start_idx:start_idx + k]
        best_local_idx = np.argmax(probs)
        predicted_arc_ids.append(cs.candidates[best_local_idx].arc_id)
        start_idx += k
    
    return np.array(predicted_arc_ids, dtype=np.int64), model


def run_hmm(
    candidate_sets: List[CandidateSet],
    model: ModelInterface,
    X: np.ndarray,
    trip: TripData,
    config: ComparisonConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run full HMM + Viterbi decoding.
    
    Returns:
        (predicted_arc_ids, log_alphas)
    """
    # Compute emissions
    emission_config = EmissionConfig(distance_penalty_lambda=config.emission_lambda)
    emissions = compute_emission_scores_from_model(candidate_sets, model, X, emission_config)
    
    # Compute transitions
    trans_config = TransitionConfig(
        hop_penalty=config.hop_penalty,
        gap_penalty=config.gap_penalty,
        max_hops=config.max_hops,
        stay_bonus=config.stay_bonus,
    )
    transitions = compute_transition_scores(candidate_sets, trip.arcs, trans_config)
    
    # Viterbi decode
    viterbi_config = ViterbiConfig()
    result = viterbi_decode(emissions, transitions, viterbi_config)
    
    # Reconstruct arc IDs
    arc_ids = reconstruct_path_from_result(result, candidate_sets)
    
    return arc_ids, result.log_alphas


def compare_methods(
    trip: TripData,
    model: ModelInterface,
    config: Optional[ComparisonConfig] = None,
) -> ComparisonResult:
    """
    Compare classifier-only vs classifier+HMM on a single trip.
    
    Returns ComparisonResult with metrics and routes.
    """
    if config is None:
        config = ComparisonConfig()
    
    # Generate candidates
    candidate_sets = generate_candidates_for_trip(
        trip, k=config.k, radius_meters=config.radius_meters, ground_truth_route=trip.route
    )
    
    # Build feature matrix
    X, y, feature_names, mapping = build_feature_matrix(candidate_sets, trip, trip.route)
    
    # Classifier-only
    classifier_route, _ = run_classifier_only(candidate_sets, model, X)
    
    # HMM
    hmm_route, _ = run_hmm(candidate_sets, model, X, trip, config)
    
    # Ground truth alignment
    gt_route = trip.route if len(trip.route) > 0 else None
    
    # Compute metrics
    classifier_top1 = compute_top1_accuracy(classifier_route, trip.route, candidate_sets) if gt_route is not None else 0.0
    hmm_top1 = compute_top1_accuracy(hmm_route, trip.route, candidate_sets) if gt_route is not None else 0.0
    
    classifier_continuity = compute_route_continuity(classifier_route, trip.arcs)
    hmm_continuity = compute_route_continuity(hmm_route, trip.arcs)
    
    classifier_f1 = compute_sequence_f1(classifier_route, trip.route, candidate_sets) if gt_route is not None else None
    hmm_f1 = compute_sequence_f1(hmm_route, trip.route, candidate_sets) if gt_route is not None else None
    
    # Logloss (same for both since they use same model probs)
    logloss = compute_logloss(model, X, y)
    
    classifier_metrics = ComparisonMetrics(
        top1_accuracy=classifier_top1,
        route_continuity=classifier_continuity,
        sequence_f1=classifier_f1,
        logloss=logloss,
    )
    
    hmm_metrics = ComparisonMetrics(
        top1_accuracy=hmm_top1,
        route_continuity=hmm_continuity,
        sequence_f1=hmm_f1,
        logloss=logloss,
    )
    
    # Improvements
    improvement = {
        "top1_accuracy": hmm_top1 - classifier_top1,
        "route_continuity": hmm_continuity - classifier_continuity,
        "sequence_f1": (hmm_f1 - classifier_f1) if classifier_f1 is not None else 0.0,
    }
    
    return ComparisonResult(
        trip_id=trip.trip_id,
        n_observations=len(candidate_sets),
        classifier_metrics=classifier_metrics,
        hmm_metrics=hmm_metrics,
        classifier_route=classifier_route,
        hmm_route=hmm_route,
        ground_truth_route=gt_route,
        improvement=improvement,
    )


def compare_on_partition(
    trips: List[TripData],
    model: ModelInterface,
    config: Optional[ComparisonConfig] = None,
) -> List[ComparisonResult]:
    """
    Run comparison on multiple trips and return aggregated results.
    """
    results = []
    
    for trip in trips:
        try:
            result = compare_methods(trip, model, config)
            results.append(result)
        except Exception as e:
            print(f"Error on trip {trip.trip_id}: {e}")
            # Create empty result
            results.append(ComparisonResult(
                trip_id=trip.trip_id,
                n_observations=0,
                classifier_metrics=ComparisonMetrics(0.0, 0.0),
                hmm_metrics=ComparisonMetrics(0.0, 0.0),
                classifier_route=np.array([]),
                hmm_route=np.array([]),
                ground_truth_route=None,
                improvement={},
            ))
    
    return results


def print_comparison_summary(results: List[ComparisonResult]) -> None:
    """Print a formatted comparison summary."""
    print("\n" + "=" * 70)
    print("CLASSIFIER vs HMM COMPARISON SUMMARY")
    print("=" * 70)
    
    n = len(results)
    if n == 0:
        print("No results")
        return
    
    # Aggregate metrics
    classifier_top1 = np.mean([r.classifier_metrics.top1_accuracy for r in results])
    hmm_top1 = np.mean([r.hmm_metrics.top1_accuracy for r in results])
    classifier_cont = np.mean([r.classifier_metrics.route_continuity for r in results])
    hmm_cont = np.mean([r.hmm_metrics.route_continuity for r in results])
    
    print(f"\nTrips evaluated: {n}")
    print(f"\n{'Metric':<30} {'Classifier':>12} {'HMM':>12} {'Delta':>10}")
    print("-" * 64)
    print(f"{'Top-1 Accuracy':<30} {classifier_top1:>12.4f} {hmm_top1:>12.4f} {hmm_top1 - classifier_top1:>+10.4f}")
    print(f"{'Route Continuity':<30} {classifier_cont:>12.4f} {hmm_cont:>12.4f} {hmm_cont - classifier_cont:>+10.4f}")
    
    # Per-trip details
    print("\nPer-trip details:")
    for r in results:
        print(f"  {r.trip_id}: "
              f"Clf Top1={r.classifier_metrics.top1_accuracy:.3f}, "
              f"HMM Top1={r.hmm_metrics.top1_accuracy:.3f}, "
              f"dTop1={r.improvement.get('top1_accuracy', 0):+.3f} | "
              f"Clf Cont={r.classifier_metrics.route_continuity:.3f}, "
              f"HMM Cont={r.hmm_metrics.route_continuity:.3f}, "
              f"dCont={r.improvement.get('route_continuity', 0):+.3f}")


if __name__ == "__main__":
    # Quick test with mock data
    from src.hmm.model_interface import MockModelInterface
    from src.data.loader import TripData
    
    # Create mock trip
    nodes = np.array([
        [8.5, 49.8], [8.6, 49.8], [8.6, 49.9], [8.5, 49.9],
    ])
    arcs = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
    ])
    track = np.array([
        [8.52, 49.8, 1.0],
        [8.55, 49.8, 2.0],
        [8.58, 49.8, 3.0],
    ])
    route = np.array([0, 0, 0])
    
    trip = TripData("test", track, nodes, arcs, route)
    
    model = MockModelInterface()
    config = ComparisonConfig()
    
    result = compare_methods(trip, model, config)
    print_comparison_summary([result])
    print("\nComparison test passed!")