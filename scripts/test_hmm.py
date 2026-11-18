#!/usr/bin/env python
"""
CLI smoke test for PathSnap HMM module.

Runs a complete HMM + Viterbi pipeline on a synthetic trip
using the MockModelInterface (no trained model required).

Usage:
    python -m scripts.test_hmm --help
    python -m scripts.test_hmm --trip-id test --k 5 --radius 50
    python -m scripts.test_hmm --compare --trips 3
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.loader import TripData
from src.features.candidates import generate_candidates_for_trip
from src.features.engineer import build_feature_matrix
from src.hmm import (
    ComparisonConfig,
    EmissionConfig,
    TransitionConfig,
    ViterbiConfig,
    compute_emission_scores_from_model,
    compute_transition_scores,
    compare_methods,
    print_comparison_summary,
    viterbi_decode,
    MockModelInterface,
)


def create_synthetic_trip(trip_id: str = "synthetic", n_obs: int = 10) -> TripData:
    """Create a synthetic trip with a simple road network."""
    # 5 nodes in a line: 0 -> 1 -> 2 -> 3 -> 4
    nodes = np.array([
        [8.5, 49.8],   # node 0
        [8.52, 49.8],  # node 1
        [8.54, 49.8],  # node 2
        [8.56, 49.8],  # node 3
        [8.58, 49.8],  # node 4
    ])
    
    # 4 arcs forming a line
    arcs = np.array([
        [0, 1],  # arc 0
        [1, 2],  # arc 1
        [2, 3],  # arc 2
        [3, 4],  # arc 3
    ])
    
    # GPS observations along the line (with some noise)
    track = np.zeros((n_obs, 3))
    for i in range(n_obs):
        # Position along the line
        progress = i / max(1, n_obs - 1)
        lon = 8.5 + progress * 0.08
        lat = 49.8 + np.random.normal(0, 0.001)  # Small noise
        track[i] = [lon, lat, float(i + 1)]
    
    # Ground truth: mostly arc 0, then arc 1, etc.
    route = np.zeros(n_obs, dtype=np.int64)
    for i in range(n_obs):
        progress = i / max(1, n_obs - 1)
        route[i] = min(int(progress * 4), 3)
    
    return TripData(
        trip_id=trip_id,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route,
    )


def run_hmm_pipeline(
    trip: TripData,
    k: int = 10,
    radius_meters: float = 100.0,
    emission_lambda: float = 0.05,
    hop_penalty: float = 1.0,
    gap_penalty: float = 10.0,
    max_hops: int = 3,
) -> dict:
    """Run the full HMM pipeline on a single trip."""
    
    results = {
        "trip_id": trip.trip_id,
        "config": {
            "k": k,
            "radius_meters": radius_meters,
            "emission_lambda": emission_lambda,
            "hop_penalty": hop_penalty,
            "gap_penalty": gap_penalty,
            "max_hops": max_hops,
        },
        "steps": {},
    }
    
    print(f"\n{'='*60}")
    print(f"PathSnap HMM Smoke Test")
    print(f"{'='*60}")
    print(f"Trip: {trip.trip_id}")
    print(f"Observations: {len(trip.track)}")
    print(f"Road nodes: {len(trip.nodes)}")
    print(f"Road arcs: {len(trip.arcs)}")
    print(f"Ground truth route length: {len(trip.route)}")
    print(f"Config: k={k}, radius={radius_meters}m, lambda={emission_lambda}")
    print()
    
    # Step 1: Generate candidates
    print("Step 1: Generating candidates...")
    start = time.time()
    candidate_sets = generate_candidates_for_trip(
        trip, k=k, radius_meters=radius_meters, ground_truth_route=trip.route
    )
    results["steps"]["candidates"] = {
        "time_seconds": time.time() - start,
        "num_observations": len(candidate_sets),
        "candidates_per_obs": [len(cs.candidates) for cs in candidate_sets],
        "miss_rate": sum(1 for cs in candidate_sets if cs.miss) / len(candidate_sets) if candidate_sets else 0,
    }
    print(f"  Generated in {results['steps']['candidates']['time_seconds']:.3f}s")
    print(f"  Candidates per obs: {results['steps']['candidates']['candidates_per_obs']}")
    print(f"  Miss rate: {results['steps']['candidates']['miss_rate']:.2%}")
    
    # Step 2: Build feature matrix
    print("\nStep 2: Building feature matrix...")
    start = time.time()
    X, y, feature_names, mapping = build_feature_matrix(candidate_sets, trip, trip.route)
    results["steps"]["features"] = {
        "time_seconds": time.time() - start,
        "matrix_shape": list(X.shape),
        "num_features": len(feature_names),
        "positive_labels": int(y.sum()),
        "negative_labels": int((y == 0).sum()),
    }
    print(f"  Built in {results['steps']['features']['time_seconds']:.3f}s")
    print(f"  Matrix shape: {X.shape}")
    print(f"  Positive labels: {results['steps']['features']['positive_labels']}")
    
    # Step 3: Compute emissions
    print("\nStep 3: Computing emission scores...")
    start = time.time()
    model = MockModelInterface()
    emission_config = EmissionConfig(distance_penalty_lambda=emission_lambda)
    emissions = compute_emission_scores_from_model(candidate_sets, model, X, emission_config)
    results["steps"]["emissions"] = {
        "time_seconds": time.time() - start,
        "T": emissions.T,
        "max_K": emissions.max_K,
        "log_emission_range": [
            float(emissions.log_emissions[emissions.valid_mask].min()),
            float(emissions.log_emissions[emissions.valid_mask].max()),
        ],
    }
    print(f"  Computed in {results['steps']['emissions']['time_seconds']:.3f}s")
    print(f"  T={emissions.T}, max_K={emissions.max_K}")
    print(f"  Log emission range: {results['steps']['emissions']['log_emission_range']}")
    
    # Step 4: Compute transitions
    print("\nStep 4: Computing transition scores...")
    start = time.time()
    trans_config = TransitionConfig(
        hop_penalty=hop_penalty,
        gap_penalty=gap_penalty,
        max_hops=max_hops,
    )
    transitions = compute_transition_scores(candidate_sets, trip.arcs, trans_config)
    results["steps"]["transitions"] = {
        "time_seconds": time.time() - start,
        "num_transitions": len(transitions),
        "transition_shapes": [list(t.shape) for t in transitions.log_transitions],
    }
    print(f"  Computed in {results['steps']['transitions']['time_seconds']:.3f}s")
    print(f"  Transitions: {len(transitions)}")
    for i, shape in enumerate(results["steps"]["transitions"]["transition_shapes"]):
        print(f"    t={i}: {shape}")
    
    # Step 5: Viterbi decode
    print("\nStep 5: Running Viterbi decoding...")
    start = time.time()
    viterbi_config = ViterbiConfig()
    viterbi_result = viterbi_decode(emissions, transitions, viterbi_config)
    results["steps"]["viterbi"] = {
        "time_seconds": time.time() - start,
        "best_log_prob": float(viterbi_result.best_log_prob),
        "path_arc_ids": viterbi_result.best_path_arc_ids.tolist(),
        "path_indices": viterbi_result.best_path_indices.tolist(),
    }
    print(f"  Decoded in {results['steps']['viterbi']['time_seconds']:.3f}s")
    print(f"  Best log prob: {viterbi_result.best_log_prob:.4f}")
    print(f"  Predicted route: {viterbi_result.best_path_arc_ids}")
    
    # Step 6: Compare with classifier-only
    print("\nStep 6: Comparing classifier-only vs HMM...")
    start = time.time()
    compare_config = ComparisonConfig(
        k=k,
        radius_meters=radius_meters,
        emission_lambda=emission_lambda,
        hop_penalty=hop_penalty,
        gap_penalty=gap_penalty,
        max_hops=max_hops,
    )
    comparison = compare_methods(trip, model, compare_config)
    results["steps"]["comparison"] = {
        "time_seconds": time.time() - start,
        "classifier_top1": comparison.classifier_metrics.top1_accuracy,
        "hmm_top1": comparison.hmm_metrics.top1_accuracy,
        "classifier_continuity": comparison.classifier_metrics.route_continuity,
        "hmm_continuity": comparison.hmm_metrics.route_continuity,
        "improvement_top1": comparison.improvement["top1_accuracy"],
        "improvement_continuity": comparison.improvement["route_continuity"],
    }
    print(f"  Compared in {results['steps']['comparison']['time_seconds']:.3f}s")
    print(f"  Classifier Top-1: {comparison.classifier_metrics.top1_accuracy:.4f}")
    print(f"  HMM Top-1: {comparison.hmm_metrics.top1_accuracy:.4f}")
    print(f"  Classifier Continuity: {comparison.classifier_metrics.route_continuity:.4f}")
    print(f"  HMM Continuity: {comparison.hmm_metrics.route_continuity:.4f}")
    
    # Summary
    total_time = sum(s.get("time_seconds", 0) for s in results["steps"].values())
    results["total_time_seconds"] = total_time
    
    print(f"\n{'='*60}")
    print(f"HMM SMOKE TEST PASSED")
    print(f"Total time: {total_time:.3f}s")
    print(f"{'='*60}")
    
    return results


def run_multi_trip_comparison(
    num_trips: int = 3,
    k: int = 10,
    radius_meters: float = 100.0,
) -> dict:
    """Run comparison on multiple synthetic trips."""
    
    print(f"\n{'='*60}")
    print(f"Multi-Trip Comparison ({num_trips} trips)")
    print(f"{'='*60}")
    
    model = MockModelInterface()
    config = ComparisonConfig(k=k, radius_meters=radius_meters)
    
    all_results = []
    
    for i in range(num_trips):
        trip = create_synthetic_trip(f"synthetic_{i:03d}", n_obs=15)
        result = compare_methods(trip, model, config)
        all_results.append(result)
        print(f"  Trip {i+1}/{num_trips}: "
              f"Clf={result.classifier_metrics.top1_accuracy:.3f}, "
              f"HMM={result.hmm_metrics.top1_accuracy:.3f}, "
              f"dTop1={result.improvement['top1_accuracy']:+.3f} | "
              f"Cont: Clf={result.classifier_metrics.route_continuity:.3f}, "
              f"HMM={result.hmm_metrics.route_continuity:.3f}")
    
    print_comparison_summary(all_results)
    
    # Aggregate results
    agg = {
        "num_trips": num_trips,
        "classifier_avg_top1": float(np.mean([r.classifier_metrics.top1_accuracy for r in all_results])),
        "hmm_avg_top1": float(np.mean([r.hmm_metrics.top1_accuracy for r in all_results])),
        "classifier_avg_continuity": float(np.mean([r.classifier_metrics.route_continuity for r in all_results])),
        "hmm_avg_continuity": float(np.mean([r.hmm_metrics.route_continuity for r in all_results])),
        "per_trip": [r.to_dict() for r in all_results],
    }
    
    return agg


def main():
    parser = argparse.ArgumentParser(description="PathSnap HMM smoke test")
    parser.add_argument("--trip-id", type=str, default="synthetic", help="Trip ID for single test")
    parser.add_argument("--n-obs", type=int, default=10, help="Number of observations for synthetic trip")
    parser.add_argument("--k", type=int, default=10, help="Number of candidates per observation")
    parser.add_argument("--radius", type=float, default=100.0, help="Search radius in meters")
    parser.add_argument("--lambda", dest="emission_lambda", type=float, default=0.05, help="Distance penalty lambda")
    parser.add_argument("--hop-penalty", type=float, default=1.0, help="Transition hop penalty")
    parser.add_argument("--gap-penalty", type=float, default=10.0, help="Transition gap penalty")
    parser.add_argument("--max-hops", type=int, default=3, help="Maximum hops for connected transitions")
    parser.add_argument("--compare", action="store_true", help="Run multi-trip comparison")
    parser.add_argument("--trips", type=int, default=3, help="Number of trips for comparison")
    parser.add_argument("--output", type=Path, default=Path("experiments/hmm_smoke_test.json"), help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    if args.compare:
        # Run multi-trip comparison
        results = run_multi_trip_comparison(
            num_trips=args.trips,
            k=args.k,
            radius_meters=args.radius,
        )
    else:
        # Run single trip pipeline
        trip = create_synthetic_trip(args.trip_id, n_obs=args.n_obs)
        results = run_hmm_pipeline(
            trip,
            k=args.k,
            radius_meters=args.radius,
            emission_lambda=args.emission_lambda,
            hop_penalty=args.hop_penalty,
            gap_penalty=args.gap_penalty,
            max_hops=args.max_hops,
        )
    
    # Save results
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(args.output, "w") as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())