#!/usr/bin/env python
"""
Smoke test for PathSnap preprocessing pipeline.

Runs the full data loading -> candidate generation -> feature engineering
pipeline on one train trip and one student_test trip.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.loader import load_trip, load_partition, get_dataset_stats
from src.features.candidates import generate_candidates_for_trip
from src.features.engineer import build_feature_matrix, build_features_for_partition


def run_smoke_test(
    data_root: Path = Path("Student folder/Student folder"),
    train_trip_id: str = "00000001",
    test_trip_id: str = "00000000",
    k: int = 10,
    radius_meters: float = 100.0,
) -> dict:
    """Run smoke test on specified trips."""
    
    results = {
        "train_trip": train_trip_id,
        "test_trip": test_trip_id,
        "k": k,
        "radius_meters": radius_meters,
        "steps": {},
    }
    
    print(f"{'='*60}")
    print(f"PathSnap Smoke Test")
    print(f"{'='*60}")
    print(f"Data root: {data_root}")
    print(f"Train trip: {train_trip_id}")
    print(f"Test trip: {test_trip_id}")
    print(f"K={k}, Radius={radius_meters}m")
    print()
    
    # Step 1: Load train trip
    print("Step 1: Loading train trip...")
    start = time.time()
    train_trip_path = data_root / "train" / train_trip_id
    train_trip = load_trip(train_trip_path, train_trip_id)
    results["steps"]["load_train_trip"] = {
        "time_seconds": time.time() - start,
        "num_observations": len(train_trip.track),
        "num_nodes": len(train_trip.nodes),
        "num_arcs": len(train_trip.arcs),
        "route_length": len(train_trip.route),
    }
    print(f"  Loaded in {results['steps']['load_train_trip']['time_seconds']:.2f}s")
    print(f"  Observations: {results['steps']['load_train_trip']['num_observations']}")
    print(f"  Nodes: {results['steps']['load_train_trip']['num_nodes']}")
    print(f"  Arcs: {results['steps']['load_train_trip']['num_arcs']}")
    print(f"  Route length: {results['steps']['load_train_trip']['route_length']}")
    print()
    
    # Step 2: Load test trip
    print("Step 2: Loading student_test trip...")
    start = time.time()
    test_trip_path = data_root / "student_test" / test_trip_id
    test_trip = load_trip(test_trip_path, test_trip_id)
    results["steps"]["load_test_trip"] = {
        "time_seconds": time.time() - start,
        "num_observations": len(test_trip.track),
        "num_nodes": len(test_trip.nodes),
        "num_arcs": len(test_trip.arcs),
        "route_length": len(test_trip.route),
    }
    print(f"  Loaded in {results['steps']['load_test_trip']['time_seconds']:.2f}s")
    print(f"  Observations: {results['steps']['load_test_trip']['num_observations']}")
    print(f"  Nodes: {results['steps']['load_test_trip']['num_nodes']}")
    print(f"  Arcs: {results['steps']['load_test_trip']['num_arcs']}")
    print(f"  Route length: {results['steps']['load_test_trip']['route_length']}")
    print()
    
    # Step 3: Generate candidates for train trip
    print("Step 3: Generating candidates for train trip...")
    start = time.time()
    train_candidates = generate_candidates_for_trip(
        train_trip, k=k, radius_meters=radius_meters, ground_truth_route=train_trip.route
    )
    results["steps"]["candidates_train"] = {
        "time_seconds": time.time() - start,
        "num_observations": len(train_candidates),
        "candidates_per_obs": [len(cs.candidates) for cs in train_candidates],
        "miss_rate": sum(1 for cs in train_candidates if cs.miss) / len(train_candidates) if train_candidates else 0,
    }
    print(f"  Generated in {results['steps']['candidates_train']['time_seconds']:.2f}s")
    print(f"  Candidates per obs: min={min(results['steps']['candidates_train']['candidates_per_obs'])}, "
          f"max={max(results['steps']['candidates_train']['candidates_per_obs'])}, "
          f"avg={sum(results['steps']['candidates_train']['candidates_per_obs'])/len(train_candidates):.1f}")
    print(f"  Miss rate: {results['steps']['candidates_train']['miss_rate']:.2%}")
    print()
    
    # Step 4: Generate candidates for test trip
    print("Step 4: Generating candidates for test trip...")
    start = time.time()
    test_candidates = generate_candidates_for_trip(
        test_trip, k=k, radius_meters=radius_meters, ground_truth_route=test_trip.route
    )
    results["steps"]["candidates_test"] = {
        "time_seconds": time.time() - start,
        "num_observations": len(test_candidates),
        "candidates_per_obs": [len(cs.candidates) for cs in test_candidates],
        "miss_rate": sum(1 for cs in test_candidates if cs.miss) / len(test_candidates) if test_candidates else 0,
    }
    print(f"  Generated in {results['steps']['candidates_test']['time_seconds']:.2f}s")
    print(f"  Candidates per obs: min={min(results['steps']['candidates_test']['candidates_per_obs'])}, "
          f"max={max(results['steps']['candidates_test']['candidates_per_obs'])}, "
          f"avg={sum(results['steps']['candidates_test']['candidates_per_obs'])/len(test_candidates):.1f}")
    print(f"  Miss rate: {results['steps']['candidates_test']['miss_rate']:.2%}")
    print()
    
    # Step 5: Build features for train trip
    print("Step 5: Building feature matrix for train trip...")
    start = time.time()
    X_train, y_train, feature_names, mapping_train = build_feature_matrix(
        train_candidates, train_trip, train_trip.route
    )
    results["steps"]["features_train"] = {
        "time_seconds": time.time() - start,
        "matrix_shape": list(X_train.shape),
        "num_features": len(feature_names),
        "positive_labels": int(y_train.sum()),
        "negative_labels": int((y_train == 0).sum()),
        "feature_names": feature_names,
    }
    print(f"  Built in {results['steps']['features_train']['time_seconds']:.2f}s")
    print(f"  Matrix shape: {X_train.shape}")
    print(f"  Features: {feature_names}")
    print(f"  Positive labels: {results['steps']['features_train']['positive_labels']}")
    print(f"  Negative labels: {results['steps']['features_train']['negative_labels']}")
    print()
    
    # Step 6: Build features for test trip
    print("Step 6: Building feature matrix for test trip...")
    start = time.time()
    X_test, y_test, _, mapping_test = build_feature_matrix(
        test_candidates, test_trip, test_trip.route
    )
    results["steps"]["features_test"] = {
        "time_seconds": time.time() - start,
        "matrix_shape": list(X_test.shape),
        "positive_labels": int(y_test.sum()),
        "negative_labels": int((y_test == 0).sum()),
    }
    print(f"  Built in {results['steps']['features_test']['time_seconds']:.2f}s")
    print(f"  Matrix shape: {X_test.shape}")
    print(f"  Positive labels: {results['steps']['features_test']['positive_labels']}")
    print(f"  Negative labels: {results['steps']['features_test']['negative_labels']}")
    print()
    
    # Step 7: Verify no NaN/inf values
    print("Step 7: Validating feature matrices...")
    assert not np.any(np.isnan(X_train)), "NaN values in train features!"
    assert not np.any(np.isinf(X_train)), "Inf values in train features!"
    assert not np.any(np.isnan(X_test)), "NaN values in test features!"
    assert not np.any(np.isinf(X_test)), "Inf values in test features!"
    results["steps"]["validation"] = {"passed": True}
    print("  All features finite - PASSED")
    print()
    
    # Summary
    total_time = sum(s.get("time_seconds", 0) for s in results["steps"].values())
    results["total_time_seconds"] = total_time
    
    print(f"{'='*60}")
    print(f"SMOKE TEST PASSED")
    print(f"Total time: {total_time:.2f}s")
    print(f"{'='*60}")
    
    return results


def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="PathSnap smoke test")
    parser.add_argument("--data-root", type=Path, default=Path("Student folder/Student folder"))
    parser.add_argument("--train-trip", type=str, default="00000001")
    parser.add_argument("--test-trip", type=str, default="00000000")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--radius", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=Path("experiments/smoke_test_results.json"))
    args = parser.parse_args()
    
    results = run_smoke_test(
        data_root=args.data_root,
        train_trip_id=args.train_trip,
        test_trip_id=args.test_trip,
        k=args.k,
        radius_meters=args.radius,
    )
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    import numpy as np
    main()