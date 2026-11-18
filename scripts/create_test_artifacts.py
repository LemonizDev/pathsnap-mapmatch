#!/usr/bin/env python
"""Create test model and HMM artifacts for real-artifact mode testing."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_test_artifacts(output_dir: Path) -> dict:
    """Create test model and HMM artifacts for testing real-artifact mode."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Feature names (must match the 11 features from feature engineering)
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
    
    # Create a simple pipeline with scaler + logistic regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42, max_iter=1000)),
    ])
    
    # Generate synthetic training data that mimics the feature distributions
    np.random.seed(42)
    n_samples = 5000
    X_train = np.zeros((n_samples, 11))
    X_train[:, 0] = np.random.exponential(20, n_samples)  # perp_distance: mostly small
    X_train[:, 1] = np.random.uniform(0, 500, n_samples)  # along_track_dist
    X_train[:, 2] = np.random.exponential(30, n_samples)  # bearing_diff
    X_train[:, 3] = np.random.exponential(5, n_samples)   # speed
    X_train[:, 4] = np.random.uniform(0.1, 10, n_samples) # time_delta
    X_train[:, 5] = np.random.uniform(10, 500, n_samples) # arc_length
    X_train[:, 6] = np.random.poisson(2, n_samples)       # from_node_out_degree
    X_train[:, 7] = np.random.poisson(2, n_samples)       # to_node_in_degree
    X_train[:, 8] = np.random.poisson(3, n_samples)       # arc_connectivity
    X_train[:, 9] = np.random.randint(1, 11, n_samples)   # candidate_rank
    X_train[:, 10] = np.random.binomial(1, 0.3, n_samples) # is_continuation
    
    # Create labels: lower distance/rank -> more likely positive
    score = (
        -X_train[:, 0] / 50.0 +
        -X_train[:, 9] / 10.0 +
        -X_train[:, 2] / 45.0 +
        np.random.normal(0, 0.5, n_samples)
    )
    y_train = (score > np.median(score)).astype(int)
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Save model
    model_path = output_dir / "test_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")
    
    # Save model metadata
    meta_path = output_dir / "test_model_meta.json"
    meta = {
        "model_name": "TestLogisticRegression",
        "feature_names": feature_names,
        "n_features": len(feature_names),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model metadata to {meta_path}")
    
    # Save HMM parameters
    hmm_path = output_dir / "test_hmm_params.json"
    hmm_params = {
        "distance_penalty_lambda": 0.05,
        "min_log_prob": -100.0,
        "max_log_prob": 0.0,
        "hop_penalty": 1.0,
        "gap_penalty": 10.0,
        "max_hops": 3,
        "stay_bonus": 0.0,
        "initial_log_prob": 0.0,
    }
    with open(hmm_path, "w") as f:
        json.dump(hmm_params, f, indent=2)
    print(f"Saved HMM params to {hmm_path}")
    
    return {
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "hmm_path": str(hmm_path),
        "feature_names": feature_names,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create test artifacts for real-artifact mode testing")
    parser.add_argument("--output-dir", type=Path, default=Path("test_artifacts"), help="Output directory")
    args = parser.parse_args()
    
    artifacts = create_test_artifacts(args.output_dir)
    print("\nTest artifacts created:")
    for k, v in artifacts.items():
        print(f"  {k}: {v}")
    
    print("\nTo run smoke test in real-artifact mode:")
    print(f"  export MODEL_PATH={artifacts['model_path']}")
    print(f"  export MODEL_META_PATH={artifacts['meta_path']}")
    print(f"  export HMM_PARAMS_PATH={artifacts['hmm_path']}")
    print(f"  export MOCK_MODE=false")
    print(f"  python scripts/run_api_smoke_test.py")


if __name__ == "__main__":
    main()