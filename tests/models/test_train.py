"""Tests for model training and evaluation."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.models.train import (
    get_model_configs,
    compute_trip_ids_for_candidates,
    evaluate_top1_accuracy_by_obs,
    evaluate_candidate_recall,
    run_cross_validation,
)


def test_get_model_configs():
    """Test that model configs are returned correctly."""
    configs = get_model_configs(random_state=42, class_weight="balanced")
    
    # Should have at least 5 models
    assert len(configs) >= 5
    
    # Check required models exist
    required_models = ["LogisticRegression", "RandomForest", "HistGradientBoosting", "MLP"]
    for model in required_models:
        assert model in configs, f"Missing required model: {model}"
    
    # Check structure
    for name, (model, family, config) in configs.items():
        assert isinstance(family, str)
        assert isinstance(config, dict)
        assert model is not None


def test_compute_trip_ids_for_candidates():
    """Test trip ID extraction from candidate mapping."""
    mapping = [
        ("00000001", 0, 0),
        ("00000001", 0, 1),
        ("00000001", 1, 0),
        ("00000002", 0, 0),
        ("00000002", 0, 1),
    ]
    
    trip_ids = compute_trip_ids_for_candidates(mapping)
    
    assert len(trip_ids) == 5
    # Returns integer group IDs (0, 1, ...) for GroupKFold efficiency
    assert list(trip_ids) == [0, 0, 0, 1, 1]


def test_evaluate_top1_accuracy_by_obs():
    """Test Top-1 accuracy computation by observation."""
    # Mock model that returns fixed probabilities
    class MockModel:
        def predict(self, X):
            return np.zeros(len(X))
        
        def predict_proba(self, X):
            # Return probabilities favoring first candidate per obs
            n = len(X)
            proba = np.zeros((n, 2))
            for i in range(n):
                if i % 3 == 0:  # Every 3rd candidate (rank 0) gets high prob
                    proba[i, 1] = 0.9
                else:
                    proba[i, 1] = 0.1
            proba[:, 0] = 1 - proba[:, 1]
            return proba
    
    model = MockModel()
    
    # 3 observations, 3 candidates each = 9 rows
    # Ground truth: rank 0 is correct for each obs
    y = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0])
    X = np.zeros((9, 5))
    
    mapping = [
        ("trip1", 0, 0), ("trip1", 0, 1), ("trip1", 0, 2),
        ("trip1", 1, 0), ("trip1", 1, 1), ("trip1", 1, 2),
        ("trip1", 2, 0), ("trip1", 2, 1), ("trip1", 2, 2),
    ]
    
    # With mock model favoring rank 0, all 3 should be correct
    acc = evaluate_top1_accuracy_by_obs(model, X, y, mapping)
    assert acc == 1.0
    
    # Test with all wrong predictions
    class MockModelWrong:
        def predict(self, X):
            return np.zeros(len(X))
        
        def predict_proba(self, X):
            n = len(X)
            proba = np.zeros((n, 2))
            for i in range(n):
                if i % 3 == 2:  # Rank 2 gets high prob (wrong)
                    proba[i, 1] = 0.9
                else:
                    proba[i, 1] = 0.1
            proba[:, 0] = 1 - proba[:, 1]
            return proba
    
    model_wrong = MockModelWrong()
    acc = evaluate_top1_accuracy_by_obs(model_wrong, X, y, mapping)
    assert acc == 0.0


def test_evaluate_candidate_recall():
    """Test candidate recall at K."""
    class MockModel:
        def predict_proba(self, X):
            n = len(X)
            proba = np.zeros((n, 2))
            for i in range(n):
                if i % 3 == 0:
                    proba[i, 1] = 0.9
                else:
                    proba[i, 1] = 0.1
            proba[:, 0] = 1 - proba[:, 1]
            return proba
    
    model = MockModel()
    
    y = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0])
    X = np.zeros((9, 5))
    
    mapping = [
        ("trip1", 0, 0), ("trip1", 0, 1), ("trip1", 0, 2),
        ("trip1", 1, 0), ("trip1", 1, 1), ("trip1", 1, 2),
        ("trip1", 2, 0), ("trip1", 2, 1), ("trip1", 2, 2),
    ]
    
    recalls = evaluate_candidate_recall(model, X, y, mapping, k_values=[1, 2, 3])
    
    # Ground truth is at rank 0, model ranks it first -> recall@1 = 1.0
    assert recalls[1] == 1.0
    assert recalls[2] == 1.0
    assert recalls[3] == 1.0


def test_run_cross_validation_group_separation():
    """Test that GroupKFold properly separates trips."""
    # Create synthetic data with clear trip structure
    np.random.seed(42)
    
    # 4 trips, 10 obs each, 3 candidates per obs = 120 rows
    n_trips = 4
    n_obs_per_trip = 10
    n_cands_per_obs = 3
    
    X = np.random.randn(n_trips * n_obs_per_trip * n_cands_per_obs, 5)
    y = np.zeros(n_trips * n_obs_per_trip * n_cands_per_obs)
    
    # Make first candidate of each obs positive (sparse)
    for t in range(n_trips):
        for o in range(n_obs_per_trip):
            idx = (t * n_obs_per_trip + o) * n_cands_per_obs
            y[idx] = 1
    
    mapping = []
    for t in range(n_trips):
        for o in range(n_obs_per_trip):
            for c in range(n_cands_per_obs):
                mapping.append((f"trip{t:03d}", o, c))
    
    trip_ids = compute_trip_ids_for_candidates(mapping)
    
    # Run CV with 2 splits
    model = LogisticRegression(random_state=42, class_weight="balanced", max_iter=500)
    
    result = run_cross_validation(
        model=model,
        model_name="TestLogReg",
        model_family="Linear",
        config={"C": 1.0},
        X=X,
        y=y,
        candidate_mapping=mapping,
        trip_ids=trip_ids,
        n_splits=2,
        random_state=42,
        calibrate=False,
    )
    
    # Verify no leakage: each fold should have disjoint trip IDs
    for fold in result.folds:
        train_set = set(fold.train_trips)
        val_set = set(fold.val_trips)
        assert len(train_set & val_set) == 0, f"Leakage detected in fold {fold.fold}"
    
    # Should have 2 folds
    assert len(result.folds) == 2
    
    # Metrics should be computed
    assert result.mean_macro_f1 >= 0
    assert result.mean_pos_precision >= 0
    assert result.mean_pos_recall >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])