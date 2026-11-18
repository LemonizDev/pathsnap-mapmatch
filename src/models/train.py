"""
Model training and evaluation for PathSnap map-matching.

Trains and compares multiple classifiers with GroupKFold CV to prevent trajectory leakage.
Handles severe class imbalance via class_weight and sampling strategies.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.data.loader import TripData, load_partition
from src.data.splits import create_group_aware_splits, get_trip_ids_from_trips, save_splits
from src.features.candidates import generate_candidates_for_partition
from src.features.engineer import build_features_for_partition


@dataclass
class FoldMetrics:
    """Metrics for a single CV fold."""
    fold: int
    train_trips: List[int]
    val_trips: List[int]
    n_train: int
    n_val: int
    n_train_pos: int
    n_val_pos: int
    macro_f1: float
    pos_precision: float
    pos_recall: float
    logloss: float
    top1_accuracy: float
    train_time_sec: float
    val_time_sec: float


@dataclass
class ModelCVResult:
    """Cross-validation results for a model."""
    model_name: str
    model_family: str
    config: Dict[str, Any]
    folds: List[FoldMetrics]
    mean_macro_f1: float
    std_macro_f1: float
    mean_pos_precision: float
    mean_pos_recall: float
    mean_logloss: float
    mean_top1_accuracy: float
    total_train_time_sec: float


@dataclass
class StudentTestResult:
    """Evaluation results on student test set."""
    model_name: str
    n_test_trips: int
    n_test_obs: int
    n_test_candidates: int
    n_test_pos: int
    macro_f1: float
    pos_precision: float
    pos_recall: float
    logloss: float
    top1_accuracy: float
    candidate_recall_at_1: float
    candidate_recall_at_5: float
    candidate_recall_at_10: float
    eval_time_sec: float


def get_model_configs(random_state: int = 42, class_weight: Optional[str] = "balanced") -> Dict[str, Tuple[Any, str, Dict]]:
    """
    Get model configurations for the model zoo.
    
    Returns:
        Dict mapping model_name -> (model_instance, model_family, config_dict)
    """
    configs = {}
    
    # 1. Logistic Regression (linear baseline)
    configs["LogisticRegression"] = (
        LogisticRegression(
            random_state=random_state,
            class_weight=class_weight,
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            n_jobs=-1,
        ),
        "Linear",
        {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "class_weight": class_weight}
    )
    
    # 2. Random Forest (tree ensemble)
    configs["RandomForest"] = (
        RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Tree Ensemble",
        {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5, 
         "min_samples_leaf": 2, "class_weight": class_weight}
    )
    
    # 3. HistGradientBoosting (fast native boosting)
    configs["HistGradientBoosting"] = (
        HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=10,
            learning_rate=0.1,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=random_state,
            class_weight=class_weight,
        ),
        "Boosting",
        {"max_iter": 200, "max_depth": 10, "learning_rate": 0.1,
         "min_samples_leaf": 20, "l2_regularization": 1.0, "class_weight": class_weight}
    )
    
    # 4. LightGBM or XGBoost (strong boosting)
    if LIGHTGBM_AVAILABLE:
        configs["LightGBM"] = (
            LGBMClassifier(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            ),
            "Boosting",
            {"n_estimators": 300, "max_depth": 10, "learning_rate": 0.05,
             "num_leaves": 63, "min_child_samples": 20, "subsample": 0.8,
             "colsample_bytree": 0.8, "class_weight": class_weight}
        )
    elif XGBOOST_AVAILABLE:
        configs["XGBoost"] = (
            XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
                eval_metric="logloss",
            ),
            "Boosting",
            {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.05,
             "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5}
        )
    else:
        # Fallback: ExtraTrees if neither boosting lib available
        from sklearn.ensemble import ExtraTreesClassifier
        configs["ExtraTrees"] = (
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=-1,
            ),
            "Tree Ensemble",
            {"n_estimators": 300, "max_depth": 15, "min_samples_split": 5,
             "min_samples_leaf": 2, "class_weight": class_weight}
        )
    
    # 5. MLP (shallow neural net)
    configs["MLP"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=random_state,
            ))
        ]),
        "Neural Network",
        {"hidden_layer_sizes": (128, 64), "activation": "relu", "solver": "adam",
         "alpha": 1e-3, "learning_rate_init": 1e-3, "max_iter": 200,
         "early_stopping": True, "validation_fraction": 0.1}
    )
    
    return configs


def compute_trip_ids_for_candidates(
    candidate_mapping: List[Tuple[str, int, int]],
) -> np.ndarray:
    """
    Extract per-candidate trip IDs from mapping for GroupKFold.
    
    Args:
        candidate_mapping: List of (trip_id, obs_idx, candidate_rank)
        
    Returns:
        Array of integer group IDs for each candidate row (for GroupKFold)
    """
    trip_id_strs = [m[0] for m in candidate_mapping]
    # Map string trip IDs to integers for memory efficiency
    unique_trips = sorted(set(trip_id_strs))
    trip_to_int = {t: i for i, t in enumerate(unique_trips)}
    return np.array([trip_to_int[t] for t in trip_id_strs], dtype=np.int32)


def train_model_on_fold(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    calibrate: bool = True,
    model_name: str = "",
) -> Tuple[Any, Dict[str, float], float, float]:
    """
    Train a model on a single fold and compute metrics.
    
    Returns:
        (trained_model, metrics_dict, train_time, val_time)
    """
    model = clone(model)
    
    # Train
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    
    # Calibrate probabilities if requested and supported
    # Skip calibration for tree-based models (they have built-in probabilities)
    tree_models = ("RandomForest", "HistGradientBoosting", "LightGBM", "XGBoost", "ExtraTrees")
    should_calibrate = calibrate and hasattr(model, "predict_proba") and model_name not in tree_models
    
    if should_calibrate:
        try:
            calibrated = CalibratedClassifierCV(model, method="isotonic", cv=3)
            calibrated.fit(X_train, y_train)
            model = calibrated
        except Exception:
            pass  # Keep uncalibrated if calibration fails
    
    # Validate
    start_val = time.time()
    y_pred = model.predict(X_val)
    val_time = time.time() - start_val
    
    # Compute metrics
    metrics = {}
    
    # Macro F1
    metrics["macro_f1"] = f1_score(y_val, y_pred, average="macro", zero_division=0)
    
    # Positive class precision/recall
    metrics["pos_precision"] = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
    metrics["pos_recall"] = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
    
    # Log loss (if probabilities available)
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_val)
            if y_proba.shape[1] == 2:
                metrics["logloss"] = log_loss(y_val, y_proba[:, 1])
            else:
                metrics["logloss"] = log_loss(y_val, y_proba)
        except Exception:
            metrics["logloss"] = float("inf")
    else:
        metrics["logloss"] = float("inf")
    
    # Top-1 Accuracy (per observation: is top-ranked candidate correct?)
    # This requires grouping by observation - we approximate with overall accuracy
    metrics["top1_accuracy"] = accuracy_score(y_val, y_pred)
    
    return model, metrics, train_time, val_time


def evaluate_top1_accuracy_by_obs(
    model,
    X: np.ndarray,
    y: np.ndarray,
    candidate_mapping: List[Tuple[str, int, int]],
) -> float:
    """
    Compute true Top-1 Accuracy: for each observation, is the highest-probability
    candidate the ground truth?
    """
    if not hasattr(model, "predict_proba"):
        return accuracy_score(y, model.predict(X))
    
    try:
        y_proba = model.predict_proba(X)[:, 1]  # Probability of positive class
    except Exception:
        return accuracy_score(y, model.predict(X))
    
    # Group by observation (trip_id, obs_idx)
    obs_to_indices = {}
    for idx, (trip_id, obs_idx, rank) in enumerate(candidate_mapping):
        key = (trip_id, obs_idx)
        if key not in obs_to_indices:
            obs_to_indices[key] = []
        obs_to_indices[key].append(idx)
    
    correct = 0
    total = 0
    
    for indices in obs_to_indices.values():
        # Find candidate with highest probability
        best_idx = max(indices, key=lambda i: y_proba[i])
        if y[best_idx] == 1:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def evaluate_candidate_recall(
    model,
    X: np.ndarray,
    y: np.ndarray,
    candidate_mapping: List[Tuple[str, int, int]],
    k_values: List[int] = [1, 5, 10],
) -> Dict[int, float]:
    """
    Compute candidate recall at K: fraction of observations where ground truth
    is in top-K ranked candidates by model probability.
    """
    if not hasattr(model, "predict_proba"):
        return {k: 0.0 for k in k_values}
    
    try:
        y_proba = model.predict_proba(X)[:, 1]
    except Exception:
        return {k: 0.0 for k in k_values}
    
    # Group by observation
    obs_to_indices = {}
    for idx, (trip_id, obs_idx, rank) in enumerate(candidate_mapping):
        key = (trip_id, obs_idx)
        if key not in obs_to_indices:
            obs_to_indices[key] = []
        obs_to_indices[key].append(idx)
    
    recalls = {k: 0 for k in k_values}
    total = 0
    
    for indices in obs_to_indices.values():
        # Rank candidates by probability (descending)
        sorted_indices = sorted(indices, key=lambda i: y_proba[i], reverse=True)
        
        # Check if any positive label in top-k
        for k in k_values:
            top_k = sorted_indices[:k]
            if any(y[i] == 1 for i in top_k):
                recalls[k] += 1
        total += 1
    
    return {k: recalls[k] / total if total > 0 else 0.0 for k in k_values}


def run_cross_validation(
    model,
    model_name: str,
    model_family: str,
    config: Dict,
    X: np.ndarray,
    y: np.ndarray,
    candidate_mapping: List[Tuple[str, int, int]],
    trip_ids: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    calibrate: bool = True,
) -> ModelCVResult:
    """
    Run GroupKFold cross-validation for a single model.
    """
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    total_train_time = 0.0
    
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=trip_ids)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Get trip IDs for this fold
        train_trip_ids = np.unique(trip_ids[train_idx]).tolist()
        val_trip_ids = np.unique(trip_ids[val_idx]).tolist()
        
        # Train and evaluate
        trained_model, metrics, train_time, val_time = train_model_on_fold(
            model, X_train, y_train, X_val, y_val, calibrate, model_name
        )
        total_train_time += train_time
        
        # Top-1 accuracy by observation
        val_mapping = [candidate_mapping[i] for i in val_idx]
        top1_acc = evaluate_top1_accuracy_by_obs(trained_model, X_val, y_val, val_mapping)
        
        fold_metrics = FoldMetrics(
            fold=fold_idx,
            train_trips=train_trip_ids,
            val_trips=val_trip_ids,
            n_train=len(train_idx),
            n_val=len(val_idx),
            n_train_pos=int(y_train.sum()),
            n_val_pos=int(y_val.sum()),
            macro_f1=metrics["macro_f1"],
            pos_precision=metrics["pos_precision"],
            pos_recall=metrics["pos_recall"],
            logloss=metrics["logloss"],
            top1_accuracy=top1_acc,
            train_time_sec=train_time,
            val_time_sec=val_time,
        )
        folds.append(fold_metrics)
        
        print(f"  Fold {fold_idx}: macro_f1={metrics['macro_f1']:.4f}, "
              f"pos_prec={metrics['pos_precision']:.4f}, "
              f"pos_rec={metrics['pos_recall']:.4f}, "
              f"logloss={metrics['logloss']:.4f}, "
              f"top1_acc={top1_acc:.4f}")
    
    # Aggregate
    macro_f1s = [f.macro_f1 for f in folds]
    
    return ModelCVResult(
        model_name=model_name,
        model_family=model_family,
        config=config,
        folds=folds,
        mean_macro_f1=np.mean(macro_f1s),
        std_macro_f1=np.std(macro_f1s),
        mean_pos_precision=np.mean([f.pos_precision for f in folds]),
        mean_pos_recall=np.mean([f.pos_recall for f in folds]),
        mean_logloss=np.mean([f.logloss for f in folds]),
        mean_top1_accuracy=np.mean([f.top1_accuracy for f in folds]),
        total_train_time_sec=total_train_time,
    )


def evaluate_on_student_test(
    model,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_mapping: List[Tuple[str, int, int]],
    test_trip_ids: np.ndarray,
    batch_size: int = 50000,
) -> StudentTestResult:
    """
    Evaluate trained model on student test set with batch processing to avoid memory issues.
    """
    start = time.time()
    n_samples = len(X_test)
    
    # Process in batches
    y_pred_all = np.zeros(n_samples, dtype=np.int64)
    y_proba_all = None
    
    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        y_proba_all = np.zeros(n_samples, dtype=np.float64)
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        X_batch = X_test[i:end]
        
        y_pred_all[i:end] = model.predict(X_batch)
        
        if has_proba:
            try:
                y_proba_all[i:end] = model.predict_proba(X_batch)[:, 1]
            except Exception:
                has_proba = False
                y_proba_all = None
                break
    
    # Basic metrics
    macro_f1 = f1_score(y_test, y_pred_all, average="macro", zero_division=0)
    pos_precision = precision_score(y_test, y_pred_all, pos_label=1, zero_division=0)
    pos_recall = recall_score(y_test, y_pred_all, pos_label=1, zero_division=0)
    
    if has_proba and y_proba_all is not None:
        try:
            logloss = log_loss(y_test, y_proba_all)
        except Exception:
            logloss = float("inf")
    else:
        logloss = float("inf")
    
    # Top-1 accuracy by observation (needs probabilities)
    if has_proba and y_proba_all is not None:
        top1_acc = evaluate_top1_accuracy_by_obs_with_proba(y_proba_all, y_test, test_mapping)
        candidate_recalls = evaluate_candidate_recall_with_proba(y_proba_all, y_test, test_mapping)
    else:
        top1_acc = accuracy_score(y_test, y_pred_all)
        candidate_recalls = {1: 0.0, 5: 0.0, 10: 0.0}
    
    eval_time = time.time() - start
    
    return StudentTestResult(
        model_name=model_name,
        n_test_trips=len(np.unique(test_trip_ids)),
        n_test_obs=len(np.unique([(m[0], m[1]) for m in test_mapping])),
        n_test_candidates=len(y_test),
        n_test_pos=int(y_test.sum()),
        macro_f1=macro_f1,
        pos_precision=pos_precision,
        pos_recall=pos_recall,
        logloss=logloss,
        top1_accuracy=top1_acc,
        candidate_recall_at_1=candidate_recalls.get(1, 0.0),
        candidate_recall_at_5=candidate_recalls.get(5, 0.0),
        candidate_recall_at_10=candidate_recalls.get(10, 0.0),
        eval_time_sec=eval_time,
    )


def evaluate_top1_accuracy_by_obs_with_proba(
    y_proba: np.ndarray,
    y: np.ndarray,
    candidate_mapping: List[Tuple[str, int, int]],
) -> float:
    """
    Compute Top-1 Accuracy using pre-computed probabilities.
    """
    # Group by observation (trip_id, obs_idx)
    obs_to_indices = {}
    for idx, (trip_id, obs_idx, rank) in enumerate(candidate_mapping):
        key = (trip_id, obs_idx)
        if key not in obs_to_indices:
            obs_to_indices[key] = []
        obs_to_indices[key].append(idx)
    
    correct = 0
    total = 0
    
    for indices in obs_to_indices.values():
        # Find candidate with highest probability
        best_idx = max(indices, key=lambda i: y_proba[i])
        if y[best_idx] == 1:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def evaluate_candidate_recall_with_proba(
    y_proba: np.ndarray,
    y: np.ndarray,
    candidate_mapping: List[Tuple[str, int, int]],
    k_values: List[int] = [1, 5, 10],
) -> Dict[int, float]:
    """
    Compute candidate recall at K using pre-computed probabilities.
    """
    # Group by observation
    obs_to_indices = {}
    for idx, (trip_id, obs_idx, rank) in enumerate(candidate_mapping):
        key = (trip_id, obs_idx)
        if key not in obs_to_indices:
            obs_to_indices[key] = []
        obs_to_indices[key].append(idx)
    
    recalls = {k: 0 for k in k_values}
    total = 0
    
    for indices in obs_to_indices.values():
        # Rank candidates by probability (descending)
        sorted_indices = sorted(indices, key=lambda i: y_proba[i], reverse=True)
        
        # Check if any positive label in top-k
        for k in k_values:
            top_k = sorted_indices[:k]
            if any(y[i] == 1 for i in top_k):
                recalls[k] += 1
        total += 1
    
    return {k: recalls[k] / total if total > 0 else 0.0 for k in k_values}


def save_model_comparison(
    cv_results: List[ModelCVResult],
    student_results: List[StudentTestResult],
    output_dir: Path,
) -> None:
    """Save model comparison results to CSV/JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CV results summary
    cv_rows = []
    for r in cv_results:
        cv_rows.append({
            "model_name": r.model_name,
            "model_family": r.model_family,
            "config": json.dumps(r.config),
            "mean_macro_f1": r.mean_macro_f1,
            "std_macro_f1": r.std_macro_f1,
            "mean_pos_precision": r.mean_pos_precision,
            "mean_pos_recall": r.mean_pos_recall,
            "mean_logloss": r.mean_logloss,
            "mean_top1_accuracy": r.mean_top1_accuracy,
            "total_train_time_sec": r.total_train_time_sec,
        })
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(output_dir / "model_comparison_cv.csv", index=False)
    
    # Per-fold details
    fold_rows = []
    for r in cv_results:
        for f in r.folds:
            fold_rows.append({
                "model_name": r.model_name,
                "fold": f.fold,
                "train_trips": "|".join(str(t) for t in f.train_trips),
                "val_trips": "|".join(str(t) for t in f.val_trips),
                "n_train": f.n_train,
                "n_val": f.n_val,
                "n_train_pos": f.n_train_pos,
                "n_val_pos": f.n_val_pos,
                "macro_f1": f.macro_f1,
                "pos_precision": f.pos_precision,
                "pos_recall": f.pos_recall,
                "logloss": f.logloss,
                "top1_accuracy": f.top1_accuracy,
                "train_time_sec": f.train_time_sec,
                "val_time_sec": f.val_time_sec,
            })
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "model_comparison_folds.csv", index=False)
    
    # Student test results
    student_rows = []
    for r in student_results:
        student_rows.append(asdict(r))
    student_df = pd.DataFrame(student_rows)
    student_df.to_csv(output_dir / "student_test_eval.csv", index=False)
    
    # JSON summary for easy reading
    summary = {
        "cv_results": [asdict(r) for r in cv_results],
        "student_test_results": [asdict(r) for r in student_results],
        "best_model_cv": max(cv_results, key=lambda r: r.mean_macro_f1).model_name,
        "best_model_student": max(student_results, key=lambda r: r.macro_f1).model_name if student_results else None,
    }
    with open(output_dir / "model_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}")
    print(f"  model_comparison_cv.csv")
    print(f"  model_comparison_folds.csv")
    print(f"  student_test_eval.csv")
    print(f"  model_comparison_summary.json")


def save_selected_model(
    model,
    model_name: str,
    feature_names: List[str],
    output_dir: Path,
) -> None:
    """Save the selected model and metadata."""
    import joblib
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    
    meta = {
        "model_name": model_name,
        "feature_names": feature_names,
        "n_features": len(feature_names),
    }
    with open(output_dir / f"{model_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    print("Model training module loaded successfully")
    configs = get_model_configs()
    print(f"Available models: {list(configs.keys())}")
    print(f"LightGBM: {LIGHTGBM_AVAILABLE}, XGBoost: {XGBOOST_AVAILABLE}")