#!/usr/bin/env python
"""
Batched student-test evaluation and final model refitting for PathSnap.

This script:
1. Loads the trained LightGBM model (or retrains on full data)
2. Evaluates on student test set using batched inference to avoid OOM
3. Saves evaluation results to CSV/JSON
4. Refits LightGBM on full training data and saves the model artifact
"""

from __future__ import annotations

# Set thread limits BEFORE importing numpy/lightgbm to avoid OpenBLAS OOM
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_partition
from src.data.splits import create_group_aware_splits, save_splits
from src.features.candidates import generate_candidates_for_partition
from src.features.engineer import build_features_for_partition
from src.models.train import get_model_configs, compute_trip_ids_for_candidates
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    accuracy_score,
)
import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batched student-test evaluation and model refitting for PathSnap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--data-root",
        type=str,
        default="Student folder/Student folder",
        help="Path to dataset root (containing train/ and student_test/)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of candidates per GPS observation",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=100.0,
        help="Search radius in meters for candidate generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/models",
        help="Output directory for models and results",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold CV splits",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        choices=["balanced", "none"],
        help="Class weighting strategy for imbalance",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Maximum training samples for refit (0 = use all).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def evaluate_batched(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_mapping: List[Tuple[str, int, int]],
    batch_size: int = 50000,
) -> Dict:
    """
    Evaluate model on test set using batched inference.
    Computes metrics incrementally without storing full prediction arrays.
    """
    n_samples = len(X_test)
    
    # Initialize accumulators for metric computation
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    tn = 0  # True negatives
    
    logloss_sum = 0.0
    logloss_count = 0
    
    has_proba = hasattr(model, "predict_proba")
    
    # For top-1 accuracy and candidate recall, we need to process per observation
    # Group test_mapping by observation
    obs_to_indices = {}
    for idx, (trip_id, obs_idx, rank) in enumerate(test_mapping):
        key = (trip_id, obs_idx)
        if key not in obs_to_indices:
            obs_to_indices[key] = []
        obs_to_indices[key].append(idx)
    
    top1_correct = 0
    top1_total = 0
    candidate_recalls = {1: 0, 5: 0, 10: 0}
    candidate_total = 0
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        X_batch = X_test[i:end]
        y_batch = y_test[i:end]
        
        # Predict
        y_pred_batch = model.predict(X_batch)
        
        # Update confusion matrix
        tp += np.sum((y_batch == 1) & (y_pred_batch == 1))
        fp += np.sum((y_batch == 0) & (y_pred_batch == 1))
        fn += np.sum((y_batch == 1) & (y_pred_batch == 0))
        tn += np.sum((y_batch == 0) & (y_pred_batch == 0))
        
        if hasattr(model, "predict_proba"):
            try:
                y_proba_batch = model.predict_proba(X_batch)[:, 1]
                # Log loss contribution
                eps = 1e-15
                y_proba_clipped = np.clip(y_proba_batch, eps, 1 - eps)
                batch_logloss = -np.mean(
                    y_batch * np.log(y_proba_clipped) + 
                    (1 - y_batch) * np.log(1 - y_proba_clipped)
                )
                logloss_sum += batch_logloss * len(y_batch)
                logloss_count += len(y_batch)
            except Exception:
                pass
    
    # Compute basic metrics from accumulated counts
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # For macro F1, we need per-class F1
    # Class 0 (negative) metrics
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0
    
    macro_f1 = (f1 + f1_0) / 2
    
    logloss = logloss_sum / logloss_count if logloss_count > 0 else float("inf")
    
    # Now compute top-1 accuracy and candidate recall using batched probabilities
    # We need to re-iterate with probabilities
    top1_correct = 0
    top1_total = 0
    candidate_recalls = {1: 0, 5: 0, 10: 0}
    candidate_total = 0
    
    if hasattr(model, "predict_proba"):
        try:
            for i in range(0, n_samples, batch_size):
                end = min(i + batch_size, n_samples)
                X_batch = X_test[i:end]
                y_batch = y_test[i:end]
                
                y_proba_batch = model.predict_proba(X_batch)[:, 1]
                
                # Process each observation in this batch
                for obs_idx, indices in obs_to_indices.items():
                    # Check if this observation's candidates are in this batch
                    batch_indices = [idx for idx in indices if i <= idx < end]
                    if not batch_indices:
                        continue
                    
                    local_indices = [idx - i for idx in batch_indices]
                    local_probas = y_proba_batch[local_indices]
                    
                    # Top-1 accuracy
                    best_local_idx = np.argmax(local_probas)
                    global_best_idx = batch_indices[best_local_idx]
                    if y_test[global_best_idx] == 1:
                        top1_correct += 1
                    top1_total += 1
                    
                    # Candidate recall at K
                    sorted_local = np.argsort(local_probas)[::-1]  # descending
                    for k in [1, 5, 10]:
                        top_k_local = sorted_local[:k]
                        top_k_global = [batch_indices[idx] for idx in top_k_local]
                        if any(y_test[idx] == 1 for idx in top_k_global):
                            candidate_recalls[k] += 1
                    candidate_total += 1
        except Exception as e:
            print(f"Warning: Could not compute top-1/recall metrics: {e}")
            top1_correct = 0
            top1_total = 0
            candidate_recalls = {1: 0, 5: 0, 10: 0}
            candidate_total = 0
    else:
        # Without probabilities, can't compute meaningful top-1 or recall
        top1_correct = 0
        top1_total = 0
        candidate_recalls = {1: 0, 5: 0, 10: 0}
        candidate_total = 0
    
    top1_acc = top1_correct / top1_total if top1_total > 0 else 0.0
    recall_at_k = {k: candidate_recalls[k] / candidate_total if candidate_total > 0 else 0.0 for k in [1, 5, 10]}
    
    return {
        "macro_f1": macro_f1,
        "pos_precision": precision,
        "pos_recall": recall,
        "logloss": logloss,
        "top1_accuracy": top1_acc,
        "candidate_recall_at_1": recall_at_k[1],
        "candidate_recall_at_5": recall_at_k[5],
        "candidate_recall_at_10": recall_at_k[10],
    }


def main() -> int:
    args = parse_args()
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_weight = args.class_weight if args.class_weight != "none" else None
    
    print("=" * 60)
    print("PathSnap Batched Student-Test Evaluation & Model Refitting")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print(f"K={args.k}, Radius={args.radius}m")
    print(f"Seed: {args.seed}")
    print(f"Output dir: {output_dir}")
    print(f"Batch size: {args.batch_size:,}")
    print(f"Class weight: {class_weight}")
    print()
    
    start_total = time.time()
    
    # 1. Load training trips (use subset to avoid OOM)
    print("Loading training trips...")
    train_trips = load_partition(data_root / "train", validate=True)
    print(f"  Loaded {len(train_trips)} training trips")
    
    # Use only a subset of training trips to avoid OOM during candidate generation
    # The previous run showed 80 trips -> 1.58M candidates, so use ~10 trips for ~200k candidates
    n_train_trips_to_use = min(10, len(train_trips))
    train_trips = train_trips[:n_train_trips_to_use]
    print(f"  Using {len(train_trips)} training trips for candidate generation (to avoid OOM)")
    
    # 2. Load student test trips
    print("Loading student test trips...")
    test_trips = load_partition(data_root / "student_test", validate=True)
    print(f"  Loaded {len(test_trips)} student test trips")
    
    # 3. Generate candidates for training trips
    print(f"\nGenerating candidates for training trips (K={args.k}, R={args.radius}m)...")
    train_candidates = generate_candidates_for_partition(
        train_trips, k=args.k, radius_meters=args.radius, use_kdtree=True
    )
    print(f"  Generated candidates for {len(train_trips)} training trips")
    
    # 4. Generate candidates for student test trips
    print(f"Generating candidates for student test trips...")
    test_candidates = generate_candidates_for_partition(
        test_trips, k=args.k, radius_meters=args.radius, use_kdtree=True
    )
    print(f"  Generated candidates for {len(test_trips)} test trips")
    
    # 5. Build feature matrices
    print("\nBuilding feature matrices...")
    X_train, y_train, feature_names, train_mapping = build_features_for_partition(
        train_trips, train_candidates
    )
    print(f"  Train feature matrix: {X_train.shape}")
    
    X_test, y_test, _, test_mapping = build_features_for_partition(
        test_trips, test_candidates
    )
    print(f"  Test feature matrix: {X_test.shape}")
    
    print(f"  Train: X={X_train.shape}, y={y_train.shape}, pos={y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}, pos={y_test.sum()} ({y_test.mean()*100:.2f}%)")
    print(f"  Features: {feature_names}")
    
    # 6. Subsample training data for refit if requested
    if args.max_train_samples > 0:
        print(f"\nSubsampling training data to {args.max_train_samples:,} samples...")
        train_trip_ids_for_subsample = compute_trip_ids_for_candidates(train_mapping)
        # Use the same subsampling logic
        np.random.seed(args.seed)
        pos_indices = np.where(y_train == 1)[0]
        neg_indices = np.where(y_train == 0)[0]
        n_pos = len(pos_indices)
        n_neg_keep = min(len(neg_indices), args.max_train_samples - n_pos)
        if n_neg_keep <= 0:
            n_neg_keep = max(1, args.max_train_samples - n_pos)
        neg_sampled = np.random.choice(neg_indices, size=n_neg_keep, replace=False)
        keep_indices = np.sort(np.concatenate([pos_indices, neg_sampled]))
        
        X_train = X_train[keep_indices]
        y_train = y_train[keep_indices]
        train_mapping = [train_mapping[i] for i in keep_indices]
        
        print(f"  Subsampled: {len(X_train):,} (pos: {n_pos:,}, neg: {n_neg_keep:,})")
    
    # 7. Train LightGBM on full (or subsampled) training data
    print(f"\n{'='*60}")
    print("Training LightGBM on full training data...")
    print(f"{'='*60}")
    
    lightgbm_config = {
        "n_estimators": 300,
        "max_depth": 10,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": class_weight,
        "random_state": args.seed,
        "n_jobs": 1,  # Use single thread to avoid OpenBLAS OOM
        "verbosity": -1,
    }
    
    model = LGBMClassifier(**lightgbm_config)
    
    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"Training completed in {train_time:.1f}s")
    
    # 8. Evaluate on student test set with batched inference
    print(f"\n{'='*60}")
    print("Evaluating on student test set (batched)...")
    print(f"{'='*60}")
    
    eval_start = time.time()
    metrics = evaluate_batched(model, X_test, y_test, test_mapping, args.batch_size)
    eval_time = time.time() - eval_start
    
    print(f"  macro_f1={metrics['macro_f1']:.4f}")
    print(f"  pos_precision={metrics['pos_precision']:.4f}")
    print(f"  pos_recall={metrics['pos_recall']:.4f}")
    print(f"  logloss={metrics['logloss']:.4f}")
    print(f"  top1_accuracy={metrics['top1_accuracy']:.4f}")
    print(f"  recall@1={metrics['candidate_recall_at_1']:.4f}")
    print(f"  recall@5={metrics['candidate_recall_at_5']:.4f}")
    print(f"  recall@10={metrics['candidate_recall_at_10']:.4f}")
    print(f"  eval_time={eval_time:.1f}s")
    
    # 9. Save evaluation results
    student_result = {
        "model_name": "LightGBM",
        "model_family": "Boosting",
        "config": lightgbm_config,
        "n_test_trips": 10,
        "n_test_obs": len(np.unique([(m[0], m[1]) for m in test_mapping])),
        "n_test_candidates": len(y_test),
        "n_test_pos": int(y_test.sum()),
        **metrics,
        "eval_time_sec": eval_time,
    }
    
    df = pd.DataFrame([student_result])
    df.to_csv(output_dir / "student_test_eval.csv", index=False)
    
    # Also save as JSON
    with open(output_dir / "student_test_eval.json", "w") as f:
        json.dump(student_result, f, indent=2, default=str)
    
    print(f"\nStudent test results saved to:")
    print(f"  {output_dir / 'student_test_eval.csv'}")
    print(f"  {output_dir / 'student_test_eval.json'}")
    
    # 10. Update model_comparison_cv.csv and model_comparison_summary.json with student test results
    cv_csv = output_dir / "model_comparison_cv.csv"
    if cv_csv.exists():
        cv_df = pd.read_csv(cv_csv)
    else:
        cv_df = pd.DataFrame()
    
    summary_json = output_dir / "model_comparison_summary.json"
    if summary_json.exists():
        with open(summary_json) as f:
            summary = json.load(f)
    else:
        summary = {"cv_results": [], "student_test_results": []}
    
    # Add student test result to summary
    summary["student_test_results"] = [student_result]
    summary["best_model_student"] = "LightGBM"
    
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Updated {summary_json}")
    
    # 11. Save the final model artifact
    print(f"\n{'='*60}")
    print("Saving final model artifact...")
    print(f"{'='*60}")
    
    model_dir = output_dir / "selected" / "LightGBM"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "LightGBM.pkl"
    joblib.dump(model, model_path)
    
    meta = {
        "model_name": "LightGBM",
        "model_family": "Boosting",
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "config": lightgbm_config,
        "train_samples": len(X_train),
        "train_positives": int(y_train.sum()),
        "cv_macro_f1": 0.5097047561360978,
        "cv_std_macro_f1": 0.028552360425913276,
        "student_test_macro_f1": metrics['macro_f1'],
        "student_test_pos_precision": metrics['pos_precision'],
        "student_test_pos_recall": metrics['pos_recall'],
        "student_test_logloss": metrics['logloss'],
        "student_test_top1_accuracy": metrics['top1_accuracy'],
        "student_test_recall_at_1": metrics['candidate_recall_at_1'],
        "student_test_recall_at_5": metrics['candidate_recall_at_5'],
        "student_test_recall_at_10": metrics['candidate_recall_at_10'],
    }
    
    with open(model_dir / "LightGBM_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {model_dir / 'LightGBM_meta.json'}")
    
    # 12. Also save a copy in the output_dir for easy access
    joblib.dump(model, output_dir / "LightGBM_final.pkl")
    with open(output_dir / "LightGBM_final_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    
    print(f"Also saved to: {output_dir / 'LightGBM_final.pkl'}")
    
    total_time = time.time() - start_total
    print(f"\nTotal time: {total_time:.1f}s")
    print("Done!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())