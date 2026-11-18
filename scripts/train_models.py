#!/usr/bin/env python
"""
CLI script for training and evaluating PathSnap models.

Usage:
    python -m scripts.train_models \
        --data-root "Student folder/Student folder" \
        --k 10 \
        --radius 100 \
        --seed 42 \
        --output-dir experiments/models \
        --n-splits 5 \
        --class-weight balanced \
        --max-train-samples 200000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.data.loader import load_partition
from src.data.splits import save_splits
from src.features.candidates import generate_candidates_for_partition
from src.features.engineer import build_features_for_partition
from src.models.train import (
    get_model_configs,
    compute_trip_ids_for_candidates,
    run_cross_validation,
    evaluate_on_student_test,
    save_model_comparison,
    save_selected_model,
)
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate PathSnap map-matching models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
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
        "--calibrate",
        action="store_true",
        default=True,
        help="Calibrate classifier probabilities",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_false",
        dest="calibrate",
        help="Disable probability calibration",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to train (default: all 5)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=200000,
        help="Maximum training samples (random subset with pos/neg balance). 0 = use all.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def subsample_training_data(
    X: np.ndarray,
    y: np.ndarray,
    candidate_mapping: List[Tuple[str, int, int]],
    max_samples: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, int, int]]]:
    """
    Subsample training data while preserving positive class and group structure.
    Keeps all positive samples and randomly samples negatives.
    """
    if max_samples <= 0 or max_samples >= len(X):
        return X, y, candidate_mapping
    
    np.random.seed(random_state)
    
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    n_pos = len(pos_indices)
    n_neg_keep = min(len(neg_indices), max_samples - n_pos)
    
    if n_neg_keep <= 0:
        n_neg_keep = max(1, max_samples - n_pos)
    
    neg_sampled = np.random.choice(neg_indices, size=n_neg_keep, replace=False)
    keep_indices = np.concatenate([pos_indices, neg_sampled])
    keep_indices = np.sort(keep_indices)
    
    X_sub = X[keep_indices]
    y_sub = y[keep_indices]
    mapping_sub = [candidate_mapping[i] for i in keep_indices]
    
    print(f"  Subsampled training: {len(X):,} -> {len(X_sub):,} "
          f"(pos: {n_pos:,}, neg: {n_neg_keep:,})")
    
    return X_sub, y_sub, mapping_sub


def main() -> int:
    args = parse_args()
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_weight = args.class_weight if args.class_weight != "none" else None
    
    print("=" * 60)
    print("PathSnap Model Zoo Training")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print(f"K={args.k}, Radius={args.radius}m")
    print(f"Seed: {args.seed}")
    print(f"Output dir: {output_dir}")
    print(f"CV splits: {args.n_splits}")
    print(f"Class weight: {class_weight}")
    print(f"Calibrate: {args.calibrate}")
    print(f"Max train samples: {args.max_train_samples:,}")
    print()
    
    start_total = time.time()
    
    # 1. Load training trips
    print("Loading training trips...")
    train_trips = load_partition(data_root / "train", validate=True)
    print(f"  Loaded {len(train_trips)} training trips")
    
    # 2. Load student test trips
    print("Loading student test trips...")
    test_trips = load_partition(data_root / "student_test", validate=True)
    print(f"  Loaded {len(test_trips)} student test trips")
    
    # 3. Generate candidates for training trips
    print(f"\nGenerating candidates for training trips (K={args.k}, R={args.radius}m)...")
    train_candidates = generate_candidates_for_partition(
        train_trips, k=args.k, radius_meters=args.radius, use_kdtree=True
    )
    
    # 4. Generate candidates for student test trips
    print(f"Generating candidates for student test trips...")
    test_candidates = generate_candidates_for_partition(
        test_trips, k=args.k, radius_meters=args.radius, use_kdtree=True
    )
    
    # 5. Build feature matrices
    print("\nBuilding feature matrices...")
    X_train, y_train, feature_names, train_mapping = build_features_for_partition(
        train_trips, train_candidates
    )
    X_test, y_test, _, test_mapping = build_features_for_partition(
        test_trips, test_candidates
    )
    
    print(f"  Train: X={X_train.shape}, y={y_train.shape}, pos={y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}, pos={y_test.sum()} ({y_test.mean()*100:.2f}%)")
    print(f"  Features: {feature_names}")
    
    # 6. Subsample training data if needed
    if args.max_train_samples > 0:
        X_train, y_train, train_mapping = subsample_training_data(
            X_train, y_train, train_mapping, args.max_train_samples, args.seed
        )
    
    # 7. Get per-candidate trip IDs for GroupKFold (as integers)
    train_trip_ids = compute_trip_ids_for_candidates(train_mapping)
    test_trip_ids = compute_trip_ids_for_candidates(test_mapping)
    
    # 8. Save splits for reproducibility
    from src.data.splits import create_group_aware_splits
    splits = create_group_aware_splits(
        [m[0] for m in train_mapping],  # trip_id per observation
        n_splits=args.n_splits,
        random_state=args.seed,
    )
    save_splits(splits, output_dir / "splits.json")
    print(f"\nSaved CV splits to {output_dir / 'splits.json'}")
    
    # 9. Get model configurations
    configs = get_model_configs(random_state=args.seed, class_weight=class_weight)
    
    if args.models:
        configs = {k: v for k, v in configs.items() if k in args.models}
        print(f"\nTraining selected models: {list(configs.keys())}")
    else:
        print(f"\nTraining all {len(configs)} models: {list(configs.keys())}")
    
    # 10. Run cross-validation for each model
    cv_results = []
    trained_models = {}
    
    for model_name, (model, model_family, config) in configs.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name} ({model_family})")
        print(f"Config: {config}")
        print(f"{'='*60}")
        
        try:
            cv_result = run_cross_validation(
                model=model,
                model_name=model_name,
                model_family=model_family,
                config=config,
                X=X_train,
                y=y_train,
                candidate_mapping=train_mapping,
                trip_ids=train_trip_ids,
                n_splits=args.n_splits,
                random_state=args.seed,
                calibrate=args.calibrate,
            )
            cv_results.append(cv_result)
            
            # Retrain on full training data for test evaluation
            print(f"  Retraining on full training set for student test evaluation...")
            full_model = clone(model)
            full_model.fit(X_train, y_train)
            # Skip calibration for tree-based models (they have built-in probabilities)
            tree_models = ("RandomForest", "HistGradientBoosting", "LightGBM", "XGBoost", "ExtraTrees")
            if args.calibrate and hasattr(full_model, "predict_proba") and model_name not in tree_models:
                try:
                    calibrated = CalibratedClassifierCV(full_model, method="isotonic", cv=3)
                    calibrated.fit(X_train, y_train)
                    full_model = calibrated
                except Exception:
                    pass
            trained_models[model_name] = full_model
            
            # Save intermediate results after each model
            save_model_comparison(cv_results, [], output_dir)
            
        except Exception as e:
            print(f"  ERROR training {model_name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    if not cv_results:
        print("\nERROR: No models trained successfully!")
        return 1
    
    # 11. Evaluate on student test set
    print(f"\n{'='*60}")
    print("Student Test Evaluation")
    print(f"{'='*60}")
    
    student_results = []
    for model_name, model in trained_models.items():
        print(f"\nEvaluating {model_name} on student test...")
        result = evaluate_on_student_test(
            model=model,
            model_name=model_name,
            X_test=X_test,
            y_test=y_test,
            test_mapping=test_mapping,
            test_trip_ids=test_trip_ids,
            batch_size=50000,
        )
        student_results.append(result)
        print(f"  macro_f1={result.macro_f1:.4f}, pos_prec={result.pos_precision:.4f}, "
              f"pos_rec={result.pos_recall:.4f}, logloss={result.logloss:.4f}, "
              f"top1_acc={result.top1_accuracy:.4f}")
        print(f"  recall@1={result.candidate_recall_at_1:.4f}, "
              f"recall@5={result.candidate_recall_at_5:.4f}, "
              f"recall@10={result.candidate_recall_at_10:.4f}")
        
        # Save intermediate results after each model evaluation
        save_model_comparison(cv_results, student_results, output_dir)
    
    # 12. Save results
    save_model_comparison(cv_results, student_results, output_dir)
    
    # 13. Select and save best model (by CV macro F1)
    best_cv = max(cv_results, key=lambda r: r.mean_macro_f1)
    best_model_name = best_cv.model_name
    print(f"\n{'='*60}")
    print(f"Best model (CV macro-F1): {best_model_name} ({best_cv.mean_macro_f1:.4f} +/- {best_cv.std_macro_f1:.4f})")
    print(f"{'='*60}")
    
    best_model = trained_models[best_model_name]
    save_selected_model(best_model, best_model_name, feature_names, output_dir / "selected")
    
    # 14. Print summary table
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'CV Macro-F1':>12} {'CV Top-1':>10} {'Test Macro-F1':>14} {'Test Top-1':>12} {'Recall@1':>10} {'Recall@5':>10} {'Recall@10':>10}")
    print("-"*80)
    for cv_r, st_r in zip(cv_results, student_results):
        print(f"{cv_r.model_name:<20} {cv_r.mean_macro_f1:>12.4f} {cv_r.mean_top1_accuracy:>10.4f} "
              f"{st_r.macro_f1:>14.4f} {st_r.top1_accuracy:>12.4f} "
              f"{st_r.candidate_recall_at_1:>10.4f} {st_r.candidate_recall_at_5:>10.4f} {st_r.candidate_recall_at_10:>10.4f}")
    
    total_time = time.time() - start_total
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())