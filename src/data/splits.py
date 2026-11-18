"""
Group-aware train/validation split utilities for PathSnap.

Prevents trajectory leakage by splitting at the trip level (group = trip_id).
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.model_selection import GroupKFold


@dataclass
class SplitIndices:
    """Container for train/validation split indices."""
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_trip_ids: List[str]
    val_trip_ids: List[str]


def create_group_aware_splits(
    trip_ids: List[str],
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True
) -> List[SplitIndices]:
    """
    Create group-aware cross-validation splits (GroupKFold).
    
    Each split ensures all observations from a trip are in either train or validation,
    never both. This prevents trajectory leakage.
    
    Args:
        trip_ids: List of trip IDs for each observation (length = total observations)
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle before splitting
        
    Returns:
        List of SplitIndices for each fold
    """
    trip_ids = np.asarray(trip_ids)
    n_samples = len(trip_ids)
    
    # Dummy X and y for GroupKFold (we only need the group structure)
    X_dummy = np.zeros(n_samples)
    y_dummy = np.zeros(n_samples)
    
    gkf = GroupKFold(n_splits=n_splits)
    
    splits = []
    for train_idx, val_idx in gkf.split(X_dummy, y_dummy, groups=trip_ids):
        train_trip_ids = np.unique(trip_ids[train_idx]).tolist()
        val_trip_ids = np.unique(trip_ids[val_idx]).tolist()
        
        splits.append(SplitIndices(
            train_indices=train_idx,
            val_indices=val_idx,
            train_trip_ids=train_trip_ids,
            val_trip_ids=val_trip_ids
        ))
    
    return splits


def create_holdout_split(
    trip_ids: List[str],
    holdout_trip_ids: List[str]
) -> SplitIndices:
    """
    Create a single train/validation split with specified holdout trips.
    
    Args:
        trip_ids: List of trip IDs for each observation
        holdout_trip_ids: Trip IDs to use as validation set
        
    Returns:
        SplitIndices with holdout trips as validation
    """
    trip_ids = np.asarray(trip_ids)
    holdout_set = set(holdout_trip_ids)
    
    val_mask = np.isin(trip_ids, list(holdout_set))
    train_mask = ~val_mask
    
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    
    train_trip_ids = np.unique(trip_ids[train_idx]).tolist()
    val_trip_ids = np.unique(trip_ids[val_idx]).tolist()
    
    return SplitIndices(
        train_indices=train_idx,
        val_indices=val_idx,
        train_trip_ids=train_trip_ids,
        val_trip_ids=val_trip_ids
    )


def save_splits(splits: List[SplitIndices], filepath: Path) -> None:
    """Save splits to JSON file."""
    data = {
        "n_splits": len(splits),
        "splits": [
            {
                "fold": i,
                "train_indices": split.train_indices.tolist(),
                "val_indices": split.val_indices.tolist(),
                "train_trip_ids": split.train_trip_ids,
                "val_trip_ids": split.val_trip_ids,
                "n_train": len(split.train_indices),
                "n_val": len(split.val_indices),
                "n_train_trips": len(split.train_trip_ids),
                "n_val_trips": len(split.val_trip_ids),
            }
            for i, split in enumerate(splits)
        ]
    }
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_splits(filepath: Path) -> List[SplitIndices]:
    """Load splits from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    
    splits = []
    for split_data in data["splits"]:
        splits.append(SplitIndices(
            train_indices=np.array(split_data["train_indices"], dtype=np.int64),
            val_indices=np.array(split_data["val_indices"], dtype=np.int64),
            train_trip_ids=split_data["train_trip_ids"],
            val_trip_ids=split_data["val_trip_ids"],
        ))
    return splits


def verify_no_leakage(splits: List[SplitIndices], trip_ids: List[str]) -> bool:
    """
    Verify that no trip appears in both train and validation in any split.
    
    Returns True if no leakage detected.
    """
    trip_ids = np.asarray(trip_ids)
    for i, split in enumerate(splits):
        train_trips = set(split.train_trip_ids)
        val_trips = set(split.val_trip_ids)
        overlap = train_trips & val_trips
        if overlap:
            print(f"LEAKAGE DETECTED in fold {i}: trips {overlap} appear in both train and val")
            return False
    print("No leakage detected across all splits")
    return True


def get_trip_ids_from_trips(trips: List) -> Tuple[np.ndarray, List[str]]:
    """
    Extract per-observation trip IDs from a list of TripData objects.
    
    Returns:
        trip_ids_array: Array of trip_id for each observation (for GroupKFold)
        unique_trip_ids: List of unique trip IDs in order
    """
    trip_ids = []
    for trip in trips:
        trip_ids.extend([trip.trip_id] * len(trip.track))
    return np.array(trip_ids), [t.trip_id for t in trips]


if __name__ == "__main__":
    # Quick test
    trip_ids = ["001"] * 10 + ["002"] * 15 + ["003"] * 8 + ["004"] * 12 + ["005"] * 20
    
    splits = create_group_aware_splits(trip_ids, n_splits=3)
    print(f"Created {len(splits)} splits")
    for i, split in enumerate(splits):
        print(f"  Fold {i}: train={split.train_trip_ids} ({len(split.train_indices)} obs), "
              f"val={split.val_trip_ids} ({len(split.val_indices)} obs)")
    
    verify_no_leakage(splits, trip_ids)