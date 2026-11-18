"""Tests for split utilities."""

import numpy as np
import pytest
from src.data.splits import (
    create_group_aware_splits,
    create_holdout_split,
    verify_no_leakage,
    get_trip_ids_from_trips,
    SplitIndices,
)


def test_create_group_aware_splits():
    # 5 trips with different numbers of observations
    trip_ids = ["001"] * 10 + ["002"] * 15 + ["003"] * 8 + ["004"] * 12 + ["005"] * 20
    
    splits = create_group_aware_splits(trip_ids, n_splits=3)
    
    assert len(splits) == 3
    
    # Check no leakage
    assert verify_no_leakage(splits, trip_ids)
    
    # Check all observations accounted for in each fold
    for split in splits:
        total = len(split.train_indices) + len(split.val_indices)
        assert total == len(trip_ids)
        
        # Check trip IDs match indices
        trip_ids_arr = np.array(trip_ids)
        train_trips = set(trip_ids_arr[split.train_indices])
        val_trips = set(trip_ids_arr[split.val_indices])
        assert set(split.train_trip_ids) == train_trips
        assert set(split.val_trip_ids) == val_trips


def test_create_holdout_split():
    trip_ids = ["001"] * 10 + ["002"] * 15 + ["003"] * 8
    
    split = create_holdout_split(trip_ids, holdout_trip_ids=["002"])
    
    assert split.val_trip_ids == ["002"]
    assert set(split.train_trip_ids) == {"001", "003"}
    assert len(split.val_indices) == 15
    assert len(split.train_indices) == 18
    
    assert verify_no_leakage([split], trip_ids)


def test_verify_no_leakage_detects_leakage():
    trip_ids = ["001"] * 10 + ["002"] * 10
    
    # Create a split with leakage manually
    split = SplitIndices(
        train_indices=np.array([0, 1, 10, 11]),
        val_indices=np.array([5, 6, 15, 16]),
        train_trip_ids=["001", "002"],  # 002 appears in both!
        val_trip_ids=["001", "002"],
    )
    
    assert not verify_no_leakage([split], trip_ids)


def test_get_trip_ids_from_trips():
    # Mock trip objects
    class MockTrip:
        def __init__(self, trip_id, n_obs):
            self.trip_id = trip_id
            self.track = np.zeros((n_obs, 3))
    
    trips = [
        MockTrip("001", 10),
        MockTrip("002", 5),
        MockTrip("003", 8),
    ]
    
    trip_ids_arr, unique_ids = get_trip_ids_from_trips(trips)
    
    assert len(trip_ids_arr) == 23
    assert list(trip_ids_arr[:10]) == ["001"] * 10
    assert list(trip_ids_arr[10:15]) == ["002"] * 5
    assert list(trip_ids_arr[15:]) == ["003"] * 8
    assert unique_ids == ["001", "002", "003"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])