#!/usr/bin/env python
"""
Dataset inventory and EDA script for PathSnap.

Discovers train and student_test trips, reports counts/shapes/missingness/
basic geographic and timestamp statistics. Does NOT load evaluator test data.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.loader import load_partition, get_dataset_stats, discover_trips


def analyze_partition(partition_name: str, partition_dir: Path) -> dict:
    """Analyze a single partition (train or student_test)."""
    print(f"\n{'='*60}")
    print(f"Analyzing {partition_name} partition: {partition_dir}")
    print(f"{'='*60}")
    
    # Discover trips
    trip_dirs, trip_ids = discover_trips(partition_dir)
    print(f"Discovered {len(trip_ids)} trips: {trip_ids}")
    
    if not trip_ids:
        return {"partition": partition_name, "num_trips": 0, "trip_ids": []}
    
    # Load all trips
    trips = load_partition(partition_dir, validate=True)
    
    # Get aggregate stats
    stats = get_dataset_stats(trips)
    stats["partition"] = partition_name
    
    # Per-trip detailed stats
    per_trip = []
    for trip in trips:
        track = trip.track
        timestamps = track[:, 2]
        dt = np.diff(timestamps) if len(timestamps) > 1 else np.array([0])
        
        trip_info = {
            "trip_id": trip.trip_id,
            "num_observations": len(track),
            "num_nodes": len(trip.nodes),
            "num_arcs": len(trip.arcs),
            "route_length": len(trip.route),
            "timestamp_range": [float(timestamps.min()), float(timestamps.max())],
            "duration_seconds": float(timestamps.max() - timestamps.min()) if len(timestamps) > 0 else 0,
            "mean_dt": float(dt.mean()) if len(dt) > 0 else 0,
            "max_dt": float(dt.max()) if len(dt) > 0 else 0,
            "lon_range": [float(track[:, 0].min()), float(track[:, 0].max())],
            "lat_range": [float(track[:, 1].min()), float(track[:, 1].max())],
        }
        per_trip.append(trip_info)
    
    stats["per_trip"] = per_trip
    
    # Print summary
    print(f"\nPartition Summary:")
    print(f"  Trips: {stats['num_trips']}")
    print(f"  Total GPS observations: {stats['total_gps_observations']}")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total arcs: {stats['total_arcs']}")
    print(f"  Total route arcs: {stats['total_route_arcs']}")
    print(f"  Lon range: {stats['lon_range']}")
    print(f"  Lat range: {stats['lat_range']}")
    print(f"  Timestamp range: {stats['timestamp_range']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="PathSnap dataset inventory and EDA")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Student folder/Student folder"),
        help="Root directory containing train/ and student_test/ subdirectories"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/dataset_inventory.json"),
        help="Output JSON file for inventory results"
    )
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=["train", "student_test"],
        default=["train", "student_test"],
        help="Partitions to analyze"
    )
    args = parser.parse_args()
    
    data_root = args.data_root
    if not data_root.exists():
        print(f"Error: Data root not found: {data_root}")
        sys.exit(1)
    
    all_stats = {}
    
    for partition in args.partitions:
        partition_dir = data_root / partition
        if not partition_dir.exists():
            print(f"Warning: Partition directory not found: {partition_dir}")
            continue
        
        stats = analyze_partition(partition, partition_dir)
        all_stats[partition] = stats
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")
    
    # Print cross-partition summary
    if "train" in all_stats and "student_test" in all_stats:
        train = all_stats["train"]
        test = all_stats["student_test"]
        print(f"\nCross-partition Summary:")
        print(f"  Train trips: {train.get('num_trips', 0)}")
        print(f"  Student test trips: {test.get('num_trips', 0)}")
        print(f"  Train GPS obs: {train.get('total_gps_observations', 0)}")
        print(f"  Test GPS obs: {test.get('total_gps_observations', 0)}")


if __name__ == "__main__":
    main()