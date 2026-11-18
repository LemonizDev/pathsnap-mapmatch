"""
Data loader for PathSnap GPS trajectory map-matching dataset.

Parses four file formats:
- .track: GPS observations (longitude, latitude, timestamp)
- .nodes: Road network nodes (longitude, latitude) indexed by node_id
- .arcs: Road network arcs (from_node, to_node) indexed by arc_id
- .route: Ground truth route (sequence of arc_ids)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import warnings


@dataclass
class TrackPoint:
    """Single GPS observation."""
    lon: float
    lat: float
    timestamp: float


@dataclass
class Node:
    """Road network node."""
    node_id: int
    lon: float
    lat: float


@dataclass
class Arc:
    """Road network arc (directed edge)."""
    arc_id: int
    from_node: int
    to_node: int


@dataclass
class TripData:
    """Complete trip data container."""
    trip_id: str
    track: np.ndarray  # shape (N, 3): lon, lat, timestamp
    nodes: np.ndarray  # shape (M, 2): lon, lat (index = node_id)
    arcs: np.ndarray   # shape (K, 2): from_node, to_node (index = arc_id)
    route: np.ndarray  # shape (L,): arc_id sequence (ground truth)
    
    def __post_init__(self):
        self.track = np.asarray(self.track, dtype=np.float64)
        self.nodes = np.asarray(self.nodes, dtype=np.float64)
        self.arcs = np.asarray(self.arcs, dtype=np.int64)
        self.route = np.asarray(self.route, dtype=np.int64)


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


# Expected coordinate bounds for the dataset (full valid WGS84 range)
# Dataset spans multiple continents: lon [-180, 180], lat [-90, 90]
# This retains geographic sanity checks while allowing all supplied data
LON_MIN, LON_MAX = -180.0, 180.0
LAT_MIN, LAT_MAX = -90.0, 90.0
MAX_TIMESTAMP_GAP = 60.0  # seconds


def validate_coordinates(lon: float, lat: float, context: str) -> None:
    """Validate that coordinates are within expected bounds."""
    if not (LON_MIN <= lon <= LON_MAX):
        raise DataValidationError(f"{context}: longitude {lon} out of bounds [{LON_MIN}, {LON_MAX}]")
    if not (LAT_MIN <= lat <= LAT_MAX):
        raise DataValidationError(f"{context}: latitude {lat} out of bounds [{LAT_MIN}, {LAT_MAX}]")


def validate_timestamp_monotonic(timestamps: np.ndarray, context: str) -> None:
    """Validate timestamps are monotonically increasing (allowing gaps)."""
    diffs = np.diff(timestamps)
    if np.any(diffs < 0):
        raise DataValidationError(f"{context}: timestamps not monotonically increasing")
    if np.any(diffs > MAX_TIMESTAMP_GAP):
        warnings.warn(f"{context}: large timestamp gap detected (max {diffs.max():.1f}s)")


def validate_arc_references(arcs: np.ndarray, num_nodes: int, context: str) -> None:
    """Validate that arc endpoints reference valid node_ids."""
    max_node_id = num_nodes - 1
    if arcs.size > 0:
        invalid_from = np.where((arcs[:, 0] < 0) | (arcs[:, 0] > max_node_id))[0]
        invalid_to = np.where((arcs[:, 1] < 0) | (arcs[:, 1] > max_node_id))[0]
        if len(invalid_from) > 0:
            raise DataValidationError(f"{context}: {len(invalid_from)} arcs reference invalid from_node (max={max_node_id})")
        if len(invalid_to) > 0:
            raise DataValidationError(f"{context}: {len(invalid_to)} arcs reference invalid to_node (max={max_node_id})")


def validate_route_references(route: np.ndarray, num_arcs: int, context: str) -> None:
    """Validate that route arc_ids exist in arcs."""
    if route.size > 0:
        invalid = np.where((route < 0) | (route >= num_arcs))[0]
        if len(invalid) > 0:
            raise DataValidationError(f"{context}: {len(invalid)} route entries reference invalid arc_id (max={num_arcs-1})")


def parse_track_file(filepath: Path) -> np.ndarray:
    """Parse .track file: whitespace-separated lon, lat, timestamp (no header)."""
    data = np.loadtxt(filepath, dtype=np.float64, ndmin=2)
    if data.shape[1] != 3:
        raise DataValidationError(f"{filepath}: expected 3 columns (lon, lat, timestamp), got {data.shape[1]}")
    
    # Validate coordinates
    for i, (lon, lat, ts) in enumerate(data):
        validate_coordinates(lon, lat, f"{filepath}: line {i+1}")
    
    # Validate timestamps
    validate_timestamp_monotonic(data[:, 2], f"{filepath}")
    
    return data


def parse_nodes_file(filepath: Path) -> np.ndarray:
    """Parse .nodes file: whitespace-separated lon, lat (no header, 0-based index = node_id)."""
    data = np.loadtxt(filepath, dtype=np.float64, ndmin=2)
    if data.shape[1] != 2:
        raise DataValidationError(f"{filepath}: expected 2 columns (lon, lat), got {data.shape[1]}")
    
    # Validate coordinates
    for i, (lon, lat) in enumerate(data):
        validate_coordinates(lon, lat, f"{filepath}: node_id {i}")
    
    return data


def parse_arcs_file(filepath: Path) -> np.ndarray:
    """Parse .arcs file: whitespace-separated from_node, to_node (no header, 0-based index = arc_id)."""
    data = np.loadtxt(filepath, dtype=np.int64, ndmin=2)
    if data.shape[1] != 2:
        raise DataValidationError(f"{filepath}: expected 2 columns (from_node, to_node), got {data.shape[1]}")
    
    if np.any(data < 0):
        raise DataValidationError(f"{filepath}: negative node indices found")
    
    return data


def parse_route_file(filepath: Path) -> np.ndarray:
    """Parse .route file: one arc_id per line (ground truth sequence)."""
    data = np.loadtxt(filepath, dtype=np.int64, ndmin=1)
    if data.ndim != 1:
        raise DataValidationError(f"{filepath}: expected 1 column (arc_id), got shape {data.shape}")
    
    if np.any(data < 0):
        raise DataValidationError(f"{filepath}: negative arc_ids found")
    
    return data


def load_trip(trip_dir: Path, trip_id: Optional[str] = None, validate: bool = True) -> TripData:
    """
    Load all four files for a single trip.
    
    Args:
        trip_dir: Directory containing the four trip files
        trip_id: Trip identifier (defaults to directory name)
        validate: Whether to run cross-file validation
        
    Returns:
        TripData object with all parsed arrays
        
    Raises:
        DataValidationError: If any validation fails
        FileNotFoundError: If any required file is missing
    """
    if trip_id is None:
        trip_id = trip_dir.name
    
    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"
    
    for path in [track_path, nodes_path, arcs_path, route_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    
    track = parse_track_file(track_path)
    nodes = parse_nodes_file(nodes_path)
    arcs = parse_arcs_file(arcs_path)
    route = parse_route_file(route_path)
    
    if validate:
        # Cross-file validation
        validate_arc_references(arcs, len(nodes), f"{trip_id}")
        validate_route_references(route, len(arcs), f"{trip_id}")
    
    return TripData(
        trip_id=trip_id,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route
    )


def discover_trips(base_dir: Path) -> Tuple[List[Path], List[str]]:
    """
    Discover all trip directories in a base directory.
    
    Returns:
        Tuple of (trip_dirs, trip_ids) sorted by trip_id
    """
    trip_dirs = []
    trip_ids = []
    
    for entry in sorted(base_dir.iterdir()):
        if entry.is_dir() and entry.name.isdigit():
            # Check that all four files exist
            required_files = [
                entry / f"{entry.name}.track",
                entry / f"{entry.name}.nodes",
                entry / f"{entry.name}.arcs",
                entry / f"{entry.name}.route"
            ]
            if all(f.exists() for f in required_files):
                trip_dirs.append(entry)
                trip_ids.append(entry.name)
            else:
                warnings.warn(f"Trip {entry.name} missing required files, skipping")
    
    return trip_dirs, trip_ids


def load_partition(partition_dir: Path, validate: bool = True) -> List[TripData]:
    """
    Load all trips in a partition directory (train or student_test).
    
    Args:
        partition_dir: Path to partition directory (e.g., .../train/)
        validate: Whether to run validation
        
    Returns:
        List of TripData objects sorted by trip_id
    """
    trip_dirs, trip_ids = discover_trips(partition_dir)
    trips = []
    
    for trip_dir, trip_id in zip(trip_dirs, trip_ids):
        try:
            trip = load_trip(trip_dir, trip_id, validate=validate)
            trips.append(trip)
        except Exception as e:
            raise DataValidationError(f"Failed to load trip {trip_id}: {e}")
    
    return trips


def get_dataset_stats(trips: List[TripData]) -> dict:
    """Compute summary statistics for a list of trips."""
    if not trips:
        return {"num_trips": 0}
    
    total_obs = sum(len(t.track) for t in trips)
    total_nodes = sum(len(t.nodes) for t in trips)
    total_arcs = sum(len(t.arcs) for t in trips)
    total_route_len = sum(len(t.route) for t in trips)
    
    all_lons = np.concatenate([t.track[:, 0] for t in trips])
    all_lats = np.concatenate([t.track[:, 1] for t in trips])
    all_timestamps = np.concatenate([t.track[:, 2] for t in trips])
    
    return {
        "num_trips": len(trips),
        "total_gps_observations": int(total_obs),
        "total_nodes": int(total_nodes),
        "total_arcs": int(total_arcs),
        "total_route_arcs": int(total_route_len),
        "avg_obs_per_trip": total_obs / len(trips),
        "avg_nodes_per_trip": total_nodes / len(trips),
        "avg_arcs_per_trip": total_arcs / len(trips),
        "avg_route_len": total_route_len / len(trips),
        "lon_range": [float(all_lons.min()), float(all_lons.max())],
        "lat_range": [float(all_lats.min()), float(all_lats.max())],
        "timestamp_range": [float(all_timestamps.min()), float(all_timestamps.max())],
        "trip_ids": [t.trip_id for t in trips],
    }


if __name__ == "__main__":
    # Quick test
    base = Path("Student folder/Student folder")
    train_trips = load_partition(base / "train")
    test_trips = load_partition(base / "student_test")
    
    print(f"Train trips: {len(train_trips)}")
    print(f"Student test trips: {len(test_trips)}")
    
    if train_trips:
        stats = get_dataset_stats(train_trips)
        print(f"Train stats: {stats}")
    
    if test_trips:
        stats = get_dataset_stats(test_trips)
        print(f"Test stats: {stats}")