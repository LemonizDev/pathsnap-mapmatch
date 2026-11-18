"""Data loading and validation module for PathSnap."""

from .loader import (
    TrackPoint,
    Node,
    Arc,
    TripData,
    DataValidationError,
    parse_track_file,
    parse_nodes_file,
    parse_arcs_file,
    parse_route_file,
    load_trip,
    discover_trips,
    load_partition,
    get_dataset_stats,
)
from .splits import (
    SplitIndices,
    create_group_aware_splits,
    create_holdout_split,
    save_splits,
    load_splits,
    verify_no_leakage,
    get_trip_ids_from_trips,
)

__all__ = [
    "TrackPoint",
    "Node", 
    "Arc",
    "TripData",
    "DataValidationError",
    "parse_track_file",
    "parse_nodes_file",
    "parse_arcs_file",
    "parse_route_file",
    "load_trip",
    "discover_trips",
    "load_partition",
    "get_dataset_stats",
    "SplitIndices",
    "create_group_aware_splits",
    "create_holdout_split",
    "save_splits",
    "load_splits",
    "verify_no_leakage",
    "get_trip_ids_from_trips",
]