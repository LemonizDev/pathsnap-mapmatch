"""Feature engineering module for PathSnap."""

from .candidates import (
    Candidate,
    CandidateSet,
    generate_candidates_for_trip,
    generate_candidates_for_partition,
    haversine_distance,
    bearing,
    perpendicular_distance_to_segment,
    compute_arc_geometry,
)

from .engineer import (
    FeatureVector,
    compute_node_degrees,
    compute_arc_lengths,
    compute_adjacency,
    extract_candidate_features,
    build_feature_matrix,
    build_features_for_partition,
    save_features,
    load_features,
)

__all__ = [
    "Candidate",
    "CandidateSet",
    "generate_candidates_for_trip",
    "generate_candidates_for_partition",
    "haversine_distance",
    "bearing",
    "perpendicular_distance_to_segment",
    "compute_arc_geometry",
    "FeatureVector",
    "compute_node_degrees",
    "compute_arc_lengths",
    "compute_adjacency",
    "extract_candidate_features",
    "build_feature_matrix",
    "build_features_for_partition",
    "save_features",
    "load_features",
]