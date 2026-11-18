"""
PathSnap HMM Module

Hidden Markov Model + Viterbi decoding for GPS trajectory map-matching.

Components:
- model_interface: Abstract interface for classifier probability providers
- emission: Emission scoring (model log-prob - lambda * distance)
- transition: Graph-aware transitions with BFS hop distances
- viterbi: Numerically stable Viterbi decoding with variable K_t
- comparison: Classifier-only vs Classifier+HMM evaluation utility
"""

from src.hmm.model_interface import (
    ModelInterface,
    ModelProbabilities,
    MockModelInterface,
)

from src.hmm.emission import (
    EmissionConfig,
    EmissionScores,
    compute_emission_scores,
    compute_emission_scores_from_model,
    get_emission_for_timestep,
)

from src.hmm.transition import (
    TransitionConfig,
    TransitionScores,
    compute_hop_distances,
    compute_transition_scores,
    compute_transition_matrix_for_viterbi,
    get_transition_matrix,
)

from src.hmm.viterbi import (
    ViterbiConfig,
    ViterbiResult,
    viterbi_decode,
    viterbi_decode_padded,
    reconstruct_path_from_result,
)

from src.hmm.comparison import (
    ComparisonConfig,
    ComparisonMetrics,
    ComparisonResult,
    compare_methods,
    compare_on_partition,
    compute_route_continuity,
    compute_top1_accuracy,
    print_comparison_summary,
)

__all__ = [
    # Model interface
    "ModelInterface",
    "ModelProbabilities",
    "MockModelInterface",
    
    # Emission
    "EmissionConfig",
    "EmissionScores",
    "compute_emission_scores",
    "compute_emission_scores_from_model",
    "get_emission_for_timestep",
    
    # Transition
    "TransitionConfig",
    "TransitionScores",
    "compute_hop_distances",
    "compute_transition_scores",
    "compute_transition_matrix_for_viterbi",
    "get_transition_matrix",
    
    # Viterbi
    "ViterbiConfig",
    "ViterbiResult",
    "viterbi_decode",
    "viterbi_decode_padded",
    "reconstruct_path_from_result",
    
    # Comparison
    "ComparisonConfig",
    "ComparisonMetrics",
    "ComparisonResult",
    "compare_methods",
    "compare_on_partition",
    "compute_route_continuity",
    "compute_top1_accuracy",
    "print_comparison_summary",
]