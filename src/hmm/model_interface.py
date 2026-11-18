"""
Model probability interface for PathSnap HMM emissions.

Defines a minimal documented interface that the trained classifier must implement
to provide probabilities for HMM emission scoring. The model-training session
can adapt to this interface when saving the final artifact.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np


@dataclass
class ModelProbabilities:
    """
    Probability outputs from the classifier for a set of candidates.
    
    Attributes:
        log_probs: Log-probability of positive class (is_ground_truth) for each candidate.
                   Shape: (n_candidates,)
        probs: Probability of positive class for each candidate. Shape: (n_candidates,)
        candidate_arc_ids: Arc IDs corresponding to each probability. Shape: (n_candidates,)
    """
    log_probs: np.ndarray
    probs: np.ndarray
    candidate_arc_ids: np.ndarray
    
    def __post_init__(self):
        self.log_probs = np.asarray(self.log_probs, dtype=np.float64)
        self.probs = np.asarray(self.probs, dtype=np.float64)
        self.candidate_arc_ids = np.asarray(self.candidate_arc_ids, dtype=np.int64)
        assert self.log_probs.shape == self.probs.shape == self.candidate_arc_ids.shape


class ModelInterface(ABC):
    """
    Abstract base class for model probability providers.
    
    The trained classifier (LogisticRegression, RandomForest, LightGBM, etc.)
    should be wrapped to implement this interface. This allows the HMM module
    to remain independent of the specific model artifact format.
    
    Implementation notes:
    - The model must output calibrated probabilities (use CalibratedClassifierCV).
    - Input features must match the 11 features from feature engineering.
    - The wrapper handles batching for memory efficiency on large trips.
    """
    
    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Return the 11 feature names in order expected by the model."""
        pass
    
    @property
    @abstractmethod
    def n_features(self) -> int:
        """Return number of features (should be 11)."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for candidate feature vectors.
        
        Args:
            X: Feature matrix of shape (n_candidates, n_features)
            
        Returns:
            Probability array of shape (n_candidates, 2) for binary classification,
            where column 1 is P(is_ground_truth=1).
        """
        pass
    
    @abstractmethod
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict log-probabilities for candidate feature vectors.
        
        Args:
            X: Feature matrix of shape (n_candidates, n_features)
            
        Returns:
            Log-probability array of shape (n_candidates, 2).
        """
        pass
    
    def get_probabilities(
        self,
        X: np.ndarray,
        candidate_arc_ids: np.ndarray,
        batch_size: int = 50000,
    ) -> ModelProbabilities:
        """
        Get ModelProbabilities for a batch of candidates.
        
        Handles large inputs by processing in batches.
        
        Args:
            X: Feature matrix (n_candidates, n_features)
            candidate_arc_ids: Arc IDs for each candidate (n_candidates,)
            batch_size: Max candidates per forward pass
            
        Returns:
            ModelProbabilities with log_probs, probs, and arc_ids
        """
        n = len(X)
        all_log_probs = np.zeros(n, dtype=np.float64)
        all_probs = np.zeros(n, dtype=np.float64)
        
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            X_batch = X[i:end]
            
            # Get log-probabilities (column 1 = positive class)
            log_proba = self.predict_log_proba(X_batch)
            if log_proba.shape[1] == 2:
                all_log_probs[i:end] = log_proba[:, 1]
            else:
                all_log_probs[i:end] = log_proba[:, 0]
            
            # Get probabilities
            proba = self.predict_proba(X_batch)
            if proba.shape[1] == 2:
                all_probs[i:end] = proba[:, 1]
            else:
                all_probs[i:end] = proba[:, 0]
        
        return ModelProbabilities(
            log_probs=all_log_probs,
            probs=all_probs,
            candidate_arc_ids=candidate_arc_ids,
        )


class MockModelInterface(ModelInterface):
    """
    Deterministic mock model for testing HMM without a trained classifier.
    
    Produces probabilities based on distance and candidate rank:
    - Lower perpendicular distance -> higher probability
    - Lower candidate rank -> higher probability
    - Uses a simple logistic function for realistic score distribution
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        self._feature_names = feature_names or [
            "perp_distance_m",
            "along_track_dist_m",
            "bearing_diff_deg",
            "speed_m_s",
            "time_delta_s",
            "arc_length_m",
            "from_node_out_degree",
            "to_node_in_degree",
            "arc_connectivity",
            "candidate_rank",
            "is_continuation",
        ]
    
    @property
    def feature_names(self) -> List[str]:
        return self._feature_names
    
    @property
    def n_features(self) -> int:
        return len(self._feature_names)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # X[:, 0] = perp_distance, X[:, 9] = candidate_rank
        # Distance: smaller is better. Use exp(-dist/50) as base.
        # Rank: smaller is better. Use 1/rank as base.
        dist_score = np.exp(-X[:, 0] / 50.0)
        rank_score = 1.0 / np.maximum(X[:, 9], 1.0)
        
        # Combine with some noise-like variation from bearing
        bearing_score = np.exp(-X[:, 2] / 45.0)
        
        # Weighted combination
        score = 0.5 * dist_score + 0.3 * rank_score + 0.2 * bearing_score
        
        # Convert to probability via logistic
        prob = 1.0 / (1.0 + np.exp(-10.0 * (score - 0.5)))
        
        # Clamp to avoid exact 0/1
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        
        return np.column_stack([1 - prob, prob])
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.log(proba)