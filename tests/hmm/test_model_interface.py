"""Tests for HMM model interface."""

import numpy as np
import pytest

from src.hmm.model_interface import (
    ModelInterface,
    ModelProbabilities,
    MockModelInterface,
)


class TestModelProbabilities:
    def test_model_probabilities_creation(self):
        log_probs = np.array([-1.0, -2.0, -0.5])
        probs = np.array([0.3, 0.1, 0.6])
        arc_ids = np.array([1, 2, 3])
        
        mp = ModelProbabilities(log_probs, probs, arc_ids)
        
        assert np.array_equal(mp.log_probs, log_probs)
        assert np.array_equal(mp.probs, probs)
        assert np.array_equal(mp.candidate_arc_ids, arc_ids)
    
    def test_model_probabilities_type_conversion(self):
        # Test that lists are converted to arrays
        mp = ModelProbabilities([-1.0, -2.0], [0.3, 0.1], [1, 2])
        
        assert isinstance(mp.log_probs, np.ndarray)
        assert isinstance(mp.probs, np.ndarray)
        assert isinstance(mp.candidate_arc_ids, np.ndarray)
        
        assert mp.log_probs.dtype == np.float64
        assert mp.probs.dtype == np.float64
        assert mp.candidate_arc_ids.dtype == np.int64


class TestMockModelInterface:
    def test_mock_model_feature_names(self):
        model = MockModelInterface()
        
        assert len(model.feature_names) == 11
        assert model.n_features == 11
        assert "perp_distance_m" in model.feature_names
        assert "candidate_rank" in model.feature_names
    
    def test_mock_model_predict_proba_shape(self):
        model = MockModelInterface()
        
        X = np.random.rand(5, 11)
        proba = model.predict_proba(X)
        
        assert proba.shape == (5, 2)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_mock_model_predict_log_proba_shape(self):
        model = MockModelInterface()
        
        X = np.random.rand(5, 11)
        log_proba = model.predict_log_proba(X)
        
        assert log_proba.shape == (5, 2)
        assert np.all(log_proba <= 0)  # log probs are <= 0
    
    def test_mock_model_distance_effect(self):
        """Test that closer distance gives higher probability."""
        model = MockModelInterface()
        
        # Create two candidates with different distances
        X_close = np.array([[5.0, 10.0, 10.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 1.0, 0.0]])
        X_far = np.array([[50.0, 10.0, 10.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 1.0, 0.0]])
        
        prob_close = model.predict_proba(X_close)[0, 1]
        prob_far = model.predict_proba(X_far)[0, 1]
        
        assert prob_close > prob_far, "Closer distance should give higher probability"
    
    def test_mock_model_rank_effect(self):
        """Test that lower rank gives higher probability."""
        model = MockModelInterface()
        
        # Same distance, different ranks
        X_rank1 = np.array([[10.0, 10.0, 10.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 1.0, 0.0]])
        X_rank3 = np.array([[10.0, 10.0, 10.0, 1.0, 1.0, 1000.0, 2.0, 2.0, 3.0, 3.0, 0.0]])
        
        prob_rank1 = model.predict_proba(X_rank1)[0, 1]
        prob_rank3 = model.predict_proba(X_rank3)[0, 1]
        
        assert prob_rank1 > prob_rank3, "Lower rank should give higher probability"
    
    def test_get_probabilities_batch(self):
        """Test batched probability computation."""
        model = MockModelInterface()
        
        X = np.random.rand(100, 11)
        arc_ids = np.arange(100)
        
        probs = model.get_probabilities(X, arc_ids, batch_size=30)
        
        assert len(probs.log_probs) == 100
        assert len(probs.probs) == 100
        assert np.array_equal(probs.candidate_arc_ids, arc_ids)
    
    def test_get_probabilities_empty(self):
        """Test with empty input."""
        model = MockModelInterface()
        
        X = np.zeros((0, 11))
        arc_ids = np.zeros(0, dtype=np.int64)
        
        probs = model.get_probabilities(X, arc_ids)
        
        assert len(probs.log_probs) == 0
        assert len(probs.probs) == 0
        assert len(probs.candidate_arc_ids) == 0


class TestModelInterfaceABC:
    def test_cannot_instantiate_abstract(self):
        """Verify ModelInterface is abstract."""
        with pytest.raises(TypeError):
            ModelInterface()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])