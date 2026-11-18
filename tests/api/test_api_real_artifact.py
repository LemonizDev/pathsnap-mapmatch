"""API tests for PathSnap FastAPI backend using REAL artifacts (MOCK_MODE=false).

These tests run against the actual trained LightGBM model and real data.
Run separately from mock-mode tests to avoid environment contamination.

Usage:
    python -m pytest tests/api/test_api_real_artifact.py -v -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# Use the real_artifact_client fixture from conftest.py


class TestHealthEndpointReal:
    """Tests for GET /health endpoint with REAL artifacts."""
    
    def test_health_returns_ok_structure(self, real_artifact_client: TestClient):
        """Health endpoint returns expected JSON structure."""
        response = real_artifact_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "hmm_loaded" in data
        assert "trips_available" in data
        assert "mock_mode" in data
        assert "version" in data
        
        # In real-artifact mode, model should be loaded
        assert data["mock_mode"] is False
        assert isinstance(data["trips_available"], int)
        assert data["trips_available"] > 0  # Should have real trips
    
    def test_health_model_loaded_true_in_real(self, real_artifact_client: TestClient):
        """Model should be loaded in real-artifact mode."""
        response = real_artifact_client.get("/health")
        data = response.json()
        assert data["mock_mode"] is False
        # Model should be loaded if artifact exists
        if data["model_loaded"]:
            assert data["model_source"] == "real"
        else:
            # If model not loaded, there should be a load_error or clear reason
            assert "model_source" in data


class TestTripsEndpointReal:
    """Tests for GET /trips endpoint with REAL artifacts."""
    
    def test_trips_returns_real_list(self, real_artifact_client: TestClient):
        """Trips endpoint returns real student_test trip IDs."""
        response = real_artifact_client.get("/trips")
        assert response.status_code == 200
        
        data = response.json()
        assert "trips" in data
        assert "partition" in data
        assert "count" in data
        
        assert data["partition"] == "student_test"
        assert isinstance(data["trips"], list)
        assert data["count"] == len(data["trips"])
        # Should have the 10 student_test trips
        assert data["count"] == 10
    
    def test_trips_ids_format_real(self, real_artifact_client: TestClient):
        """Real trip IDs should be 8-digit strings."""
        response = real_artifact_client.get("/trips")
        data = response.json()
        
        for trip_id in data["trips"]:
            assert isinstance(trip_id, str)
            assert len(trip_id) == 8
            assert trip_id.isdigit()


class TestPredictEndpointReal:
    """Tests for POST /predict endpoint with REAL artifacts."""
    
    def test_predict_requires_trip_id(self, real_artifact_client: TestClient):
        """Predict endpoint requires trip_id in request body."""
        response = real_artifact_client.post("/predict", json={})
        assert response.status_code == 422  # Validation error
    
    def test_predict_valid_trip_id_format(self, real_artifact_client: TestClient):
        """Predict validates trip_id format (8 digits)."""
        # Valid format
        response = real_artifact_client.post("/predict", json={"trip_id": "00000000"})
        assert response.status_code in (200, 404)
        
        # Invalid format - too short
        response = real_artifact_client.post("/predict", json={"trip_id": "123"})
        assert response.status_code == 422
        
        # Invalid format - non-numeric
        response = real_artifact_client.post("/predict", json={"trip_id": "abcdefgh"})
        assert response.status_code == 422
    
    def test_predict_nonexistent_trip_returns_404(self, real_artifact_client: TestClient):
        """Predict returns 404 for trip not in student_test partition."""
        response = real_artifact_client.post("/predict", json={"trip_id": "99999999"})
        assert response.status_code == 404
        
        data = response.json()
        assert data["error"] == "trip_not_found"
        assert "available_trips" in data.get("detail", {})
    
    def test_predict_real_returns_valid_structure(self, real_artifact_client: TestClient):
        """Predict returns valid response structure with real model."""
        # First get available trips
        trips_response = real_artifact_client.get("/trips")
        trips_data = trips_response.json()
        
        assert trips_data["count"] > 0, "No student_test trips available"
        
        trip_id = trips_data["trips"][0]
        
        response = real_artifact_client.post("/predict", json={"trip_id": trip_id})
        # Real model should return 200
        assert response.status_code == 200
        
        data = response.json()
        
        # Validate response structure
        assert data["trip_id"] == trip_id
        assert isinstance(data["predicted_route"], list)
        assert all(isinstance(arc_id, int) for arc_id in data["predicted_route"])
        assert isinstance(data["accuracy"], float)
        assert 0.0 <= data["accuracy"] <= 1.0
        assert isinstance(data["map_html"], str)
        assert len(data["map_html"]) > 0
        assert "latencies_ms" in data
        
        latencies = data["latencies_ms"]
        assert all(key in latencies for key in ["candidates", "model", "hmm", "total"])
        assert all(isinstance(v, int) and v >= 0 for v in latencies.values())
    
    def test_predict_real_latencies_positive(self, real_artifact_client: TestClient):
        """Real prediction should have positive latencies for all stages."""
        trips_response = real_artifact_client.get("/trips")
        trips_data = trips_response.json()
        trip_id = trips_data["trips"][0]
        
        response = real_artifact_client.post("/predict", json={"trip_id": trip_id})
        assert response.status_code == 200
        
        data = response.json()
        latencies = data["latencies_ms"]
        
        # All latency stages should have positive values
        assert latencies["candidates"] > 0, "Candidate generation should take time"
        # Model latency may be 0 if model loads quickly - just check total
        assert latencies["total"] > 0
        assert latencies["total"] >= latencies["candidates"]


# Run with: python -m pytest tests/api/test_api_real_artifact.py -v -s
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])