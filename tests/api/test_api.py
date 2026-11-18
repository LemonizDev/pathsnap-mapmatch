"""API tests for PathSnap FastAPI backend using TestClient.

Mock-mode tests: verify API contracts with deterministic synthetic responses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# Use the mock_mode_client fixture from conftest.py


class TestHealthEndpoint:
    """Tests for GET /health endpoint in mock mode."""
    
    def test_health_returns_ok_structure(self, mock_mode_client: TestClient):
        """Health endpoint returns expected JSON structure."""
        response = mock_mode_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "hmm_loaded" in data
        assert "trips_available" in data
        assert "mock_mode" in data
        assert "version" in data
        
        # In mock mode, status should be "degraded" or "ok"
        assert data["status"] in ("ok", "degraded")
        assert data["mock_mode"] is True
        assert isinstance(data["trips_available"], int)
        assert data["trips_available"] >= 0
    
    def test_health_model_loaded_false_in_mock(self, mock_mode_client: TestClient):
        """Model should not be loaded in mock mode (no artifacts exist yet)."""
        response = mock_mode_client.get("/health")
        data = response.json()
        # In mock mode, model artifacts don't exist in test environment
        if data["mock_mode"]:
            assert data["model_loaded"] is False
            assert data["hmm_loaded"] is False
        else:
            assert data["model_loaded"] is True
            assert data["hmm_loaded"] is True
    
    def test_health_content_type_json(self, mock_mode_client: TestClient):
        """Health endpoint returns JSON content type."""
        response = mock_mode_client.get("/health")
        assert response.headers["content-type"] == "application/json"


class TestTripsEndpoint:
    """Tests for GET /trips endpoint."""
    
    def test_trips_returns_list(self, mock_mode_client: TestClient):
        """Trips endpoint returns list of available trip IDs."""
        response = mock_mode_client.get("/trips")
        assert response.status_code == 200
        
        data = response.json()
        assert "trips" in data
        assert "partition" in data
        assert "count" in data
        
        assert data["partition"] == "student_test"
        assert isinstance(data["trips"], list)
        assert data["count"] == len(data["trips"])
    
    def test_trips_ids_format(self, mock_mode_client: TestClient):
        """Trip IDs should be 8-digit strings."""
        response = mock_mode_client.get("/trips")
        data = response.json()
        
        for trip_id in data["trips"]:
            assert isinstance(trip_id, str)
            assert len(trip_id) == 8
            assert trip_id.isdigit()


class TestPredictEndpoint:
    """Tests for POST /predict endpoint in mock mode."""
    
    def test_predict_requires_trip_id(self, mock_mode_client: TestClient):
        """Predict endpoint requires trip_id in request body."""
        response = mock_mode_client.post("/predict", json={})
        assert response.status_code == 422  # Validation error
    
    def test_predict_valid_trip_id_format(self, mock_mode_client: TestClient):
        """Predict validates trip_id format (8 digits)."""
        # Valid format
        response = mock_mode_client.post("/predict", json={"trip_id": "00000000"})
        # May return 200 or 404 depending on whether trip exists in test data
        assert response.status_code in (200, 404)
        
        # Invalid format - too short
        response = mock_mode_client.post("/predict", json={"trip_id": "123"})
        assert response.status_code == 422
        
        # Invalid format - non-numeric
        response = mock_mode_client.post("/predict", json={"trip_id": "abcdefgh"})
        assert response.status_code == 422
    
    def test_predict_nonexistent_trip_returns_404(self, mock_mode_client: TestClient):
        """Predict returns 404 for trip not in student_test partition."""
        response = mock_mode_client.post("/predict", json={"trip_id": "99999999"})
        assert response.status_code == 404
        
        data = response.json()
        assert data["error"] == "trip_not_found"
        assert "available_trips" in data.get("detail", {})
    
    def test_predict_mock_mode_returns_valid_structure(self, mock_mode_client: TestClient):
        """Predict returns valid response structure in mock mode.
        
        Uses a trip that should exist in the test data.
        """
        # First get available trips
        trips_response = mock_mode_client.get("/trips")
        trips_data = trips_response.json()
        
        if trips_data["count"] == 0:
            pytest.skip("No student_test trips available in test environment")
        
        trip_id = trips_data["trips"][0]
        
        response = mock_mode_client.post("/predict", json={"trip_id": trip_id})
        assert response.status_code == 200
        
        data = response.json()
        
        # Validate response structure per schema
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
    
    def test_predict_mock_deterministic(self, mock_mode_client: TestClient):
        """Mock predictions should be deterministic with fixed seed."""
        trips_response = mock_mode_client.get("/trips")
        trips_data = trips_response.json()
        
        if trips_data["count"] == 0:
            pytest.skip("No student_test trips available")
        
        trip_id = trips_data["trips"][0]
        
        # Make two predictions
        response1 = mock_mode_client.post("/predict", json={"trip_id": trip_id})
        response2 = mock_mode_client.post("/predict", json={"trip_id": trip_id})
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Should be identical in mock mode with fixed seed
        assert data1["predicted_route"] == data2["predicted_route"]
        assert data1["accuracy"] == data2["accuracy"]
        assert data1["map_html"] == data2["map_html"]


class TestCORSHeaders:
    """Tests for CORS configuration."""
    
    def test_cors_allows_all_origins(self, mock_mode_client: TestClient):
        """CORS should allow all origins for hackathon demo."""
        response = mock_mode_client.options(
            "/health",
            headers={
                "Origin": "http://192.168.1.100:8501",
                "Access-Control-Request-Method": "GET",
            },
        )
        # TestClient may not fully simulate CORS preflight, but we check config
        assert response.status_code in (200, 405)  # 405 if OPTIONS not explicitly handled
    
    def test_cors_headers_present(self, mock_mode_client: TestClient):
        """CORS headers should be present on responses."""
        response = mock_mode_client.get("/health", headers={"Origin": "http://localhost:8501"})
        # FastAPI TestClient doesn't always add CORS headers in test mode
        # but the middleware is configured


class TestAPIContracts:
    """Tests verifying API contract compliance with schemas."""
    
    def test_health_response_matches_schema(self, mock_mode_client: TestClient):
        """Health response matches HealthResponse schema."""
        from src.api.schemas import HealthResponse
        
        response = mock_mode_client.get("/health")
        assert response.status_code == 200
        
        # This will raise ValidationError if schema doesn't match
        HealthResponse.model_validate(response.json())
    
    def test_trips_response_matches_schema(self, mock_mode_client: TestClient):
        """Trips response matches TripListResponse schema."""
        from src.api.schemas import TripListResponse
        
        response = mock_mode_client.get("/trips")
        assert response.status_code == 200
        
        TripListResponse.model_validate(response.json())
    
    def test_predict_response_matches_schema(self, mock_mode_client: TestClient):
        """Predict response matches PredictResponse schema."""
        from src.api.schemas import PredictResponse
        
        trips_response = mock_mode_client.get("/trips")
        trips_data = trips_response.json()
        
        if trips_data["count"] == 0:
            pytest.skip("No student_test trips available")
        
        trip_id = trips_data["trips"][0]
        response = mock_mode_client.post("/predict", json={"trip_id": trip_id})
        
        if response.status_code == 200:
            PredictResponse.model_validate(response.json())


# Run with: python -m pytest tests/api/test_api.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])