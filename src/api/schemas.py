"""Pydantic schemas for PathSnap API request/response validation."""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class PredictRequest(BaseModel):
    """Request body for POST /predict endpoint."""
    trip_id: str = Field(
        ...,
        description="Trip identifier (e.g., '00000000'). Must exist in student_test partition.",
        pattern=r"^\d{8}$",
        examples=["00000000"]
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"trip_id": "00000000"}
        }
    )


class Latencies(BaseModel):
    """Latency breakdown in milliseconds."""
    candidates: int = Field(..., description="Candidate generation time (ms)")
    model: int = Field(..., description="Model inference time (ms)")
    hmm: int = Field(..., description="HMM/Viterbi decoding time (ms)")
    total: int = Field(..., description="Total end-to-end time (ms)")


class PredictResponse(BaseModel):
    """Response body for POST /predict endpoint."""
    trip_id: str = Field(..., description="Echo of requested trip_id")
    predicted_route: List[int] = Field(
        ...,
        description="Predicted sequence of arc_ids (road segments)"
    )
    accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Route accuracy vs ground truth (computed server-side)"
    )
    map_html: str = Field(
        ...,
        description="Folium HTML map with raw GPS (blue), predicted route (green), ground truth (hidden)"
    )
    latencies_ms: Latencies = Field(..., description="Per-stage latency breakdown")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trip_id": "00000000",
                "predicted_route": [42, 43, 44, 45, 46, 47, 48],
                "accuracy": 0.87,
                "map_html": "<div>...Folium map HTML...</div>",
                "latencies_ms": {
                    "candidates": 45,
                    "model": 12,
                    "hmm": 8,
                    "total": 78
                }
            }
        }
    )


class HealthResponse(BaseModel):
    """Response body for GET /health endpoint."""
    status: str = Field(..., description="Overall health status: 'ok' or 'degraded'")
    model_loaded: bool = Field(..., description="Whether ML model artifact is loaded")
    model_source: str = Field(..., description="Model artifact source: 'real' or 'synthetic'")
    hmm_loaded: bool = Field(..., description="Whether HMM parameters are loaded")
    hmm_source: str = Field(..., description="HMM params source: 'real', 'defaulted', or 'synthetic'")
    trips_available: int = Field(..., description="Number of student_test trips available")
    mock_mode: bool = Field(..., description="Whether deterministic mock mode is active")
    version: str = Field(default="0.1.0", description="API version")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "model_loaded": True,
                "model_source": "real",
                "hmm_loaded": True,
                "hmm_source": "real",
                "trips_available": 10,
                "mock_mode": False,
                "version": "0.1.0"
            }
        }
    )


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "trip_not_found",
                "message": "Trip '00000099' not found in student_test partition",
                "detail": {"available_trips": ["00000000", "00000007", "..."]}
            }
        }
    )


class TripListResponse(BaseModel):
    """Response for GET /trips endpoint (utility for UI)."""
    trips: List[str] = Field(..., description="Available student_test trip IDs")
    partition: str = Field(default="student_test", description="Partition name")
    count: int = Field(..., description="Number of available trips")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trips": ["00000000", "00000007", "00000009", "00000040", "00000052", 
                         "00000056", "00000065", "00000076", "00000077", "00000091"],
                "partition": "student_test",
                "count": 10
            }
        }
    )