"""PathSnap FastAPI Application.

Provides REST endpoints for GPS trajectory map-matching:
- GET /health - Health check with artifact loading status
- POST /predict - Predict route for a trip_id
- GET /trips - List available student_test trip IDs

Designed for two-laptop demo: binds to 0.0.0.0 for LAN access.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.config import settings
from src.api.schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    ErrorResponse,
    TripListResponse,
    Latencies,
)
from src.api.load_artifacts import (
    ArtifactLoader,
    init_artifacts,
    get_artifact_loader,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load artifacts at startup."""
    # Startup
    init_artifacts()
    yield
    # Shutdown (no cleanup needed)


app = FastAPI(
    title="PathSnap Map-Matching API",
    description="GPS trajectory to road network map-matching using ML + HMM",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for two-laptop demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors.allow_origins,
    allow_credentials=settings.cors.allow_credentials,
    allow_methods=settings.cors.allow_methods,
    allow_headers=settings.cors.allow_headers,
)


# Dependency for artifact loader
def get_loader() -> ArtifactLoader:
    return get_artifact_loader()


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(loader: ArtifactLoader = Depends(get_loader)) -> HealthResponse:
    """Health check endpoint.
    
    Returns artifact loading status and available trip count.
    Used by frontend and monitoring to verify backend readiness.
    """
    model_loaded = loader.model_artifacts.loaded
    hmm_loaded = loader.hmm_params.loaded
    model_source = loader.model_artifacts.source
    hmm_source = loader.hmm_params.source
    
    # Overall status: ok if at least mock mode works, degraded if artifacts missing but mock enabled
    if model_loaded and hmm_loaded:
        status = "ok"
    elif settings.mock_enabled:
        status = "degraded"  # Running in mock mode
    else:
        status = "degraded"  # No artifacts and no mock
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        model_source=model_source,
        hmm_loaded=hmm_loaded,
        hmm_source=hmm_source,
        trips_available=len(loader.student_test_trip_ids),
        mock_mode=settings.mock_enabled,
        version=app.version,
    )


@app.get("/trips", response_model=TripListResponse, tags=["Utility"])
async def list_trips(loader: ArtifactLoader = Depends(get_loader)) -> TripListResponse:
    """List available student_test trip IDs for UI dropdown."""
    return TripListResponse(
        trips=loader.student_test_trip_ids,
        partition="student_test",
        count=len(loader.student_test_trip_ids),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Trip not found"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
    },
    tags=["Prediction"],
)
async def predict(
    request: PredictRequest,
    loader: ArtifactLoader = Depends(get_loader),
) -> PredictResponse:
    """Predict the most likely road route for a GPS trajectory.
    
    Takes a trip_id from the student_test partition and returns:
    - predicted_route: sequence of arc_ids
    - accuracy: route accuracy vs ground truth (computed server-side)
    - map_html: Folium map with raw GPS, predicted route, hidden ground truth
    - latencies_ms: per-stage timing breakdown
    
    Ground truth .route files never leave the backend.
    """
    trip_id = request.trip_id
    
    # Validate trip exists
    if not loader.is_trip_available(trip_id):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "trip_not_found",
                "message": f"Trip '{trip_id}' not found in student_test partition",
                "detail": {"available_trips": loader.student_test_trip_ids[:20]},
            },
        )
    
    total_start = time.perf_counter()
    
    try:
        # Stage 1: Candidate generation
        cand_start = time.perf_counter()
        candidate_sets = loader.generate_candidates(trip_id, k=10, radius_meters=100.0)
        candidates_ms = int((time.perf_counter() - cand_start) * 1000)
        
        # Stage 2: Model inference + HMM/Viterbi (real or mock)
        model_start = time.perf_counter()
        
        if loader.model_artifacts.loaded and loader.hmm_params.loaded and not settings.mock_enabled:
            # Real model + HMM prediction
            predicted_route, accuracy, map_html, model_ms, hmm_ms = loader.predict_with_model(trip_id, candidate_sets)
        else:
            # Deterministic mock prediction for testing
            predicted_route, accuracy, map_html, model_ms, hmm_ms = loader.predict_mock(trip_id)
        
        total_ms = int((time.perf_counter() - total_start) * 1000)
        
        return PredictResponse(
            trip_id=trip_id,
            predicted_route=predicted_route,
            accuracy=accuracy,
            map_html=map_html,
            latencies_ms=Latencies(
                candidates=candidates_ms,
                model=model_ms,
                hmm=hmm_ms,
                total=total_ms,
            ),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Log error and return 500
        raise HTTPException(
            status_code=500,
            detail={
                "error": "prediction_failed",
                "message": f"Prediction failed for trip '{trip_id}': {str(e)}",
                "detail": {"trip_id": trip_id},
            },
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "message": str(exc.detail)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": f"Internal server error: {str(exc)}"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        log_level=settings.api.log_level,
    )