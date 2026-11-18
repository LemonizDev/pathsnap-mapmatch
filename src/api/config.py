"""Configuration management for PathSnap API.

Loads settings from environment variables with sensible defaults.
Designed for two-laptop demo: backend binds to 0.0.0.0 for LAN access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"


@dataclass
class ArtifactPaths:
    """Expected artifact file paths (to be populated by model/HMM sessions)."""
    # Model artifacts (set by Model Lead after training)
    # Default to real LightGBM artifact when MOCK_MODE=false
    model_path: Optional[Path] = None  # e.g., experiments/models/LightGBM_final.pkl
    model_meta_path: Optional[Path] = None  # e.g., experiments/models/LightGBM_final_meta.json
    scaler_path: Optional[Path] = None  # if separate from model pipeline
    
    # HMM parameters (set by Model Lead after HMM tuning)
    hmm_params_path: Optional[Path] = None  # e.g., experiments/models/hmm_params.json
    
    # Road network artifacts (per-trip, set by Data/ML Lead)
    # These are loaded per-trip at prediction time from the data partition
    data_root: Path = Path("Student folder/Student folder")


@dataclass
class MockConfig:
    """Configuration for deterministic mock prediction mode."""
    enabled: bool = True  # Enable mock mode when real artifacts unavailable
    seed: int = 42


@dataclass
class CORSConfig:
    """CORS configuration for two-laptop demo."""
    allow_origins: list[str] = None  # Defaults to ["*"] for hackathon
    allow_credentials: bool = True
    allow_methods: list[str] = None  # Defaults to ["*"]
    allow_headers: list[str] = None  # Defaults to ["*"]

    def __post_init__(self):
        if self.allow_origins is None:
            self.allow_origins = ["*"]
        if self.allow_methods is None:
            self.allow_methods = ["*"]
        if self.allow_headers is None:
            self.allow_headers = ["*"]


@dataclass
class Settings:
    """Global application settings loaded from environment."""
    api: APIConfig
    artifacts: ArtifactPaths
    mock: MockConfig
    cors: CORSConfig
    
    # Derived: available trip IDs from student_test partition
    student_test_trip_ids: list[str] = None


def _default_real_model_path() -> Path:
    """Default path for the real LightGBM artifact."""
    return Path("experiments/models/LightGBM_final.pkl")


def _default_real_model_meta_path() -> Path:
    """Default path for the real LightGBM metadata."""
    return Path("experiments/models/LightGBM_final_meta.json")


def _default_real_hmm_params_path() -> Path:
    """Default path for the real HMM parameters."""
    return Path("experiments/models/hmm_params.json")


def load_settings() -> Settings:
    """Load all settings from environment variables with defaults.
    
    When MOCK_MODE=false, defaults to real production artifact paths.
    When MOCK_MODE=true, uses synthetic/mock artifacts (for offline tests).
    """
    # Determine if we're in mock mode from env
    mock_enabled = os.getenv("MOCK_MODE", "true").lower() == "true"
    
    # API config
    api = APIConfig(
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
        log_level=os.getenv("API_LOG_LEVEL", "info"),
    )
    
    # Artifact paths - use real artifacts by default when not in mock mode
    data_root = Path(os.getenv("DATA_ROOT", "Student folder/Student folder"))
    
    if mock_enabled:
        # Mock mode: no real artifact paths required
        model_path = _optional_path(os.getenv("MODEL_PATH"))
        model_meta_path = _optional_path(os.getenv("MODEL_META_PATH"))
        hmm_params_path = _optional_path(os.getenv("HMM_PARAMS_PATH"))
    else:
        # Real-artifact mode: default to production paths, allow override via env
        model_path = _optional_path(os.getenv("MODEL_PATH")) or _default_real_model_path()
        model_meta_path = _optional_path(os.getenv("MODEL_META_PATH")) or _default_real_model_meta_path()
        hmm_params_path = _optional_path(os.getenv("HMM_PARAMS_PATH")) or _default_real_hmm_params_path()
    
    artifacts = ArtifactPaths(
        model_path=model_path,
        model_meta_path=model_meta_path,
        scaler_path=_optional_path(os.getenv("SCALER_PATH")),
        hmm_params_path=hmm_params_path,
        data_root=data_root,
    )
    
    # Mock config
    mock = MockConfig(
        enabled=mock_enabled,
        seed=int(os.getenv("MOCK_SEED", "42")),
    )
    
    # CORS config
    cors = CORSConfig(
        allow_origins=_parse_list(os.getenv("CORS_ALLOW_ORIGINS")),
        allow_credentials=os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true",
        allow_methods=_parse_list(os.getenv("CORS_ALLOW_METHODS")),
        allow_headers=_parse_list(os.getenv("CORS_ALLOW_HEADERS")),
    )
    
    return Settings(
        api=api,
        artifacts=artifacts,
        mock=mock,
        cors=cors,
        student_test_trip_ids=None,  # Populated at startup
    )


def _optional_path(value: Optional[str]) -> Optional[Path]:
    """Convert string to Path if provided and non-empty."""
    if value and value.strip():
        return Path(value)
    return None


def _parse_list(value: Optional[str]) -> Optional[list[str]]:
    """Parse comma-separated string into list, or return None."""
    if value and value.strip():
        return [item.strip() for item in value.split(",") if item.strip()]
    return None


def refresh_settings() -> Settings:
    """Reload settings from environment variables.
    
    Useful for tests that need to change environment variables after initial import.
    """
    global settings
    settings = load_settings()
    return settings


# Global settings instance (loaded once at import)
settings = load_settings()