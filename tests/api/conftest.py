"""Pytest configuration for PathSnap API tests.

Provides proper environment isolation for mock-mode vs real-artifact tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Global thread limits
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Default to real-artifact mode
os.environ["MOCK_MODE"] = "false"
os.environ["DATA_ROOT"] = str(Path(__file__).parent.parent / "Student folder" / "Student folder")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


# Mock-mode tests fixture
@pytest.fixture(scope="function")
def mock_mode_client():
    """Create TestClient for API testing in MOCK_MODE."""
    import os
    from fastapi.testclient import TestClient
    
    # Set mock mode
    os.environ["MOCK_MODE"] = "true"
    os.environ["MOCK_SEED"] = "42"
    
    # Reset artifact loader to ensure clean state
    from src.api.load_artifacts import reset_artifact_loader
    from src.api.config import refresh_settings
    reset_artifact_loader()
    refresh_settings()
    
    # Import app fresh (will pick up MOCK_MODE=true)
    from src.api.main import app
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Cleanup: reset to real-artifact defaults
    os.environ["MOCK_MODE"] = "false"
    os.environ.pop("MOCK_SEED", None)
    from src.api.load_artifacts import reset_artifact_loader
    from src.api.config import refresh_settings
    reset_artifact_loader()
    refresh_settings()


# Real-artifact tests fixture
@pytest.fixture(scope="function")
def real_artifact_client():
    """Create TestClient for API testing with REAL artifacts (MOCK_MODE=false)."""
    import os
    from fastapi.testclient import TestClient
    
    os.environ["MOCK_MODE"] = "false"
    os.environ["DATA_ROOT"] = str(Path(__file__).parent.parent / "Student folder" / "Student folder")
    os.environ["MODEL_PATH"] = str(Path(__file__).parent.parent / "experiments" / "models" / "LightGBM_final.pkl")
    os.environ["MODEL_META_PATH"] = str(Path(__file__).parent.parent / "experiments" / "models" / "LightGBM_final_meta.json")
    os.environ["HMM_PARAMS_PATH"] = str(Path(__file__).parent.parent / "experiments" / "models" / "hmm_params.json")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    
    # Reset artifact loader to ensure clean state
    from src.api.load_artifacts import reset_artifact_loader
    from src.api.config import refresh_settings
    reset_artifact_loader()
    refresh_settings()
    
    # Import app fresh (will pick up MOCK_MODE=false)
    from src.api.main import app
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Cleanup
    from src.api.load_artifacts import reset_artifact_loader
    from src.api.config import refresh_settings
    reset_artifact_loader()
    refresh_settings()