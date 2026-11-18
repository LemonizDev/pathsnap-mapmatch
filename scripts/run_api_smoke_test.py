#!/usr/bin/env python
"""API smoke test for PathSnap FastAPI backend.

Tests the API endpoints using TestClient (in-process) and optionally
against a running server via HTTP.

Usage:
    # In-process test (default)
    python scripts/run_api_smoke_test.py
    
    # Against running server
    python scripts/run_api_smoke_test.py --url http://localhost:8000
    
    # Against LAN IP (two-laptop demo)
    python scripts/run_api_smoke_test.py --url http://192.168.1.47:8000
"""

from __future__ import annotations

import os
# Set OpenBLAS threads BEFORE any numpy/scipy imports to avoid memory allocation issues
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import requests

# Add src to path for in-process testing
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_in_process_test() -> dict:
    """Run smoke test using FastAPI TestClient (in-process)."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    
    print("=" * 60)
    print("PathSnap API Smoke Test (In-Process)")
    print("=" * 60)
    
    # Use context manager to ensure lifespan runs (startup/shutdown events)
    with TestClient(app) as client:
        results = {"mode": "in_process", "tests": {}}
        
        # Test 1: Health endpoint
        print("\n[1/5] Testing GET /health...")
        start = time.perf_counter()
        response = client.get("/health")
        elapsed = (time.perf_counter() - start) * 1000
        
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        health_data = response.json()
        
        print(f"  Status: {health_data['status']}")
        print(f"  Model loaded: {health_data['model_loaded']}")
        print(f"  HMM loaded: {health_data['hmm_loaded']}")
        print(f"  Trips available: {health_data['trips_available']}")
        print(f"  Mock mode: {health_data['mock_mode']}")
        print(f"  Response time: {elapsed:.1f}ms")
        
        results["tests"]["health"] = {
            "status_code": response.status_code,
            "response_time_ms": elapsed,
            "data": health_data,
        }
        
        # Test 2: Trips endpoint
        print("\n[2/5] Testing GET /trips...")
        start = time.perf_counter()
        response = client.get("/trips")
        elapsed = (time.perf_counter() - start) * 1000
        
        assert response.status_code == 200, f"Trips list failed: {response.status_code}"
        trips_data = response.json()
        
        print(f"  Partition: {trips_data['partition']}")
        print(f"  Count: {trips_data['count']}")
        print(f"  Trip IDs: {trips_data['trips'][:5]}{'...' if len(trips_data['trips']) > 5 else ''}")
        print(f"  Response time: {elapsed:.1f}ms")
        
        results["tests"]["trips"] = {
            "status_code": response.status_code,
            "response_time_ms": elapsed,
            "data": trips_data,
        }
        
        if trips_data["count"] == 0:
            print("\n  WARNING: No student_test trips found. Skipping predict test.")
            results["tests"]["predict"] = {"skipped": True, "reason": "no_trips_available"}
            return results
        
        # Test 3: Predict endpoint - valid trip
        trip_id = trips_data["trips"][0]
        print(f"\n[3/5] Testing POST /predict with trip_id={trip_id}...")
        start = time.perf_counter()
        response = client.post("/predict", json={"trip_id": trip_id})
        elapsed = (time.perf_counter() - start) * 1000
        
        assert response.status_code == 200, f"Predict failed: {response.status_code} - {response.text}"
        predict_data = response.json()
        
        print(f"  Predicted route length: {len(predict_data['predicted_route'])} arcs")
        print(f"  Accuracy: {predict_data['accuracy']:.2%}")
        print(f"  Map HTML length: {len(predict_data['map_html'])} chars")
        print(f"  Latencies (ms): {predict_data['latencies_ms']}")
        print(f"  Response time: {elapsed:.1f}ms")
        
        # Validate response structure
        required_keys = ["trip_id", "predicted_route", "accuracy", "map_html", "latencies_ms"]
        for key in required_keys:
            assert key in predict_data, f"Missing key in response: {key}"
        
        assert predict_data["trip_id"] == trip_id
        assert isinstance(predict_data["predicted_route"], list)
        assert all(isinstance(x, int) for x in predict_data["predicted_route"])
        assert 0.0 <= predict_data["accuracy"] <= 1.0
        assert isinstance(predict_data["map_html"], str)
        assert len(predict_data["map_html"]) > 100  # Should have substantial HTML
        
        latencies = predict_data["latencies_ms"]
        for stage in ["candidates", "model", "hmm", "total"]:
            assert stage in latencies, f"Missing latency stage: {stage}"
            assert isinstance(latencies[stage], int)
            assert latencies[stage] >= 0
        
        results["tests"]["predict"] = {
            "status_code": response.status_code,
            "response_time_ms": elapsed,
            "trip_id": trip_id,
            "predicted_route_length": len(predict_data["predicted_route"]),
            "accuracy": predict_data["accuracy"],
            "latencies_ms": latencies,
        }
        
        # Test 4: Predict determinism (mock mode)
        print(f"\n[4/5] Testing mock mode determinism...")
        response2 = client.post("/predict", json={"trip_id": trip_id})
        assert response2.status_code == 200
        predict_data2 = response2.json()
        
        assert predict_data["predicted_route"] == predict_data2["predicted_route"], \
            "Mock predictions should be deterministic"
        assert predict_data["accuracy"] == predict_data2["accuracy"]
        assert predict_data["map_html"] == predict_data2["map_html"]
        
        print("  Predictions are deterministic: PASSED")
        results["tests"]["determinism"] = {"passed": True}
        
        # Test 5: Predict 404 for invalid trip
        print(f"\n[5/5] Testing 404 for invalid trip_id...")
        response = client.post("/predict", json={"trip_id": "99999999"})
        assert response.status_code == 404, f"Expected 404, got {response.status_code}"
        
        error_data = response.json()
        assert error_data["error"] == "trip_not_found"
        assert "available_trips" in error_data.get("detail", {})
        
        print("  404 response structure: PASSED")
        results["tests"]["not_found"] = {
            "status_code": response.status_code,
            "error_code": error_data["error"],
        }
        
        print("\n" + "=" * 60)
        print("ALL SMOKE TESTS PASSED")
        print("=" * 60)
        
        return results


def run_remote_test(base_url: str) -> dict:
    """Run smoke test against a running server via HTTP."""
    print("=" * 60)
    print(f"PathSnap API Smoke Test (Remote: {base_url})")
    print("=" * 60)
    
    results = {"mode": "remote", "base_url": base_url, "tests": {}}
    
    # Test 1: Health endpoint
    print("\n[1/5] Testing GET /health...")
    start = time.perf_counter()
    response = requests.get(f"{base_url}/health", timeout=10)
    elapsed = (time.perf_counter() - start) * 1000
    
    assert response.status_code == 200, f"Health check failed: {response.status_code}"
    health_data = response.json()
    
    print(f"  Status: {health_data['status']}")
    print(f"  Model loaded: {health_data['model_loaded']}")
    print(f"  HMM loaded: {health_data['hmm_loaded']}")
    print(f"  Trips available: {health_data['trips_available']}")
    print(f"  Mock mode: {health_data['mock_mode']}")
    print(f"  Response time: {elapsed:.1f}ms")
    
    results["tests"]["health"] = {
        "status_code": response.status_code,
        "response_time_ms": elapsed,
        "data": health_data,
    }
    
    # Test 2: Trips endpoint
    print("\n[2/5] Testing GET /trips...")
    start = time.perf_counter()
    response = requests.get(f"{base_url}/trips", timeout=10)
    elapsed = (time.perf_counter() - start) * 1000
    
    assert response.status_code == 200, f"Trips list failed: {response.status_code}"
    trips_data = response.json()
    
    print(f"  Partition: {trips_data['partition']}")
    print(f"  Count: {trips_data['count']}")
    print(f"  Trip IDs: {trips_data['trips'][:5]}{'...' if len(trips_data['trips']) > 5 else ''}")
    print(f"  Response time: {elapsed:.1f}ms")
    
    results["tests"]["trips"] = {
        "status_code": response.status_code,
        "response_time_ms": elapsed,
        "data": trips_data,
    }
    
    if trips_data["count"] == 0:
        print("\n  WARNING: No student_test trips found. Skipping predict test.")
        results["tests"]["predict"] = {"skipped": True, "reason": "no_trips_available"}
        return results
    
    # Test 3: Predict endpoint
    trip_id = trips_data["trips"][0]
    print(f"\n[3/5] Testing POST /predict with trip_id={trip_id}...")
    start = time.perf_counter()
    response = requests.post(
        f"{base_url}/predict",
        json={"trip_id": trip_id},
        timeout=30,
    )
    elapsed = (time.perf_counter() - start) * 1000
    
    assert response.status_code == 200, f"Predict failed: {response.status_code} - {response.text}"
    predict_data = response.json()
    
    print(f"  Predicted route length: {len(predict_data['predicted_route'])} arcs")
    print(f"  Accuracy: {predict_data['accuracy']:.2%}")
    print(f"  Map HTML length: {len(predict_data['map_html'])} chars")
    print(f"  Latencies (ms): {predict_data['latencies_ms']}")
    print(f"  Response time: {elapsed:.1f}ms")
    
    # Validate response structure
    required_keys = ["trip_id", "predicted_route", "accuracy", "map_html", "latencies_ms"]
    for key in required_keys:
        assert key in predict_data, f"Missing key in response: {key}"
    
    results["tests"]["predict"] = {
        "status_code": response.status_code,
        "response_time_ms": elapsed,
        "trip_id": trip_id,
        "predicted_route_length": len(predict_data["predicted_route"]),
        "accuracy": predict_data["accuracy"],
        "latencies_ms": predict_data["latencies_ms"],
    }
    
    # Test 4: Predict determinism
    print(f"\n[4/5] Testing mock mode determinism...")
    response2 = requests.post(f"{base_url}/predict", json={"trip_id": trip_id}, timeout=30)
    assert response2.status_code == 200
    predict_data2 = response2.json()
    
    assert predict_data["predicted_route"] == predict_data2["predicted_route"]
    assert predict_data["accuracy"] == predict_data2["accuracy"]
    
    print("  Predictions are deterministic: PASSED")
    results["tests"]["determinism"] = {"passed": True}
    
    # Test 5: 404 for invalid trip
    print(f"\n[5/5] Testing 404 for invalid trip_id...")
    response = requests.post(f"{base_url}/predict", json={"trip_id": "99999999"}, timeout=10)
    assert response.status_code == 404, f"Expected 404, got {response.status_code}"
    
    error_data = response.json()
    assert error_data["error"] == "trip_not_found"
    
    print("  404 response structure: PASSED")
    results["tests"]["not_found"] = {
        "status_code": response.status_code,
        "error_code": error_data["error"],
    }
    
    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="PathSnap API smoke test")
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Base URL of running API server (e.g., http://localhost:8000). If omitted, runs in-process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/api_smoke_test_results.json"),
        help="Output JSON file for results",
    )
    args = parser.parse_args()
    
    try:
        if args.url:
            results = run_remote_test(args.url.rstrip("/"))
        else:
            results = run_in_process_test()
        
        # Save results
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {args.output}")
        return 0
        
    except Exception as e:
        print(f"\nSMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())