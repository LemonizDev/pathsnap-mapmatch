"""Contract tests for the PathSnap Streamlit frontend (``src/ui/app.py``).

These tests exercise the HTTP client functions against a mocked
``urllib.request.urlopen`` so they run without a real backend. They verify the
request shape sent to ``POST /predict`` and the response parsing, plus the
offline demo fallback.
"""

from __future__ import annotations

import json
from unittest import mock

import pytest

from src.ui.app import (
    STUDENT_TEST_TRIPS,
    api_predict,
    api_base_url,
    mock_predict,
)


DOCUMENTED_STUDENT_TEST_TRIPS = [
    "00000000",
    "00000007",
    "00000009",
    "00000040",
    "00000052",
    "00000056",
    "00000065",
    "00000076",
    "00000077",
    "00000091",
]


class _FakeResponse:
    """Minimal context-manager stand-in for an urlopen response."""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


def test_student_test_trips_match_documentation():
    """The UI trip selector exposes exactly the documented student-test IDs."""
    assert STUDENT_TEST_TRIPS == DOCUMENTED_STUDENT_TEST_TRIPS


def test_api_predict_posts_correct_payload_and_parses_response():
    """POST /predict receives {trip_id} and the JSON response is parsed."""
    expected = {
        "trip_id": "00000000",
        "predicted_route": [401, 402, 403],
        "accuracy": 0.87,
        "map_html": "<html></html>",
        "latencies_ms": {"candidates": 45, "model": 12, "hmm": 8, "total": 78},
    }
    with mock.patch(
        "urllib.request.urlopen", return_value=_FakeResponse(expected)
    ) as m:
        result = api_predict("00000000")

    assert result == expected
    request = m.call_args[0][0]
    assert request.full_url == "http://localhost:8000/predict"
    assert request.method == "POST"
    # urllib normalizes the header key to "Content-type".
    assert request.headers.get("Content-type") == "application/json"
    assert json.loads(request.data) == {"trip_id": "00000000"}


def test_api_predict_uses_configured_base_url(monkeypatch):
    """The request URL honors API_BASE_URL."""
    import src.ui.app as app

    monkeypatch.setattr(app, "API_BASE_URL", "http://192.168.1.47:8000")
    expected = {
        "trip_id": "00000007",
        "predicted_route": [1],
        "accuracy": 0.5,
        "map_html": "",
        "latencies_ms": {},
    }
    with mock.patch(
        "urllib.request.urlopen", return_value=_FakeResponse(expected)
    ) as m:
        api_predict("00000007")
    assert m.call_args[0][0].full_url == "http://192.168.1.47:8000/predict"


def test_api_predict_raises_when_backend_unreachable():
    """A connection failure propagates so the UI can show a clear error."""
    with mock.patch(
        "urllib.request.urlopen",
        side_effect=__import__("urllib.error").error.URLError("connection refused"),
    ):
        with pytest.raises(__import__("urllib.error").error.URLError):
            api_predict("00000000")


def test_api_predict_raises_on_http_error():
    """Non-2xx responses raise HTTPError for the UI to display."""
    import urllib.error

    with mock.patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(
            "http://localhost:8000/predict", 500, "Internal Server Error", {}, None
        ),
    ):
        with pytest.raises(urllib.error.HTTPError):
            api_predict("00000000")


def test_mock_predict_is_deterministic_and_offline():
    """Demo predictions are stable per trip and never touch the network."""
    with mock.patch("urllib.request.urlopen") as m:
        first = mock_predict("00000000")
        second = mock_predict("00000000")
    m.assert_not_called()
    assert first["trip_id"] == "00000000"
    assert isinstance(first["predicted_route"], list) and len(first["predicted_route"]) > 0
    assert 0.0 <= first["accuracy"] <= 1.0
    assert isinstance(first["map_html"], str) and "<html" in first["map_html"]
    assert set(first["latencies_ms"]) == {"candidates", "model", "hmm", "total"}
    # Folium assigns a unique div id per map, so compare everything but map_html
    # byte-for-byte; map_html should be stable in structure across calls.
    without_map = lambda d: {k: v for k, v in d.items() if k != "map_html"}
    assert without_map(first) == without_map(second)
    assert "Predicted route (demo / mock)" in first["map_html"]
    assert "Predicted route (demo / mock)" in second["map_html"]


def test_api_base_url_reads_env_var(monkeypatch):
    """api_base_url() reflects the API_BASE_URL environment variable."""
    monkeypatch.setenv("API_BASE_URL", "http://192.168.1.5:8000")
    # Re-import not needed; read the module attribute directly after env set.
    import importlib

    import src.ui.app as app

    importlib.reload(app)
    assert app.api_base_url() == "http://192.168.1.5:8000"
