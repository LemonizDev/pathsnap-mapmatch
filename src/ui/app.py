"""PathSnap frontend demo — Streamlit UI.

Calls the FastAPI backend contract documented in ``context/initial/ARCHITECTURE.md``:

    GET  /health  ->  { "status", "model_loaded", "hmm_loaded", "trips_available" }
    POST /predict { "trip_id": "..." }  ->
        { "trip_id", "predicted_route": [int], "accuracy": float,
          "map_html": str, "latencies_ms": { "candidates", "model", "hmm", "total" } }

The backend URL is configured through the ``API_BASE_URL`` environment variable
(default ``http://localhost:8000``).

A deterministic demo/mock fallback mode runs without any backend so the UI can
be developed and demonstrated locally. Ground truth is never exposed: the UI
only displays the server-computed accuracy and predicted route.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from typing import Any

import folium
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")

# The 10 student-test trip IDs from ARCHITECTURE.md (Data Loader partitions).
STUDENT_TEST_TRIPS = [
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

PREDICT_TIMEOUT = 30
HEALTH_TIMEOUT = 5


def api_base_url() -> str:
    """Return the configured backend base URL (from API_BASE_URL env var)."""
    return API_BASE_URL


def api_health(timeout: float = HEALTH_TIMEOUT) -> dict[str, Any] | None:
    """Query GET /health. Returns None when the backend is unreachable."""
    url = f"{API_BASE_URL}/health"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def api_predict(trip_id: str, timeout: float = PREDICT_TIMEOUT) -> dict[str, Any]:
    """POST {trip_id} to /predict and return the parsed JSON response."""
    url = f"{API_BASE_URL}/predict"
    body = json.dumps({"trip_id": trip_id}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def mock_predict(trip_id: str) -> dict[str, Any]:
    """Deterministic demo prediction that never contacts the backend.

    Used for local UI development and as a fallback when the backend is
    unavailable. Output is stable for a given trip_id so the demo is
    reproducible.
    """
    seed = int(hashlib.md5(trip_id.encode()).hexdigest(), 16)
    n_arcs = 10 + (seed % 15)
    route = [(seed * 31 + i * 7) % 5000 for i in range(n_arcs)]
    accuracy = round(0.70 + (seed % 30) / 100, 2)
    latencies_ms = {"candidates": 42, "model": 18, "hmm": 9, "total": 69}
    return {
        "trip_id": trip_id,
        "predicted_route": route,
        "accuracy": accuracy,
        "map_html": _mock_map_html(trip_id),
        "latencies_ms": latencies_ms,
    }


def _mock_map_html(trip_id: str) -> str:
    """Build a small Folium map for demo mode (no real road geometry)."""
    seed = int(hashlib.md5(trip_id.encode()).hexdigest(), 16)
    base_lat = 49.0 + (seed % 200) / 1000.0
    base_lon = 8.0 + (seed % 1500) / 1000.0
    coords = [
        (base_lat + i * 0.002, base_lon + i * 0.0015) for i in range(9)
    ]
    m = folium.Map(location=coords[4], zoom_start=14)
    folium.PolyLine(
        coords,
        color="green",
        weight=5,
        opacity=0.8,
        tooltip="Predicted route (demo / mock)",
    ).add_to(m)
    folium.Marker(coords[0], tooltip="Trip start (demo)").add_to(m)
    folium.Marker(coords[-1], tooltip="Trip end (demo)").add_to(m)
    return m.get_root().render()


def render_map(map_html: str, height: int = 600) -> None:
    """Render a Folium HTML map string inside Streamlit."""
    if not map_html:
        st.info("No map data available from the backend.")
        return
    st.components.v1.html(map_html, height=height, scrolling=True)


def display_result(result: dict[str, Any], demo: bool) -> None:
    """Show predicted route, accuracy, latencies, and the map."""
    st.subheader(f"Trip {result.get('trip_id')}")
    if demo:
        st.info("Demo mode — mock prediction, not from the backend.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted route length", f"{len(result.get('predicted_route', []))} arcs")
    c2.metric("Accuracy", f"{result.get('accuracy', 0) * 100:.1f}%")
    c3.metric("Total latency (ms)", result.get("latencies_ms", {}).get("total", "-"))

    with st.expander("Predicted route (arc IDs)", expanded=False):
        route = result.get("predicted_route", [])
        st.write(route)

    with st.expander("Latency breakdown (ms)", expanded=False):
        st.json(result.get("latencies_ms", {}))

    st.subheader("Route map")
    render_map(result.get("map_html", ""))


def main() -> None:
    st.set_page_config(
        page_title="PathSnap — Route Reconstruction",
        page_icon="🗺️",
        layout="wide",
    )

    st.title("PathSnap — GPS Trajectory Map-Matching")
    st.caption(
        "Reconstruct the most likely road route from a noisy GPS trajectory. "
        "Backend: FastAPI ``POST /predict``."
    )

    with st.sidebar:
        st.header("Connection")
        st.code(f"API_BASE_URL = {API_BASE_URL}", language="bash")
        demo_mode = st.toggle(
            "Demo mode (no backend)",
            value=st.session_state.get("demo_mode", False),
            help="Use deterministic mock predictions without calling the backend.",
        )
        st.session_state["demo_mode"] = demo_mode

        if st.button("Check backend health", use_container_width=True, disabled=demo_mode):
            try:
                health = api_health()
                st.session_state["health"] = health
                st.session_state["health_error"] = None
            except Exception as exc:  # noqa: BLE001 — surface any backend failure
                st.session_state["health"] = None
                st.session_state["health_error"] = str(exc)

        health = st.session_state.get("health")
        health_error = st.session_state.get("health_error")
        if health_error:
            st.error(f"Backend unreachable: {health_error}")
        elif health:
            st.success(
                "Backend healthy "
                f"(model={health.get('model_loaded')}, "
                f"hmm={health.get('hmm_loaded')}, "
                f"trips={health.get('trips_available')})"
            )

    st.divider()

    col_trip, col_predict = st.columns([3, 1])
    with col_trip:
        trip_id = st.selectbox(
            "Student-test trip ID",
            options=STUDENT_TEST_TRIPS,
            index=0,
            help="One of the 10 student-test trip IDs from the dataset partition.",
        )
    with col_predict:
        predict_clicked = st.button(
            "Predict",
            type="primary",
            use_container_width=True,
        )

    if predict_clicked:
        if demo_mode:
            result = mock_predict(trip_id)
            display_result(result, demo=True)
        else:
            try:
                result = api_predict(trip_id)
                display_result(result, demo=False)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                st.error(
                    f"Backend returned HTTP {exc.code} for trip {trip_id}.\n\n{body}"
                )
                st.info("Enable 'Demo mode (no backend)' in the sidebar to run without the backend.")
            except urllib.error.URLError as exc:
                st.error(
                    f"Could not reach the backend at {API_BASE_URL}.\n\n{exc}"
                )
                st.info("Enable 'Demo mode (no backend)' in the sidebar to run without the backend.")
            except Exception as exc:  # noqa: BLE001 — surface unexpected failures clearly
                st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
