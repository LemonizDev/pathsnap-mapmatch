"""Tests for data loader module."""

import tempfile
import numpy as np
from pathlib import Path
import pytest

from src.data.loader import (
    parse_track_file,
    parse_nodes_file,
    parse_arcs_file,
    parse_route_file,
    load_trip,
    discover_trips,
    DataValidationError,
)


def create_test_trip_dir(base: Path, trip_id: str) -> Path:
    """Create a minimal valid trip directory for testing."""
    trip_dir = base / trip_id
    trip_dir.mkdir(parents=True)
    
    # .track: lon, lat, timestamp
    track_data = """8.595203\t49.844053\t1.0
8.595294\t49.844441\t2.0
8.595352\t49.844883\t3.0"""
    (trip_dir / f"{trip_id}.track").write_text(track_data)
    
    # .nodes: lon, lat
    nodes_data = """8.581679\t49.857880
8.584637\t49.858295
8.580173\t49.858567"""
    (trip_dir / f"{trip_id}.nodes").write_text(nodes_data)
    
    # .arcs: from_node, to_node
    arcs_data = """0\t1
1\t2
2\t0"""
    (trip_dir / f"{trip_id}.arcs").write_text(arcs_data)
    
    # .route: arc_id sequence
    route_data = """0
1
2"""
    (trip_dir / f"{trip_id}.route").write_text(route_data)
    
    return trip_dir


def test_parse_track_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.track"
        path.write_text("8.595203\t49.844053\t1.0\n8.595294\t49.844441\t2.0")
        data = parse_track_file(path)
        assert data.shape == (2, 3)
        assert np.allclose(data[0], [8.595203, 49.844053, 1.0])


def test_parse_nodes_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.nodes"
        path.write_text("8.581679\t49.857880\n8.584637\t49.858295")
        data = parse_nodes_file(path)
        assert data.shape == (2, 2)


def test_parse_arcs_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.arcs"
        path.write_text("0\t1\n1\t2")
        data = parse_arcs_file(path)
        assert data.shape == (2, 2)
        assert data.dtype == np.int64


def test_parse_route_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.route"
        path.write_text("0\n1\n2")
        data = parse_route_file(path)
        assert data.shape == (3,)
        assert data.dtype == np.int64
        assert np.array_equal(data, [0, 1, 2])


def test_load_trip_valid():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        trip_dir = create_test_trip_dir(base, "00000001")
        trip = load_trip(trip_dir, "00000001")
        
        assert trip.trip_id == "00000001"
        assert trip.track.shape == (3, 3)
        assert trip.nodes.shape == (3, 2)
        assert trip.arcs.shape == (3, 2)
        assert trip.route.shape == (3,)


def test_load_trip_invalid_coordinates():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        trip_dir = base / "00000001"
        trip_dir.mkdir()
        
        # Invalid longitude (outside WGS84 [-180, 180])
        (trip_dir / "00000001.track").write_text("200.0\t49.844053\t1.0")
        (trip_dir / "00000001.nodes").write_text("8.581679\t49.857880")
        (trip_dir / "00000001.arcs").write_text("0\t0")
        (trip_dir / "00000001.route").write_text("0")
        
        with pytest.raises(DataValidationError, match="longitude.*out of bounds"):
            load_trip(trip_dir, "00000001")


def test_load_trip_non_monotonic_timestamps():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        trip_dir = base / "00000001"
        trip_dir.mkdir()
        
        # Non-monotonic timestamps
        (trip_dir / "00000001.track").write_text("8.595203\t49.844053\t2.0\n8.595294\t49.844441\t1.0")
        (trip_dir / "00000001.nodes").write_text("8.581679\t49.857880")
        (trip_dir / "00000001.arcs").write_text("0\t0")
        (trip_dir / "00000001.route").write_text("0")
        
        with pytest.raises(DataValidationError, match="timestamps not monotonically increasing"):
            load_trip(trip_dir, "00000001")


def test_load_trip_invalid_arc_reference():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        trip_dir = base / "00000001"
        trip_dir.mkdir()
        
        (trip_dir / "00000001.track").write_text("8.595203\t49.844053\t1.0")
        (trip_dir / "00000001.nodes").write_text("8.581679\t49.857880")  # Only 1 node (id 0)
        (trip_dir / "00000001.arcs").write_text("0\t5")  # References node 5 which doesn't exist
        (trip_dir / "00000001.route").write_text("0")
        
        with pytest.raises(DataValidationError, match="invalid.*to_node"):
            load_trip(trip_dir, "00000001")


def test_load_trip_invalid_route_reference():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        trip_dir = base / "00000001"
        trip_dir.mkdir()
        
        (trip_dir / "00000001.track").write_text("8.595203\t49.844053\t1.0")
        (trip_dir / "00000001.nodes").write_text("8.581679\t49.857880")
        (trip_dir / "00000001.arcs").write_text("0\t0")  # Only 1 arc (id 0)
        (trip_dir / "00000001.route").write_text("5")  # References arc 5 which doesn't exist
        
        with pytest.raises(DataValidationError, match="invalid arc_id"):
            load_trip(trip_dir, "00000001")


def test_discover_trips():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        create_test_trip_dir(base, "00000001")
        create_test_trip_dir(base, "00000002")
        create_test_trip_dir(base, "00000003")
        
        # Create a directory without required files (should be skipped)
        (base / "invalid").mkdir()
        
        dirs, ids = discover_trips(base)
        assert len(dirs) == 3
        assert ids == ["00000001", "00000002", "00000003"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])