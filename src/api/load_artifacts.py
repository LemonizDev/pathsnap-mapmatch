"""Artifact loading with dependency injection for PathSnap API.

Loads ML model, HMM parameters, and road network data at startup.
Provides clear behavior when artifacts are unavailable (falls back to mock mode).
Integrates with ModelInterface and HMM configs from src/hmm/.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from src.api.config import settings, ArtifactPaths
from src.data.loader import TripData, load_trip, discover_trips
from src.features.candidates import (
    Candidate,
    CandidateSet,
    generate_candidates_for_trip,
    compute_arc_geometry,
)
from src.features.engineer import (
    build_feature_matrix,
    compute_arc_lengths,
    compute_node_degrees,
    compute_adjacency,
)

# HMM imports
from src.hmm.model_interface import ModelInterface, ModelProbabilities
from src.hmm.emission import EmissionConfig
from src.hmm.transition import TransitionConfig
from src.hmm.viterbi import ViterbiConfig, viterbi_decode, reconstruct_path_from_result


@dataclass
class ModelArtifacts:
    """Loaded ML model artifacts."""
    model: Any = None
    model_interface: Optional[ModelInterface] = None
    feature_names: List[str] = None
    model_name: str = "unknown"
    loaded: bool = False
    load_error: Optional[str] = None
    source: str = "synthetic"


@dataclass
class HMMParams:
    """HMM/Viterbi parameters loaded from JSON."""
    distance_penalty_lambda: float = 0.05
    min_log_prob: float = -100.0
    max_log_prob: float = 0.0
    hop_penalty: float = 1.0
    gap_penalty: float = 10.0
    max_hops: int = 3
    stay_bonus: float = 0.0
    initial_log_prob: float = 0.0
    loaded: bool = False
    load_error: Optional[str] = None
    source: str = "defaulted"

    def to_emission_config(self) -> EmissionConfig:
        return EmissionConfig(
            distance_penalty_lambda=self.distance_penalty_lambda,
            min_log_prob=self.min_log_prob,
            max_log_prob=self.max_log_prob,
        )

    def to_transition_config(self) -> TransitionConfig:
        return TransitionConfig(
            hop_penalty=self.hop_penalty,
            gap_penalty=self.gap_penalty,
            max_hops=self.max_hops,
            stay_bonus=self.stay_bonus,
        )

    def to_viterbi_config(self) -> ViterbiConfig:
        return ViterbiConfig(initial_log_prob=self.initial_log_prob)


@dataclass
class TripArtifacts:
    """Per-trip loaded data (loaded on-demand)."""
    trip_data: TripData
    arc_midpoints: np.ndarray
    arc_start_points: np.ndarray
    arc_end_points: np.ndarray
    arc_lengths: np.ndarray
    in_degree: np.ndarray
    out_degree: np.ndarray
    adjacency: Dict[int, List[int]]


class SklearnModelInterface(ModelInterface):
    """Adapter wrapping a sklearn Pipeline/classifier to implement ModelInterface.

    Expects the model to have:
    - predict_proba(X) -> (n_samples, 2) array
    - predict_log_proba(X) -> (n_samples, 2) array
    - Optionally: named_steps['scaler'] for Pipeline with scaler
    """

    def __init__(self, model: Any, feature_names: List[str]):
        self._model = model
        self._feature_names = feature_names

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @property
    def n_features(self) -> int:
        return len(self._feature_names)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self._model, "predict_log_proba"):
            return self._model.predict_log_proba(X)
        proba = self._model.predict_proba(X)
        return np.log(np.clip(proba, 1e-15, 1.0))


class MockModelInterface(ModelInterface):
    """Deterministic mock model for testing HMM without a trained classifier.

    Produces probabilities based on distance and candidate rank:
    - Lower perpendicular distance -> higher probability
    - Lower candidate rank -> higher probability
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self._feature_names = feature_names or [
            "perp_distance_m",
            "along_track_dist_m",
            "bearing_diff_deg",
            "speed_m_s",
            "time_delta_s",
            "arc_length_m",
            "from_node_out_degree",
            "to_node_in_degree",
            "arc_connectivity",
            "candidate_rank",
            "is_continuation",
        ]

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @property
    def n_features(self) -> int:
        return len(self._feature_names)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dist_score = np.exp(-X[:, 0] / 50.0)
        rank_score = 1.0 / np.maximum(X[:, 9], 1.0)
        bearing_score = np.exp(-X[:, 2] / 45.0)

        score = 0.5 * dist_score + 0.3 * rank_score + 0.2 * bearing_score
        prob = 1.0 / (1.0 + np.exp(-10.0 * (score - 0.5)))
        prob = np.clip(prob, 1e-10, 1 - 1e-10)

        return np.column_stack([1 - prob, prob])

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.log(proba)


class ArtifactLoader:
    """Loads and caches all artifacts needed for prediction.

    Usage:
        loader = ArtifactLoader()
        loader.load_all()  # At startup

        # At prediction time:
        trip_artifacts = loader.get_trip_artifacts(trip_id)
        model_interface = loader.model_artifacts.model_interface
        hmm_params = loader.hmm_params
    """

    def __init__(
        self,
        artifact_paths: ArtifactPaths = None,
        mock_mode: bool = None,
        mock_seed: int = 42
    ):
        self.paths = artifact_paths or settings.artifacts
        self.mock_mode = mock_mode if mock_mode is not None else settings.mock_enabled
        self.mock_seed = mock_seed

        self.model_artifacts = ModelArtifacts()
        self.hmm_params = HMMParams()
        self._trip_cache: Dict[str, TripArtifacts] = {}
        self._student_test_trip_ids: List[str] = []

    def load_all(self) -> None:
        """Load all artifacts at startup. Populates student_test_trip_ids."""
        self._load_model()
        self._load_hmm_params()
        self._discover_student_test_trips()

    def _load_model(self) -> None:
        if self.mock_mode:
            warnings.warn("MOCK_MODE=true: Skipping real model artifact loading.")
            self._init_mock_model()
            return

        model_path = self.paths.model_path
        meta_path = self.paths.model_meta_path

        if not model_path or not model_path.exists():
            self.model_artifacts.load_error = f"Model file not found: {model_path}"
            raise FileNotFoundError(
                f"Required model artifact not found: {model_path}. "
                f"Set MODEL_PATH environment variable or enable MOCK_MODE=true"
            )

        try:
            model_obj = joblib.load(model_path)

            feature_names = None
            model_name = "unknown"
            if meta_path and meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                feature_names = meta.get("feature_names")
                model_name = meta.get("model_name", "unknown")

            if feature_names is None:
                feature_names = [
                    "perp_distance_m",
                    "along_track_dist_m",
                    "bearing_diff_deg",
                    "speed_m_s",
                    "time_delta_s",
                    "arc_length_m",
                    "from_node_out_degree",
                    "to_node_in_degree",
                    "arc_connectivity",
                    "candidate_rank",
                    "is_continuation",
                ]
                warnings.warn(
                    f"No model metadata found at {meta_path}. "
                    f"Using default feature names. "
                    f"Ensure model expects these 11 features in this order."
                )

            model_interface = SklearnModelInterface(model_obj, feature_names)

            if model_interface.n_features != 11:
                warnings.warn(
                    f"Model expects {model_interface.n_features} features, "
                    f"but pipeline produces 11. This may cause errors."
                )

            self.model_artifacts = ModelArtifacts(
                model=model_obj,
                model_interface=model_interface,
                feature_names=feature_names,
                model_name=model_name,
                loaded=True,
                load_error=None,
                source="real",
            )
            print(f"Loaded model: {model_name} from {model_path}")
            print(f"  Feature names: {feature_names}")

        except Exception as e:
            self.model_artifacts.load_error = f"Failed to load model: {e}"
            raise RuntimeError(f"Failed to load model from {model_path}: {e}") from e

    def _init_mock_model(self) -> None:
        self.model_artifacts = ModelArtifacts(
            model=None,
            model_interface=MockModelInterface(),
            feature_names=[
                "perp_distance_m",
                "along_track_dist_m",
                "bearing_diff_deg",
                "speed_m_s",
                "time_delta_s",
                "arc_length_m",
                "from_node_out_degree",
                "to_node_in_degree",
                "arc_connectivity",
                "candidate_rank",
                "is_continuation",
            ],
            model_name="mock",
            loaded=False,
            load_error=self.model_artifacts.load_error,
            source="synthetic",
        )

    def _load_hmm_params(self) -> None:
        hmm_path = self.paths.hmm_params_path

        if not hmm_path or not hmm_path.exists():
            self.hmm_params.load_error = f"HMM params file not found: {hmm_path}"
            if not self.mock_mode:
                raise FileNotFoundError(
                    f"Required HMM params not found: {hmm_path}. "
                    f"Set HMM_PARAMS_PATH environment variable or enable MOCK_MODE=true"
                )
            self.hmm_params = HMMParams(
                distance_penalty_lambda=0.05,
                min_log_prob=-100.0,
                max_log_prob=0.0,
                hop_penalty=1.0,
                gap_penalty=10.0,
                max_hops=3,
                stay_bonus=0.0,
                initial_log_prob=0.0,
                loaded=False,
                load_error=self.hmm_params.load_error,
                source="synthetic",
            )
            warnings.warn(f"HMM params not found at {hmm_path}. Using synthetic defaults.")
            return

        try:
            with open(hmm_path, "r") as f:
                params = json.load(f)

            self.hmm_params = HMMParams(
                distance_penalty_lambda=params.get("distance_penalty_lambda", 0.05),
                min_log_prob=params.get("min_log_prob", -100.0),
                max_log_prob=params.get("max_log_prob", 0.0),
                hop_penalty=params.get("hop_penalty", 1.0),
                gap_penalty=params.get("gap_penalty", 10.0),
                max_hops=params.get("max_hops", 3),
                stay_bonus=params.get("stay_bonus", 0.0),
                initial_log_prob=params.get("initial_log_prob", 0.0),
                loaded=True,
                load_error=None,
                source="real",
            )
            print(f"Loaded HMM params from {hmm_path}: {params}")

        except Exception as e:
            self.hmm_params.load_error = f"Failed to load HMM params: {e}"
            if not self.mock_mode:
                raise RuntimeError(f"Failed to load HMM params from {hmm_path}: {e}") from e
            self.hmm_params = HMMParams(
                distance_penalty_lambda=0.05,
                min_log_prob=-100.0,
                max_log_prob=0.0,
                hop_penalty=1.0,
                gap_penalty=10.0,
                max_hops=3,
                stay_bonus=0.0,
                initial_log_prob=0.0,
                loaded=False,
                load_error=self.hmm_params.load_error,
                source="synthetic",
            )
            warnings.warn(f"HMM params loading failed: {e}. Using synthetic defaults.")

    def _discover_student_test_trips(self) -> None:
        test_dir = self.paths.data_root / "student_test"
        if test_dir.exists():
            _, trip_ids = discover_trips(test_dir)
            self._student_test_trip_ids = sorted(trip_ids)
        else:
            warnings.warn(f"Student test directory not found: {test_dir}")
            self._student_test_trip_ids = []

        settings.student_test_trip_ids = self._student_test_trip_ids

    @property
    def student_test_trip_ids(self) -> List[str]:
        return self._student_test_trip_ids

    def is_trip_available(self, trip_id: str) -> bool:
        return trip_id in self._student_test_trip_ids

    def get_trip_artifacts(self, trip_id: str) -> TripArtifacts:
        if trip_id in self._trip_cache:
            return self._trip_cache[trip_id]

        trip_dir = self.paths.data_root / "student_test" / trip_id
        if not trip_dir.exists():
            raise ValueError(f"Trip {trip_id} not found in student_test partition")

        trip_data = load_trip(trip_dir, trip_id)

        midpoints, start_points, end_points = compute_arc_geometry(
            trip_data.nodes, trip_data.arcs
        )
        arc_lengths = compute_arc_lengths(trip_data.nodes, trip_data.arcs)
        in_degree, out_degree = compute_node_degrees(trip_data.arcs, len(trip_data.nodes))
        adjacency = compute_adjacency(trip_data.arcs, len(trip_data.arcs))

        artifacts = TripArtifacts(
            trip_data=trip_data,
            arc_midpoints=midpoints,
            arc_start_points=start_points,
            arc_end_points=end_points,
            arc_lengths=arc_lengths,
            in_degree=in_degree,
            out_degree=out_degree,
            adjacency=adjacency,
        )

        self._trip_cache[trip_id] = artifacts
        return artifacts

    def generate_candidates(
        self,
        trip_id: str,
        k: int = 10,
        radius_meters: float = 100.0
    ) -> List[CandidateSet]:
        trip_artifacts = self.get_trip_artifacts(trip_id)
        trip_data = trip_artifacts.trip_data

        return generate_candidates_for_trip(
            trip_data,
            k=k,
            radius_meters=radius_meters,
            ground_truth_route=None,
        )

    def build_features(
        self,
        candidate_sets: List[CandidateSet],
        trip_id: str
    ) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]]]:
        trip_artifacts = self.get_trip_artifacts(trip_id)
        trip_data = trip_artifacts.trip_data

        X, _, feature_names, mapping = build_feature_matrix(
            candidate_sets,
            trip_data,
            ground_truth_arcs=None,
        )

        return X, feature_names, mapping

    def _align_features(
        self,
        X: np.ndarray,
        feature_names: List[str],
        model_feature_names: List[str]
    ) -> np.ndarray:
        if feature_names == model_feature_names:
            return X

        aligned_X = np.zeros((X.shape[0], len(model_feature_names)), dtype=np.float64)

        for i, name in enumerate(model_feature_names):
            if name in feature_names:
                src_idx = feature_names.index(name)
                aligned_X[:, i] = X[:, src_idx]
            else:
                warnings.warn(f"Feature '{name}' expected by model but not in pipeline")

        return aligned_X

    def predict_with_model(
        self,
        trip_id: str,
        candidate_sets: List[CandidateSet]
    ) -> Tuple[List[int], float, str, int, int]:
        X, feature_names, mapping = self.build_features(candidate_sets, trip_id)

        model_interface = self.model_artifacts.model_interface
        X_aligned = self._align_features(X, feature_names, model_interface.feature_names)

        model_start = time.perf_counter()
        model_probs = model_interface.get_probabilities(
            X_aligned,
            np.array([cand.arc_id for cs in candidate_sets for cand in cs.candidates])
        )
        model_ms = int((time.perf_counter() - model_start) * 1000)

        emission_config = self.hmm_params.to_emission_config()
        from src.hmm.emission import compute_emission_scores_from_model
        emissions = compute_emission_scores_from_model(
            candidate_sets, model_interface, X_aligned, emission_config
        )

        trip_artifacts = self.get_trip_artifacts(trip_id)
        transition_config = self.hmm_params.to_transition_config()
        from src.hmm.transition import compute_transition_scores
        transitions = compute_transition_scores(
            candidate_sets, trip_artifacts.trip_data.arcs, transition_config
        )

        viterbi_start = time.perf_counter()
        viterbi_config = self.hmm_params.to_viterbi_config()
        result = viterbi_decode(emissions, transitions, viterbi_config)
        hmm_ms = int((time.perf_counter() - viterbi_start) * 1000)

        predicted_route = reconstruct_path_from_result(result, candidate_sets)

        trip_data = trip_artifacts.trip_data
        true_route = trip_data.route.tolist()

        min_len = min(len(predicted_route), len(true_route))
        if min_len > 0:
            correct = sum(1 for p, t in zip(predicted_route[:min_len], true_route[:min_len]) if p == t)
            accuracy = correct / min_len
        else:
            accuracy = 0.0

        map_html = self._generate_map_html(trip_data, predicted_route, true_route)

        return predicted_route, accuracy, map_html, model_ms, hmm_ms

    def predict_mock(self, trip_id: str) -> Tuple[List[int], float, str, int, int]:
        np.random.seed(self.mock_seed + int(trip_id))

        trip_artifacts = self.get_trip_artifacts(trip_id)
        trip_data = trip_artifacts.trip_data

        true_route = trip_data.route.tolist()

        predicted = []
        for arc_id in true_route:
            if np.random.random() < 0.15:
                n_arcs = len(trip_data.arcs)
                predicted.append(int(np.random.randint(0, n_arcs)))
            else:
                predicted.append(arc_id)

        correct = sum(1 for p, t in zip(predicted, true_route) if p == t)
        accuracy = correct / len(true_route) if true_route else 0.0

        map_html = self._generate_map_html(trip_data, predicted, true_route)

        return predicted, accuracy, map_html, 0, 0

    def _generate_map_html(
        self,
        trip_data: TripData,
        predicted_route: List[int],
        true_route: List[int]
    ) -> str:
        center_lat = float(np.mean(trip_data.track[:, 1]))
        center_lon = float(np.mean(trip_data.track[:, 0]))

        def arcs_to_coords(arc_ids: List[int]) -> List[List[float]]:
            coords = []
            for arc_id in arc_ids:
                if 0 <= arc_id < len(trip_data.arcs):
                    from_node, to_node = trip_data.arcs[arc_id]
                    from_lon, from_lat = trip_data.nodes[from_node]
                    to_lon, to_lat = trip_data.nodes[to_node]
                    if not coords:
                        coords.append([float(from_lat), float(from_lon)])
                    coords.append([float(to_lat), float(to_lon)])
            return coords

        pred_coords = arcs_to_coords(predicted_route)
        true_coords = arcs_to_coords(true_route)
        gps_coords = [[float(lat), float(lon)] for lon, lat in trip_data.track[:, :2]]

        html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>PathSnap Prediction - Trip {trip_data.trip_id}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        #map { height: 500px; width: 100% !important; }
        .legend {
            position: absolute; bottom: 50px; left: 50px; z-index: 1000;
            background: white; padding: 10px; border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); font-size: 12px;
        }
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="legend">
        <div><span style="color: blue;">\u25cf</span> Raw GPS</div>
        <div><span style="color: green;">\u25cf</span> Predicted Route</div>
        <div><span style="color: red; opacity: 0.5;">\u25cf</span> Ground Truth (hidden)</div>
    </div>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 14);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '\u00a9 OpenStreetMap contributors'
        }}).addTo(map);

        var gpsCoords = {gps_coords};
        L.polyline(gpsCoords, {{color: 'blue', weight: 2, opacity: 0.7}}).addTo(map);

        var predCoords = {pred_coords};
        if (predCoords.length > 0) {{
            L.polyline(predCoords, {{color: 'green', weight: 3, opacity: 0.9}}).addTo(map);
        }}

        var trueCoords = {true_coords};
        if (trueCoords.length > 0) {{
            var trueLayer = L.polyline(trueCoords, {{color: 'red', weight: 3, opacity: 0.5, dashArray: '5, 10'}}).addTo(map);
        }}

        map.fitBounds(L.polyline(gpsCoords).getBounds());
    </script>
</body>
</html>
        """.format(
            trip_data=trip_data,
            center_lat=center_lat,
            center_lon=center_lon,
            gps_coords=gps_coords,
            pred_coords=pred_coords,
            true_coords=true_coords,
        )
        return html


# Global loader instance (initialized at startup)
artifact_loader: Optional[ArtifactLoader] = None


def get_artifact_loader() -> ArtifactLoader:
    global artifact_loader
    if artifact_loader is None:
        artifact_loader = ArtifactLoader(
            artifact_paths=settings.artifacts,
            mock_mode=settings.mock_enabled,
            mock_seed=settings.mock.seed,
        )
    return artifact_loader


def reset_artifact_loader() -> None:
    global artifact_loader
    artifact_loader = None


def init_artifacts() -> ArtifactLoader:
    global artifact_loader
    artifact_loader = ArtifactLoader(
        artifact_paths=settings.artifacts,
        mock_mode=settings.mock_enabled,
        mock_seed=settings.mock.seed,
    )
    artifact_loader.load_all()
    return artifact_loader