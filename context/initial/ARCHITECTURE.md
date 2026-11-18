# PathSnap — System Architecture (Reconnaissance Level)

**Milestone:** 11:30 Evaluation (Preprocessing & Feature Engineering Baseline)  
**Date:** 18 November 2026  
**Status:** Approved design — implementation in progress

> **Note:** This document captures the current approved/reconnaissance-level design. Final API details, model architectures, and deployment configs will be added at later checkpoints (CP2, CP3, CP4). Do not invent final API details here.

---

## High-Level Data Flow

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Raw Data   │────▶│  Data Loader     │────▶│  Candidate Gen  │────▶│   Features   │
│ (.track,    │     │  (validation,    │     │  (K-nearest     │     │  (geometric, │
│  .nodes,    │     │   schema checks) │     │   arcs per GPS) │     │  topological,│
│  .arcs,     │     │                  │     │                 │     │  temporal)   │
│  .route)    │     │                  │     │                 │     │              │
└─────────────┘     └──────────────────┘     └─────────────────┘     └──────┬───────┘
                                                                            │
                                                                            ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Evaluator  │◀─── │  FastAPI Server  │◀─── │  HMM + Viterbi  │◀─── │  ML Models   │
│  Test Trips │     │  (/predict)      │     │  (decoding)     │     │  (5+ models) │
└─────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │     UI       │
                    │ (trip selector│
                    │  + Folium)   │
                    └──────────────┘
```

---

## Component Specifications

### 1. Data Loader (`src/data/loader.py`)

**Input:** Per-trip directories with 4 files:
- `{trip_id}.track` — `lon lat timestamp` (whitespace, no header)
- `{trip_id}.nodes` — `lon lat` (0-based index = node_id)
- `{trip_id}.arcs` — `from_node to_node` (0-based index = arc_id)
- `{trip_id}.route` — `arc_id` per line (ground truth)

**Output:** `TripData` dataclass with validated NumPy arrays

**Validation:**
- Coordinate bounds: lon ∈ [8.0, 9.5], lat ∈ [49.0, 51.0]
- Timestamp monotonicity (gaps allowed, flagged >60s)
- Arc endpoints reference valid node_ids
- Route arc_ids exist in arcs
- No duplicate timestamps

**Partitions:**
- `train/` — 80 trips (IDs: 00000001–00000099, gaps)
- `student_test/` — 10 trips (IDs: 00000000, 00000007, 00000009, 00000040, 00000052, 00000056, 00000065, 00000076, 00000077, 00000091)
- `evaluator_test/` — **Not present** (reserved for final evaluation)

---

### 2. Candidate Generation (`src/features/candidates.py`)

**Algorithm:**
1. Precompute arc midpoints, start/end points from nodes + arcs
2. Build KDTree on arc midpoints (requires `scipy`; fallback: brute force)
3. For each GPS observation:
   - Radius search on midpoints (default 100m)
   - For each candidate arc, compute perpendicular distance to line segment
   - Keep top-K by perpendicular distance (default K=10)
4. Log candidate miss rate when ground truth route available

**Configuration:**
- `k`: Number of candidates per observation (default 10)
- `radius_meters`: Search radius for midpoint KDTree (default 100m)
- `use_kdtree`: Enable/disable KDTree (default True)

**Output:** `List[CandidateSet]` per trip, each containing sorted `Candidate` objects with:
- `arc_id`, `distance` (perp, meters), `along_track_dist`, `bearing_diff`, `midpoint_dist`

---

### 3. Feature Engineering (`src/features/engineer.py`)

**Features per candidate (11 total):**

| # | Feature | Description | Available at Inference |
|---|---------|-------------|------------------------|
| 1 | `perp_distance_m` | Perpendicular distance to arc segment | ✅ |
| 2 | `along_track_dist_m` | Distance along arc from start node | ✅ |
| 3 | `bearing_diff_deg` | |GPS bearing − arc bearing| (degrees) | ✅ |
| 4 | `speed_m_s` | perp_distance / time_delta | ✅ |
| 5 | `time_delta_s` | Timestamp difference from prev obs | ✅ |
| 6 | `arc_length_m` | Haversine length of arc | ✅ |
| 7 | `from_node_out_degree` | Out-degree of arc start node | ✅ |
| 8 | `to_node_in_degree` | In-degree of arc end node | ✅ |
| 9 | `arc_connectivity` | Number of adjacent arcs (1-hop) | ✅ |
| 10 | `candidate_rank` | 1-based rank by distance | ✅ |
| 11 | `is_continuation` | 1 if prev top-candidate connects to this arc | ✅ |

**Target:** Binary `is_ground_truth` (1 if candidate arc_id == aligned ground-truth arc_id)

**Leakage Prevention:**
- Ground truth alignment uses proportional index mapping (approx.)
- No future information used
- Scalers/encoders fit on train trips only

---

### 4. Train/Validation Splits (`src/data/splits.py`)

**Strategy:** GroupKFold (n_splits=5, groups=trip_id)

**Guarantee:** All observations from a trip belong to exactly one fold — zero trajectory leakage.

**Outputs:**
- `splits.json` with per-fold train/val indices and trip IDs
- `verify_no_leakage()` utility for validation

---

### 5. ML Models (Phase 2 — Not Yet Implemented)

**Minimum 5 models from different families:**
1. Logistic Regression (linear baseline)
2. Random Forest (tree ensemble)
3. HistGradientBoosting (fast boosting)
4. LightGBM / XGBoost (strong boosting)
5. MLP (shallow neural net) or SVM (kernel)

**CV:** GroupKFold on 80 train trips  
**Primary Metric:** Macro F1  
**Calibration:** `CalibratedClassifierCV` for probability quality (needed for HMM emissions)

---

### 6. HMM + Viterbi (Phase 3 — Not Yet Implemented)

**Hidden States:** Candidate arc_ids at each time step (variable K_t)

**Emissions:**
```
log P(obs_t | arc) = model_log_prob(arc) - λ * distance(gps_t, arc)
```

**Transitions:**
```
log P(arc_j | arc_i) = 
    0                    if arc_i == arc_j (stay)
    -hop_penalty * hops  if connected within H hops (BFS)
    -gap_penalty         otherwise (gap transition)
```

**Viterbi:** Standard DP, O(T × K²) with K≈5–10

---

### 7. FastAPI Server (Phase 5 — Not Yet Implemented)

**Endpoint:** `POST /predict`

**Request:**
```json
{ "trip_id": "00000000" }
```

**Response:**
```json
{
  "trip_id": "00000000",
  "predicted_route": [arc_id, ...],
  "accuracy": 0.87,
  "map_html": "<folium html>",
  "latencies_ms": { "candidates": 45, "model": 12, "hmm": 8, "total": 78 }
}
```

**Artifacts loaded at startup:**
- `model.pkl` (tuned classifier + scaler)
- `hmm_params.json` (λ, hop_penalty, gap_penalty, max_hops)
- Per-trip road network: `nodes.npy`, `arcs.npy`, `adjacency.npz`

---

### 8. UI (Phase 5 — Not Yet Implemented)

**Stack:** Streamlit or plain HTML+JS + Folium

**Features:**
- Dropdown of 10 evaluator trip IDs
- One-click prediction → calls `/predict`
- Interactive Folium map:
  - Raw GPS (blue)
  - Predicted route (green)
  - Ground truth (hidden until reveal)

---

## Data Contracts

### TripData
```python
@dataclass
class TripData:
    trip_id: str
    track: np.ndarray      # (N, 3): lon, lat, timestamp
    nodes: np.ndarray      # (M, 2): lon, lat
    arcs: np.ndarray       # (K, 2): from_node, to_node
    route: np.ndarray      # (L,): arc_id sequence
```

### Candidate
```python
@dataclass
class Candidate:
    arc_id: int
    distance: float          # perpendicular distance (meters)
    along_track_dist: float  # along-track distance (meters)
    bearing_diff: float      # bearing difference (degrees)
    midpoint_dist: float     # distance to midpoint (meters)
```

### FeatureVector
```python
@dataclass
class FeatureVector:
    candidate: Candidate
    features: np.ndarray     # (11,)
    feature_names: List[str]
    label: Optional[int]     # 1 if ground truth
```

---

## Coordinate System & Distance

- **Raw coordinates:** WGS84 longitude/latitude (degrees)
- **Distance calculations:** Haversine formula (meters)
- **Projection:** No projection used in baseline; Haversine for all distances
- **Bearing:** Initial bearing formula (degrees, 0=N, 90=E)

---

## Reproducibility

- Fixed random seeds: `random_state=42` for all sklearn operations
- Environment: `requirements.txt` (to be created)
- Experiment logging: `experiments/` directory with CSV/JSON logs
- Artifacts: Saved via `joblib`/`np.savez_compressed`

---

## Known Limitations (Current Milestone)

1. **Ground truth alignment** uses proportional index mapping — approximate
2. **No map-matching** of GPS to route polyline for exact labels
3. **Candidate generation** uses midpoint KDTree + perpendicular distance — not graph-aware
4. **No model training** yet — features only
5. **Python environment** must be fixed (missing `ast` module in system Python)

---

## Next Checkpoint Deliverables (CP2: Model Selected)

- [ ] `models/` directory with 5+ trained model artifacts
- [ ] `experiments/model_comparison.csv` with CV metrics
- [ ] `experiments/student_test_eval.csv` with test metrics
- [ ] `EXPERIMENTS.md` updated with measured results
- [ ] Bayesian optimisation study (`experiments/optuna_study.pkl`)
- [ ] `ARCHITECTURE.md` updated with final model/HMM details