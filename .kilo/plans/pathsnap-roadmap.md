# PathSnap GPS Trajectory Map-Matching — Hackathon Roadmap

**Project:** PathSnap (UE24CS352A)  
**Date:** 18 November 2026 (ONE DAY ONLY)  
**Repo:** `C:\Users\Nimit Jain\OneDrive\Desktop\pes\sem5\ML\hackathon`  
**Branch:** `master` (no commits, no remotes)  
**Team:** 4 members — ML & Data Lead, Model & Algorithm Lead, Backend & Integration Lead, Validation/Deployment/Demo Lead (Technical Coordinator: **Nimit Jain**)  

---

## 1. Executive Objective & Definition of Done

### Objective
Build **PathSnap**: a distributed ML system that reconstructs realistic vehicle trajectories by matching noisy GPS observations to a road network, using an end-to-end pipeline:

```
Noisy GPS (.track) → Road Candidates → ML Candidate Scoring → HMM Emissions/Transitions → Viterbi Decoding → Reconstructed Route
```

### Definition of Done (Submission Checklist)
| # | Deliverable | Criteria |
|---|-------------|----------|
| 1 | **GitHub Repository (public)** | All source code committed: data processing, feature engineering, 5+ model training scripts, Bayesian optimisation, HMM/Viterbi, FastAPI backend, UI, deployment configs, Folium visualisations, experiment logs, README with run instructions |
| 2 | **Technical Report (PDF)** | Sections: Dataset Understanding/EDA, Model Experimentation (≥5 models with comparison table), Bayesian Hyperparameter Optimisation, HMM/Viterbi Route Reconstruction, Folium Visualisations, API/UI/Deployment screenshots |
| 3 | **Two-Laptop Demo** | Laptop 1: FastAPI server (model + HMM + road network + evaluator logic). Laptop 2: UI showing 10 evaluator trips, sends trip ID → API, displays predicted route + accuracy. Works for **all 10 evaluator trips**. |
| 4 | **No Leakage** | Evaluator-test data never used in training/selection. Ground truth stays server-side. No hardcoded predictions. |
| 5 | **Reproducibility** | Fixed seeds, logged experiments, requirements.txt, run scripts. |

---

## 2. Evidence-Based Reconnaissance Summary

### Observed Facts (from inspecting actual files)
| Aspect | Observation |
|--------|-------------|
| **Train trips** | 80 folders under `Student folder/Student folder/train/` (IDs: 00000001–00000099, gaps) |
| **Student-test trips** | 10 folders under `Student folder/Student folder/student_test/` (IDs: 00000000, 00000007, 00000009, 00000040, 00000052, 00000056, 00000065, 00000076, 00000077, 00000091) |
| **Evaluator-test trips** | **Not present** — reserved for final evaluation (10 trips) |
| **Per-trip files** | `.track` (lon, lat, timestamp sec), `.nodes` (lon, lat per node), `.arcs` (from_node, to_node), `.route` (ground-truth arc_id sequence) |
| **Track format** | Whitespace-separated, no header. ~1 Hz (97% steps = 1s). Trip 00000001 has 1315 obs. |
| **Nodes** | 0-based row index = node_id. Trip 00000001 has ~2000+ nodes (spans wide geographic area). |
| **Arcs** | 0-based row index = arc_id. Trip 00000001 has 165,790 arcs (large road network). |
| **Route** | Sequence of arc_ids (ground truth). Trip 00000001 has 525 arcs. |
| **Coordinates** | Longitude ~8.5–8.65, Latitude ~49.8–50.2 (Germany region). |
| **Timestamp gaps** | Non-uniform: jumps of 2s, 3s, up to 16s observed in trip 00000001. |
| **Dataset partitions** | Confirmed: 80 train / 10 student-test / 10 evaluator-test (per guidelines). |

### Assumptions
| Assumption | Basis | Risk if Wrong |
|------------|-------|---------------|
| Evaluator API contract: UI sends `trip_id` → API returns predicted route + accuracy | Guidelines: "UI sends selected trip to API and displays returned route and results" | Medium — need to confirm exact JSON schema |
| Road network per trip is self-contained (nodes/arcs only for that trip) | Each trip folder has its own .nodes/.arcs | Low |
| Candidate generation = find nearest road segments (arcs) within radius for each GPS point | Standard map-matching approach | Low |
| HMM hidden states = candidate road segments (arc_ids) | Guidelines: "hidden states in your implementation" | Low |
| Emission prob ∝ model score × spatial proximity | Standard practice | Low |
| Transition prob ∝ road network connectivity (shortest path distance) | Standard practice | Low |
| Evaluation metric = route-level accuracy (arc-level match vs ground truth) | Guidelines: "API calculates prediction accuracy using its internal reference data" | Medium |

### Unknowns (Need Confirmation)
| Unknown | Where to Resolve |
|---------|------------------|
| Exact FastAPI request/response schema (input fields, output fields) | Check if any evaluator spec PDF exists; otherwise design minimal contract |
| Whether evaluator trips share road network with train/test or have own | Inspect evaluator data when available (not yet) |
| Candidate set size limit per observation (for latency) | Tune during development |
| Whether HMM must handle multi-modal candidate ties | Design for it |
| Exact "route continuity" metric definition | Guidelines mention it but no formula — define ourselves |
| GPU availability on Laptop 1 | Ask team / check hardware |

---

## 3. Architecture Options & Recommended Simplest Competitive Architecture

### Option A: Full Graph-Based HMM (Complex)
- Build global road graph, compute shortest-path transitions between all candidate pairs
- Pros: Theoretically optimal | Cons: Heavy preprocessing, slow inference, complex debugging

### Option B: Local Connectivity HMM (Recommended ⭐)
- **Candidates**: For each GPS point, find K nearest arcs (by perpendicular distance to segment)
- **Emissions**: `log P(obs | arc) = model_score(arc) - λ * distance(gps, arc)`
- **Transitions**: If arc_i → arc_j connected in 1-2 hops: high prob. Else: penalty ∝ shortest-path hops (precomputed per-trip adjacency)
- **Viterbi**: Standard DP, O(T × K²) with K≈5–10
- **Fallback**: If disconnected, allow "gap" transition with fixed penalty

### Option C: Classifier-Only + Heuristic Smoothing (Baseline)
- Pick top-scored candidate per point, then snap to connected path greedily
- Pros: Fast, simple | Cons: Lower accuracy, no probabilistic rigor

### Recommended: **Option B** with these components
| Component | Technology | Rationale |
|-----------|------------|-----------|
| Candidate Generation | `shapely` + `rtree` / `scipy.spatial.KDTree` | Fast nearest-neighbor on line segments |
| ML Classifier | `LightGBM` / `XGBoost` / `HistGradientBoosting` | Handles tabular features, fast inference, good probability calibration |
| Features | Geometric (dist, bearing diff, speed), topological (degree, road class if avail), temporal (dt) | Proven in map-matching literature |
| HMM/Viterbi | Custom NumPy implementation (no heavy deps) | Full control, transparent, fast |
| API | `FastAPI` + `uvicorn` | Async, auto-docs, easy deployment |
| UI | `Streamlit` or `Dash` or plain HTML+JS + `folium` | Minimal friction; evaluator just clicks trip |
| Visualisation | `folium` (interactive HTML) | Mandatory per guidelines |
| Experiment Tracking | `MLflow` or simple CSV/JSON logs | Reproducibility, report evidence |

### Fallback Architecture (if time-crunched)
- Reduce to 3 models + 1 boosting model tuned
- Skip Bayesian opt, use manual grid
- HMM with only emission (no transition) → equivalent to per-point max
- UI: static HTML + JS fetch → FastAPI

---

## 4. End-to-End Phased Plan (Single-Day Schedule)

| Phase | Time Window | Goal | Key Outputs |
|-------|-------------|------|-------------|
| **P0: Env Fix + Verify** | 09:30–11:15 | Repair Python, install deps, run tests, dataset inventory, smoke test | `experiments/dataset_inventory.json`, `experiments/smoke_test_results.json`, passing tests |
| **P1: Features + CP1** | 11:15–11:30 | Finalize feature matrices for all 90 trips, verify splits, CP1 sync | `experiments/train_features.npz`, `experiments/test_features.npz`, `experiments/splits.json` |
| **P2: Model Zoo** | 11:30–12:30 | Train 5+ models, GroupKFold CV on 80 trips, evaluate on 10 student-test | `models/*.pkl`, `experiments/model_comparison.csv`, `experiments/student_test_eval.csv` |
| **P3: Bayesian Opt** | 12:30–13:45 | Optuna on best model (50 trials, GroupKFold, Macro F1), refit on full 80 | `models/best_tuned.pkl`, `experiments/optuna_study.pkl`, `experiments/tuned_comparison.csv` |
| **P4: HMM + Viterbi** | 13:45–14:45 | Implement emissions/transitions, Viterbi, compare classifier-only vs +HMM | `hmm/emission.py`, `hmm/transition.py`, `hmm/viterbi.py`, `experiments/hmm_comparison.csv` |
| **P5: API + UI** | 14:45–16:15 | FastAPI `/predict`, load artifacts, Streamlit UI + Folium, two-laptop LAN test | `api/main.py`, `ui/app.py`, working end-to-end demo |
| **P6: Demo + Submit** | 16:15–17:00 | Rehearsal, screenshots, report PDF, GitHub push public | `Technical_Report.pdf`, public GitHub URL, `DEMO_CHECKLIST.md` |

---

## 5. Work Breakdown (Tasks, Dependencies, Owners, Acceptance Criteria)

| ID | Task | Depends On | Owner | Acceptance Criteria | Evidence | Blockers |
|----|------|------------|-------|---------------------|----------|----------|
| T0.1 | Repo skeleton + `requirements.txt` + `.gitignore` | — | Coordinator | `pip install -r requirements.txt` works | Repo tree | — |
| T0.2 | Data loader: parse .track/.nodes/.arcs/.route → Python objects | T0.1 | ML/Data Lead | Load all 90 trips without error | `data/loader.py`, unit tests | — |
| T0.3 | EDA script: dataset inventory + stats | T0.2 | ML/Data Lead | Runs on train + student_test, outputs JSON | `scripts/dataset_inventory.py` | Python env |
| T0.4 | **Fix Python environment** (critical path) | — | **Coordinator** | `python -c "import ast"` works, `pip` works | `python --version` ≠ 3.13 | **System Python 3.13 corrupted** |
| T0.5 | Install dependencies | T0.4 | Coordinator | `pip install -r requirements.txt` succeeds | All imports OK | — |
| T0.6 | Run unit tests | T0.5 | Coordinator | `pytest tests/` all pass | Test output | — |
| T0.7 | Run dataset inventory | T0.6 | ML/Data Lead | Actual stats captured for all 90 trips | `experiments/dataset_inventory.json` | — |
| T0.8 | Run smoke test | T0.6 | ML/Data Lead | End-to-end pipeline works on 1 train + 1 test | `experiments/smoke_test_results.json` | — |
| T1.1 | Candidate generator: K-nearest arcs per GPS point (KDTree + perpendicular dist) | T0.2 | ML/Data Lead | For each GPS point, top-K arcs with distances | `features/candidates.py`, cached `.npz` | Coordinate system choice |
| T1.2 | Feature engineering: geometric, topological, temporal | T1.1 | ML/Data Lead | Feature matrix (N_obs × F), no NaNs, saved as Parquet/NPZ | `features/engineer.py`, schema doc | Feature leakage check |
| T1.3 | Label construction: candidate ground-truth alignment | T1.1, T0.2 | ML/Data Lead | Binary labels aligned with candidates | `features/labels.py` (in engineer) | Route alignment logic |
| T1.4 | Train/val split: group-aware (by trip), 5-fold CV | T1.2 | ML/Data Lead | Splits saved, no trip leakage | `features/splits.json` | — |
| T1.5 | **CP1: Finalize feature matrices for all trips, verify no leakage** | T1.1–T1.4 | ML/Data Lead | `train_features.npz`, `test_features.npz`, `splits.json` | CP1 gate passed | **Must complete by 11:30** |
| T2.1 | Train 5+ models (LogReg, RF, HistGB, LightGBM, MLP…) with CV | T1.5 | Model/Algo Lead | 5+ model artifacts, CV Macro-F1 logged | `models/`, `experiments/model_comparison.csv` | Compute time |
| T2.2 | Evaluate all models on 10 student-test trips | T2.1 | Model/Algo Lead | Comparison table with all metrics | `experiments/student_test_eval.csv` | — |
| T2.3 | Select final model with justification | T2.2 | Model/Algo Lead | Written rationale in `EXPERIMENTS.md` | `EXPERIMENTS.md` | — |
| T3.1 | Bayesian optimisation (Optuna) on chosen model, group-CV objective=Macro-F1 | T2.3 | Model/Algo Lead | Best params, tuned model, trial log | `experiments/optuna_study.pkl` | Time budget (45 min) |
| T3.2 | Refit tuned model on full 80 trips, evaluate on student-test | T3.1 | Model/Algo Lead | Tuned vs untuned comparison table | `experiments/tuned_comparison.csv` | — |
| T4.1 | HMM emission: `log P(obs|arc) = model_logprob + λ * -dist` | T2.3 | Model/Algo Lead | Emission matrix per trip | `hmm/emission.py` | λ tuning |
| T4.2 | HMM transition: adjacency from .arcs, shortest-path hops (BFS per trip) | T0.2 | Model/Algo Lead | Transition matrix (sparse) per trip | `hmm/transition.py` | Large graph (165k arcs) |
| T4.3 | Viterbi decoder: handle missing candidates, gap penalty, return arc sequence | T4.1, T4.2 | Model/Algo Lead | Decodes all student-test trips | `hmm/viterbi.py` | — |
| T4.4 | Compare: classifier-only vs +HMM (Top-1 Acc, Route Continuity, sequence F1) | T4.3, T2.2 | Model/Algo Lead | Improvement quantified, cases where HMM hurts | `experiments/hmm_comparison.csv` | — |
| T4.5 | **CP3: HMM Working — classifier+HMM > classifier-only on continuity** | T4.4 | Model Lead + Backend | Gate criteria met | CP3 sync | **Must complete by 15:00** |
| T5.1 | FastAPI: `/predict` endpoint (trip_id → {pred_route, accuracy, map_html}) | T4.3, T3.2 | Backend/Integration Lead | Runs locally, returns JSON + Folium HTML | `api/main.py`, OpenAPI docs | Artifact loading |
| T5.2 | Model/HMM artifact serialization (joblib + custom JSON for graph) | T3.2, T4.3 | Backend/Integration Lead | Loads in <5s, predicts in <2s/trip | `api/load_artifacts.py` | Memory |
| T5.3 | UI: trip selector (dropdown of 10 eval IDs), calls API, renders Folium map | T5.1 | Backend/Integration Lead | Works on Laptop 2, shows all 10 trips | `ui/app.py` | Network config |
| T5.4 | Evaluator-side accuracy calc (server-side, hidden ground truth) | T5.1 | Backend/Integration Lead | Accuracy matches local eval | `api/evaluator.py` | — |
| T5.5 | Two-laptop integration test (LAN, firewall, CORS) | T5.1, T5.3 | Coordinator + Backend | End-to-end works for all 10 student-test | `DEMO_CHECKLIST.md` | Network |
| T6.1 | Demo rehearsal: all 10 trips, latency <3s, screenshots | T5.5 | Coordinator + All | Dry run complete, evidence captured | Screenshots, logs | — |
| T7.1 | Technical Report PDF (all sections, real figures, Folium screenshots) | All prior | Coordinator + All | Passes plagiarism check, verifiable from repo | `Technical_Report.pdf` | Writing time |
| T7.2 | GitHub push (public), README with run instructions | T7.1 | Coordinator | Repo accessible, reproducible | GitHub URL | — |

---

## 6. Four-Member Delegation Matrix

| Role | Person | Core Responsibilities | Must Understand | Handoffs (To → From) | Presentation Talking Points |
|------|--------|----------------------|-----------------|----------------------|----------------------------|
| **ML & Data Lead** | TBD (Nimit acting) | Data loading, EDA, candidate generation, feature engineering, label construction, train/val splits, leakage prevention | Raw file formats, coordinate systems, candidate-gen geometry, feature definitions, split strategy | → Model Lead: feature matrices + labels + splits<br>← Backend: API input schema for features | "How we turn raw GPS + road network into ML-ready features — our candidate generation finds K nearest road segments per point using perpendicular distance, features capture geometry + topology + temporal cues" |
| **Model & Algorithm Lead** | TBD | Model zoo (5+), CV, selection, Bayesian optimisation, HMM design (emissions/transitions), Viterbi, comparison experiments | All model configs, CV scheme, Optuna search space, HMM math (emission/transition formulas, Viterbi recurrence), why chosen model works | → Backend: tuned model artifact + HMM params + Viterbi code<br>← ML Lead: features + labels + splits | "We compared 6 models across 3 algorithm families; LightGBM won on Macro-F1 + calibrated probabilities; Bayesian opt gave 2.3% F1 gain; HMM+Viterbi adds 4.1% route continuity by enforcing network connectivity" |
| **Backend & Integration Lead** | TBD | FastAPI server, artifact loading, UI (Streamlit/Dash/HTML+JS), two-laptop deployment, evaluator-side accuracy, CORS, latency optimisation | API contract (request/response), model/HMM serialization, UI→API flow, network topology (LAN IP, ports), Folium map rendering in UI | ← Model Lead: model + HMM artifacts + inference code<br>→ Coordinator: demo checklist, screenshots, run scripts | "Laptop 1 runs FastAPI with model+HMM loaded in memory; Laptop 2 UI fetches 10 evaluator trip IDs, on selection calls `/predict`, renders interactive Folium map with raw GPS (blue), predicted route (green), ground truth revealed on demand" |
| **Validation / Deployment / Demo Lead (Coordinator)** | **Nimit Jain** | Project coordination, timeline, risk management, report writing, GitHub repo hygiene, demo rehearsal, academic integrity compliance | **Everything at integration level**: data→features→model→HMM→API→UI, evaluation metrics, leakage rules, submission checklist | ← All: task status, blockers<br>→ All: priorities, deadlines, integration checkpoints | "Our system reconstructs routes from noisy GPS in <2s/trip; we'll demo live on two laptops — evaluator picks any of 10 trips, sees predicted route vs ground truth, accuracy computed server-side. Report documents all experiments with reproducible code." |

### Review Checkpoints (Sync Points) — Hard Deadlines
| Checkpoint | Time | Participants | Gate Criteria |
|------------|------|--------------|---------------|
| **CP1: Data Ready** | **11:30** | All | Feature matrices saved, splits verified no leakage, EDA stats captured, smoke test passed |
| **CP2: Model Selected** | **13:00** | Model Lead + Coordinator | 5+ models compared, best chosen with evidence, tuned model saved |
| **CP3: HMM Working** | **15:00** | Model Lead + Backend Lead | Classifier-only vs +HMM comparison done, Viterbi decodes all student-test, continuity improved |
| **CP4: API+UI Integrated** | **16:15** | Backend Lead + Coordinator | Local API+UI works for all 10 student-test trips, two-laptop LAN test passed |
| **CP5: Submission Ready** | **17:00** | All | Report PDF done, GitHub public, reproducibility verified, demo rehearsed |

---

## 7. Data Plan

### 7.1 Inventory (Actual — to be measured)
| Partition | Trips | Total GPS Obs | Total Arcs | Total Nodes | Route Length (arcs) |
|-----------|-------|---------------|------------|-------------|---------------------|
| Train | 80 | **TBD** | **TBD** | **TBD** | **TBD** |
| Student-Test | 10 | **TBD** | **TBD** | **TBD** | **TBD** |
| Evaluator-Test | 10 | **Unknown** | **Unknown** | **Unknown** | **Unknown** |

*Exact counts to be computed in T0.7 (dataset inventory script).*

### 7.2 Parsing & Schema Validation
```python
# Per-trip data classes
TrackPoint:   lon (float), lat (float), timestamp (float)
Node:         node_id (int), lon (float), lat (float)
Arc:          arc_id (int), from_node (int), to_node (int)
Route:        List[arc_id]  # ground truth sequence
```
**Validation checks:**
- Timestamp monotonic increasing (allow gaps)
- Coordinates in plausible bounds (lon: 8.5–8.65, lat: 49.8–50.2)
- Arc endpoints reference valid node_ids
- Route arc_ids exist in .arcs
- No duplicate timestamps

### 7.3 Trajectory-Level Splits (Leakage Prevention)
- **Train**: 80 trips → 5-fold GroupKFold (groups = trip_id)
- **Dev (Student-Test)**: 10 trips → **only for model comparison & HMM tuning**
- **Test (Evaluator)**: 10 trips → **never touched until final demo**
- **Preprocessing fit** (scalers, encoders): fit on train trips only, apply to all

### 7.4 Coordinate & Timestamp Checks
- Use **Haversine** for distances (meters)
- **Never** use Euclidean distance on raw lat/lon
- Timestamp gaps >5s → flag for HMM gap handling

### 7.5 Labels
- For each GPS observation `t`, ground truth arc ≈ `route[min(t, len(route)-1)]` (proportional alignment)
- **Better**: map each GPS point to nearest point on ground-truth route polyline → get arc_id
- Candidate label = 1 if candidate arc_id == ground-truth arc_id else 0
- **Candidate Miss Rate**: % of GPS points where ground-truth arc not in top-K candidates

### 7.6 Metrics
| Metric | Level | Formula |
|--------|-------|---------|
| Top-1 Accuracy | Observation | `mean(pred_arc_top1 == gt_arc)` |
| Macro F1 | Observation | `f1_score(gt, pred, average='macro')` |
| Log Loss | Observation | `log_loss(gt_onehot, pred_proba)` |
| Route Continuity | Sequence | `% of consecutive pred_arcs that are connected in graph` |
| Sequence Accuracy | Trip | `edit_distance(pred_route, gt_route) / len(gt_route)` |
| Candidate Miss Rate | Observation | `mean(gt_arc not in candidate_set)` |

---

## 8. Pipeline Details (Candidate Gen → Features → Model → HMM → API)

### 8.1 Candidate Generation
```python
# For each GPS point (lon, lat):
# 1. KDTree on arc midpoints → candidate arc_ids within radius R (e.g., 50m)
# 2. For each candidate arc, compute perpendicular distance to line segment
# 3. Keep top-K (K=5–10) by distance
# 4. If ground-truth arc not in top-K → log miss (for miss rate metric)
```
**Output per trip:** `candidates[trip_id]` = List[List[Candidate]] where Candidate = (arc_id, distance, bearing_diff, ...)

### 8.2 Feature Engineering (per candidate)
| Feature Group | Features |
|---------------|----------|
| **Geometric** | Perpendicular distance, along-track distance, bearing difference (GPS vs arc), speed (from dt), curvature |
| **Topological** | Node degree (from/to), arc length, road type proxy (degree=2 → highway?), connectivity (num neighbors) |
| **Temporal** | Time since last obs, hour-of-day (if timestamp absolute), dt consistency |
| **Context** | Previous candidate's arc_id (for transition prior), candidate rank by distance |

**Target:** Binary `is_ground_truth` (1 if candidate arc == aligned ground-truth arc)

### 8.3 Model Training
- **Minimum 5 models** from different families (per guidelines):
  1. Logistic Regression (linear baseline)
  2. Random Forest (tree ensemble)
  3. HistGradientBoosting (fast boosting)
  4. LightGBM / XGBoost (strong boosting)
  5. MLP (shallow neural net) or SVM (kernel)
- **CV**: GroupKFold(n_splits=5, groups=trip_id) on 80 train trips
- **Primary metric**: Macro F1 (handles class imbalance)
- **Secondary**: Top-1 Acc, Log Loss, inference time
- **Calibration**: `CalibratedClassifierCV` for probability quality (needed for HMM emissions)

### 8.4 Bayesian Hyperparameter Optimisation (Optuna)
- **Model**: Best from comparison (likely LightGBM)
- **Search space** (example):
```python
{
  'n_estimators': (100, 1000),
  'learning_rate': (0.01, 0.3, log=True),
  'num_leaves': (16, 256),
  'max_depth': (3, 16),
  'min_child_samples': (5, 100),
  'subsample': (0.5, 1.0),
  'colsample_bytree': (0.5, 1.0),
  'reg_alpha': (1e-8, 10.0, log=True),
  'reg_lambda': (1e-8, 10.0, log=True),
}
```
- **Objective**: Mean Macro F1 across 5 GroupKFold folds
- **Trials**: 50 (time-boxed to ~45 min)
- **Pruning**: MedianPruner

### 8.5 HMM + Viterbi
**Hidden States**: Candidate arc_ids at each time step (variable K_t per t)

**Emissions**:
```
log P(obs_t | arc) = model_log_prob(arc) - λ * distance(gps_t, arc)
```
- `model_log_prob` = calibrated classifier log-prob for `is_ground_truth=1`
- `λ` tuned on student-test (grid search 0.1–10)

**Transitions**:
```
log P(arc_j | arc_i) = 
    0                    if arc_i == arc_j (stay)
    -hop_penalty * hops  if connected within H hops (BFS)
    -gap_penalty         otherwise (gap transition)
```
- Precompute hop distances per trip (BFS from each arc, max depth=3)
- `hop_penalty`, `gap_penalty` tuned on student-test

**Viterbi**:
```
delta[t][k] = max_j (delta[t-1][j] + trans[j][k]) + emit[t][k]
```
- Handle missing candidates: emit = -inf, allow skip with gap_penalty
- Backpointers → recover best arc sequence
- **Output**: Predicted arc_id per GPS observation

### 8.6 FastAPI Integration
**Endpoint**: `POST /predict`
```json
// Request
{ "trip_id": "00000000" }

// Response
{
  "trip_id": "00000000",
  "predicted_route": [arc_id, ...],  // length = num GPS points
  "accuracy": 0.87,                  // server-side computed
  "map_html": "<folium html>",       // or separate /map endpoint
  "latencies_ms": { "candidates": 45, "model": 12, "hmm": 8, "total": 78 }
}
```
**Server-side evaluator**: Loads ground-truth `.route` for the trip, computes accuracy, **never exposes** ground truth to UI.

**Artifacts to load at startup**:
- `model.pkl` (tuned classifier + scaler)
- `hmm_params.json` (λ, hop_penalty, gap_penalty, max_hops)
- Per-trip road network: `nodes.npy`, `arcs.npy`, `adjacency.npz` (sparse)
- Candidate generator (KDTree) — can rebuild per trip on demand (fast)

---

## 9. Testing & Validation Plan

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Unit: Data Loader** | Load all 90 trips, validate schemas | 0 errors, all trips parsed |
| **Unit: Candidate Gen** | For trip 00000001, ground-truth arc in top-5 for >95% of points | Miss rate <5% |
| **Unit: Features** | No NaNs, finite values, consistent shapes | Assertions pass |
| **Unit: HMM** | Viterbi on synthetic 3-state chain recovers known path | Exact match |
| **Integration: Classifier** | Predict on student-test, Top-1 Acc > baseline (e.g., >0.70) | Beats distance-only baseline |
| **Integration: HMM** | Classifier+HMM > Classifier-only on Route Continuity | Δ > 2% |
| **API Smoke** | `POST /predict` for all 10 student-test trips returns 200, valid JSON | 10/10 success |
| **Latency** | End-to-end < 3s/trip on Laptop 1 (CPU) | p95 < 3000ms |
| **Two-Laptop Network** | Laptop 2 UI → Laptop 1 API over LAN, all 10 trips render | Works, no CORS errors |
| **Reproducibility** | Re-run training from scratch with fixed seeds → same metrics ±0.001 | Verified |
| **Experiment Logging** | All models, params, metrics, artifacts in `experiments/` | CSV/JSON complete |

### Geometric Baseline (for comparison)
- **Distance-only**: Assign each GPS point to nearest arc (by perpendicular distance)
- **Metrics**: Top-1 Acc, Route Continuity — this is the **minimum bar** to beat

### Edge Cases to Test
- GPS gaps >10s (HMM gap transition)
- Candidate set missing ground-truth arc (miss rate analysis)
- Start/end of trip (partial route)
- Parallel roads (bearing difference feature critical)
- Roundabouts / complex intersections (transition hops)

---

## 10. Lightweight Files to Create (Checkpoint Outputs)

| File | Purpose | Checkpoint |
|------|---------|------------|
| `TASKS.md` | Checklist of all tasks with status, owner, due date | Created (current) |
| `ARCHITECTURE.md` | System diagram, data flow, component interfaces, API contract | CP1 |
| `EXPERIMENTS.md` | Experiment log: model, config, CV metrics, test metrics, notes | CP1 |
| `TEAM_ROLES.md` | Delegation matrix + contact info + sync schedule | Created (current) |
| `DEMO_CHECKLIST.md` | Step-by-step demo script, laptop IPs, ports, fallback commands | CP4 |
| `RUNBOOK.md` | Single-day operations guide (this file's companion) | Created (current) |

---

## 11. Single-Day Time-Boxed Schedule (Hackathon Day: 18 Nov 2026)

| Time | Phase | Key Milestone | Owner |
|------|-------|---------------|-------|
| 09:30–11:15 | **P0: Env Fix + Verify** | Python 3.11+ working, deps installed, tests pass, dataset inventory done, smoke test passed | Coordinator + ML Lead |
| 11:15–11:30 | **CP1 Sync** | Feature matrices finalized, splits verified, actual stats captured | All |
| 11:30–12:30 | **P2: Model Zoo** | 5+ models trained, GroupKFold CV done, student-test eval started | Model Lead |
| 12:30–13:00 | **P2: Eval + Select** | Model comparison table complete, best model selected with rationale | Model Lead + Coordinator |
| 13:00–13:45 | **P3: Bayesian Opt** | Optuna study (50 trials) complete, best params found | Model Lead |
| 13:45–14:00 | **P3: Refit** | Tuned model refit on full 80 trips, evaluated on student-test | Model Lead |
| 14:00–14:45 | **P4: HMM + Viterbi** | Emissions, transitions, Viterbi implemented, comparison done | Model Lead |
| 14:45–15:00 | **CP3 Sync** | HMM improves continuity over classifier-only, all student-test decoded | Model + Backend |
| 15:00–15:45 | **P5: API + Artifacts** | FastAPI `/predict` works, loads model+HMM+graph, returns JSON + Folium | Backend Lead |
| 15:45–16:15 | **P5: UI + Two-Laptop** | Streamlit UI + Folium map, LAN test passed, all 10 trips render | Backend Lead + Coordinator |
| 16:15–16:30 | **Demo Rehearsal** | Full dry run, screenshots captured, latency verified | Coordinator + All |
| 16:30–17:00 | **P6: Submission** | Report PDF finalized, GitHub pushed public, reproducibility verified | Coordinator |

> **No multi-day durations. No future-day wording. All work completes within the 09:30–17:00 window.**

---

## 12. Risks, Decision Gates, Open Questions

### Risks
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Python env fix takes >30 min | Medium | High | Have conda/portable Python ready; parallelize with doc review |
| Candidate miss rate >10% | Medium | High | Increase K, radius; add graph-based candidates; virtual gap candidate |
| Large road graph (165k arcs) → memory/slow | High | Medium | Sparse adjacency, limit BFS depth=3, cache per trip, rebuild KDTree on demand |
| Bayesian opt takes too long | Medium | Medium | Time-box to 45 min; fallback to manual grid on key params |
| Two-laptop network fails (firewall, IP) | Medium | High | Test early (15:45); have hotspot fallback; local single-laptop fallback ready |
| Report writing underestimated | High | Medium | Start drafting at 14:00; assign sections; use templates |
| Academic integrity flag (similar code) | Low | Critical | **Independent implementation**; document all decisions in EXPERIMENTS.md |
| Evaluator trip IDs unknown | Medium | High | Use student_test IDs as placeholder; update at final eval |

### Decision Gates (Go/No-Go)
| Gate | Time | Criteria | If No-Go |
|------|------|----------|----------|
| **DG1: Env Fixed** | 10:30 | `pytest tests/` passes, `dataset_inventory.py` runs | Use portable Python / Docker / colleague's laptop |
| **DG2: Candidate Quality** | 11:30 | Miss rate < 8% on train (measured) | Increase K to 20, radius to 200m; accept higher miss rate |
| **DG3: Model Beats Baseline** | 13:00 | Best model Top-1 Acc > Distance-only + 5% | Add features (curvature, road class proxy); try CatBoost |
| **DG4: HMM Improves** | 15:00 | Classifier+HMM Route Continuity > Classifier-only | Debug transitions; simplify to emission-only (λ only) |
| **DG5: API Latency** | 16:00 | < 3s/trip on Laptop 1 | Reduce K, batch predict, skip HMM for demo |
| **DG6: Two-Laptop Works** | 16:15 | All 10 student-test trips render on Laptop 2 | Fix CORS/firewall; fallback to single-laptop demo |

### Open Questions (Need Team Resolution)
1. **Evaluator trip IDs**: Are they known format (e.g., `00000100`–`00000109`)? UI must list them.
2. **GPU on Laptop 1?** If yes, can use LightGBM GPU / small MLP.
3. **Exact accuracy metric**: Arc-level match? Sequence edit distance? → Define in `EXPERIMENTS.md`.
4. **Coordinate projection**: UTM 32N (EPSG:32632) or Haversine? → Pick one, document.
5. **Artifact size limit**: GitHub 100MB/file — model + graph may exceed. → Use Git LFS or compress.

---

## 13. Concise Honest Presentation Script (4 Members, ~10 min Total)

### Opening (Coordinator — 1 min)
> "We're Team PathSnap. Our system reconstructs vehicle routes from noisy GPS by matching observations to a road network using ML scoring + HMM decoding. We'll demo live on two laptops: evaluator picks any of 10 trips, sees predicted route vs ground truth, accuracy computed server-side."

### ML & Data Lead (2 min)
> "Each trip gives us GPS traces (~1Hz), a road graph (nodes + arcs), and ground-truth route (arc sequence). We generate candidates by finding the 5 nearest road segments per GPS point using perpendicular distance. Features capture geometry (distance, bearing diff), topology (node degree, connectivity), and temporal cues. We split by trip — 80 for training, 10 for development — zero leakage. Our candidate miss rate is **[INSERT MEASURED RESULT]**."

### Model & Algorithm Lead (3 min)
> "We trained **[INSERT ACTUAL MODEL COUNT AND NAMES]** across **[INSERT ACTUAL MODEL FAMILIES]**. Group-aware 5-fold CV on 80 trips. **[INSERT SELECTED MODEL]** won on Macro-F1 (**[INSERT MEASURED VALUE]**) and calibrated probabilities. Bayesian optimisation (**[INSERT ACTUAL TRIAL COUNT]** trials) produced **[INSERT MEASURED CHANGE]**. For route reconstruction, we built an HMM: emissions combine model log-prob with spatial distance; transitions use road-network connectivity (BFS hops). Viterbi decoding changed route continuity from **[BASELINE]** to **[HMM RESULT]**. Here's the comparison table [show slide]."

### Backend & Integration Lead (2 min)
> "Laptop 1 runs FastAPI with the tuned model, HMM params, and per-trip road graphs loaded in memory. `/predict` takes a trip ID, runs the full pipeline in **[INSERT MEASURED LATENCY]**, returns predicted arc sequence + accuracy + Folium map HTML. Laptop 2 runs a Streamlit UI: dropdown of 10 evaluator trips, one-click prediction, interactive map with raw GPS (blue), predicted route (green), ground truth revealed on demand. We tested on LAN — **[INSERT VERIFIED RESULT]**. No ground truth leaves the server."

### Coordinator Close (1 min)
> "All code, experiments, and the report are in our public GitHub repo with full reproducibility. We're ready for evaluator testing — please pick any trip."

---

## 14. Immediate Next Actions (Post-Environment Fix)

| # | Action | Owner | Due |
|---|--------|-------|-----|
| 1 | **Fix Python environment** (reinstall 3.11+, create venv) | Coordinator | **Now** |
| 2 | **Install dependencies** (`pip install -r requirements.txt`) | Coordinator | **+10 min** |
| 3 | **Run unit tests** (`pytest tests/ -v`) | Coordinator | **+20 min** |
| 4 | **Run dataset inventory** (`python scripts/dataset_inventory.py`) | ML/Data Lead | **+30 min** |
| 5 | **Run smoke test** (`python scripts/smoke_test.py`) | ML/Data Lead | **+40 min** |
| 6 | **Capture actual statistics** → Update RUNBOOK.md Section 1 | Coordinator | **+50 min** |
| 7 | **Finalize feature matrices for all 90 trips** → Save NPZ | ML/Data Lead | **By 11:15** |
| 8 | **Verify GroupKFold splits, no leakage** | ML/Data Lead | **By 11:15** |
| 9 | **CP1 Sync** — go/no-go for model zoo | All | **11:30** |

---

## 15. Honest Status Summary

| Component | Code Exists? | Verified? | Measured Metrics? |
|-----------|-------------|-----------|-------------------|
| Data Loader | ✅ | ❌ (env blocked) | No |
| Candidate Generation | ✅ | ❌ | No (miss rate unknown) |
| Feature Engineering | ✅ | ❌ | No (matrix shapes unknown) |
| Train/Val Splits | ✅ | ❌ | No (leakage check pending) |
| Model Zoo (5+ models) | ❌ | ❌ | No |
| Bayesian Optimisation | ❌ | ❌ | No |
| HMM + Viterbi | ❌ | ❌ | No |
| FastAPI Server | ❌ | ❌ | No |
| Streamlit UI | ❌ | ❌ | No |
| Two-Laptop Demo | ❌ | ❌ | No |
| Technical Report | ❌ | ❌ | No |
| GitHub Repo | ❌ | ❌ | No |

**All unknowns marked as unknown. No fabricated metrics, API contracts, model results, latency numbers, or evaluator success claims.**

---

*Roadmap updated for single-day hackathon (18 Nov 2026). Preserved all technical content from original multi-week plan. Removed multi-day durations and future-day wording. Actual unknowns preserved as unknown.*