# PathSnap — GPS Trajectory Map-Matching

**Hackathon:** UE24CS352A  
**Milestone:** 11:30 Evaluation — Preprocessing & Feature Engineering Baseline

---

## Quick Start

### 1. Fix Python Environment (Required)

The system Python 3.13 is corrupted (missing `ast` module). Create a clean virtual environment:

```bash
# Option A: Using python3.11+ if available
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Option B: Reinstall Python from python.org or use conda
conda create -n pathsnap python=3.11
conda activate pathsnap
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Dataset Inventory (EDA)

```bash
python scripts/dataset_inventory.py --data-root "Student folder/Student folder" --output experiments/dataset_inventory.json
```

### 4. Run Smoke Test

```bash
python scripts/smoke_test.py --train-trip 00000001 --test-trip 00000000 --k 10 --radius 100 --output experiments/smoke_test_results.json
```

### 5. Run Unit Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
├── src/
│   ├── data/           # Data loading & validation
│   │   ├── loader.py   # Parses .track, .nodes, .arcs, .route
│   │   └── splits.py   # GroupKFold splits (no leakage)
│   ├── features/       # Candidate generation & feature engineering
│   │   ├── candidates.py  # K-nearest arcs via KDTree + perpendicular distance
│   │   └── engineer.py    # 11 features (geometric, topological, temporal)
│   ├── models/         # Model training (Phase 2)
│   ├── hmm/            # HMM + Viterbi (Phase 3)
│   ├── api/            # FastAPI server (Phase 5)
│   └── ui/             # Streamlit/Folium UI (Phase 5)
├── tests/
│   ├── data/           # Data loader & split tests
│   └── features/       # Candidate & feature tests
├── scripts/
│   ├── dataset_inventory.py  # EDA CLI
│   └── smoke_test.py         # End-to-end pipeline test
├── experiments/        # Logs, metrics, artifacts
├── notebooks/          # EDA notebooks
├── requirements.txt    # Python dependencies
├── TASKS.md           # Task tracker with statuses
├── TEAM_ROLES.md      # Team delegation matrix
├── ARCHITECTURE.md    # System design (reconnaissance level)
├── MILESTONE_1130.md  # This milestone report
└── EXPERIMENTS.md     # Experiment log (measured results only)
```

---

## Data Format

Per-trip directory with 4 files (whitespace-separated, no headers):

| File | Columns | Description |
|------|---------|-------------|
| `.track` | `lon lat timestamp` | GPS observations (~1Hz) |
| `.nodes` | `lon lat` | Road nodes (0-based index = node_id) |
| `.arcs` | `from_node to_node` | Directed arcs (0-based index = arc_id) |
| `.route` | `arc_id` | Ground truth arc sequence |

**Partitions:**
- `Student folder/Student folder/train/` — 80 trips
- `Student folder/Student folder/student_test/` — 10 trips
- Evaluator test — reserved (not in repo)

---

## Key Features (Implemented)

- ✅ Robust parsers with schema validation
- ✅ Coordinate bounds, timestamp monotonicity, reference integrity checks
- ✅ Group-aware train/val splits (GroupKFold, zero trajectory leakage)
- ✅ Candidate generation: KDTree on midpoints + perpendicular distance to segments
- ✅ Configurable K (default 10) and radius (default 100m)
- ✅ Candidate miss rate logging when ground truth available
- ✅ 11 inference-time features: geometric, topological, temporal, context
- ✅ Binary labels via proportional ground-truth alignment
- ✅ Reproducible smoke test on real train + student_test trips
- ✅ Comprehensive unit tests

---

## Next Milestones

| Phase | Goal | Target |
|-------|------|--------|
| P2 | Model Zoo (5+ models, GroupKFold CV) | Day 3–4 |
| P3 | Bayesian Optimisation (Optuna) | Day 6–7 |
| P4 | HMM + Viterbi Decoding | Day 7–9 |
| P5 | FastAPI + UI + Two-Laptop Demo | Day 9–11 |
| P6 | Integration & Demo Prep | Day 11–13 |
| P7 | Report & Submission | Day 13–14 |

---

## Academic Integrity

- All code independently implemented
- No evaluator test data used for training
- Ground truth never exposed to client/UI
- Experiment logs record only measured results
- See `Assessment and Plagiarism Policy.md` for full guidelines

---

## License

Academic project — UE24CS352A Hackathon