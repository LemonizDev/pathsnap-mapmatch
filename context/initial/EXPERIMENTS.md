# PathSnap — Experiment Log

**Milestone:** 11:30 Evaluation (Preprocessing & Feature Engineering Baseline)  
**Date:** 18 November 2026

> **Rule:** Only record **measured results** from actual runs. Never fabricate metrics. Update this file after each experiment.

---

## Experiment Template

### Experiment: [Name/ID]
**Date:** YYYY-MM-DD  
**Phase:** [P1/P2/P3/P4/P5]  
**Description:** [What was tested]  
**Config:** [Key parameters]  
**Results:** [Measured metrics]  
**Artifacts:** [Paths to saved models/logs]  
**Notes:** [Observations, issues, next steps]

---

## Phase 1: Data & Features (Current Milestone)

### Exp 1.1: Data Loading Validation
**Date:** 2026-11-18  
**Phase:** P1  
**Description:** Load all 90 trips (80 train + 10 student_test), validate schemas  
**Config:** `validate=True`  
**Results:** [PENDING — run `scripts/dataset_inventory.py`]  
**Artifacts:** `experiments/dataset_inventory.json`  
**Notes:** Blocked by Python environment

### Exp 1.2: Candidate Generation Quality
**Date:** 2026-11-18  
**Phase:** P1  
**Description:** Generate candidates for train trips, measure miss rate  
**Config:** `k=10`, `radius=100m`, KDTree enabled  
**Results:** [PENDING — run `scripts/smoke_test.py`]  
**Artifacts:** `experiments/smoke_test_results.json`  
**Notes:** Roadmap reconnaissance shows trip 00000001 has 165,790 arcs

### Exp 1.3: Feature Matrix Construction
**Date:** 2026-11-18  
**Phase:** P1  
**Description:** Build feature matrices for train + student_test  
**Config:** 11 features, binary labels via proportional alignment  
**Results:** [PENDING — run smoke test]  
**Artifacts:** Feature matrices (NPZ)  
**Notes:** Verify no NaN/inf, label distribution

### Exp 1.4: Group-Aware Split Verification
**Date:** 2026-11-18  
**Phase:** P1  
**Description:** Verify GroupKFold splits have zero trip leakage  
**Config:** `n_splits=5`, `groups=trip_id`  
**Results:** [PENDING — run `pytest tests/data/test_splits.py`]  
**Artifacts:** `experiments/splits.json`  
**Notes:** Critical for no-leakage guarantee

---

## Phase 2: Model Zoo (Future)

### Exp 2.1: Baseline Models CV
**Date:** TBD  
**Phase:** P2  
**Description:** Train 5+ models with GroupKFold CV on 80 train trips  
**Config:** [Model hyperparameters]  
**Results:** [Macro F1, Top-1 Acc, LogLoss per model]  
**Artifacts:** `models/*.pkl`, `experiments/model_comparison.csv`  
**Notes:** 

### Exp 2.2: Student-Test Evaluation
**Date:** TBD  
**Phase:** P2  
**Description:** Evaluate all models on 10 student-test trips  
**Config:** [Best CV model selection]  
**Results:** [Top-1 Acc, Macro F1, LogLoss, Route Continuity]  
**Artifacts:** `experiments/student_test_eval.csv`  
**Notes:** 

### Exp 2.3: Model Selection
**Date:** TBD  
**Phase:** P2  
**Description:** Select final model with justification  
**Config:** [Selection criteria]  
**Results:** [Chosen model + rationale]  
**Artifacts:** `EXPERIMENTS.md` updated  
**Notes:** 

---

## Phase 3: Bayesian Optimisation (Future)

### Exp 3.1: Optuna Study
**Date:** TBD  
**Phase:** P3  
**Description:** Bayesian optimisation on chosen model  
**Config:** [Search space, n_trials=50-100, GroupKFold objective=Macro F1]  
**Results:** [Best params, best CV score, trial count]  
**Artifacts:** `experiments/optuna_study.pkl`, `models/best_tuned.pkl`  
**Notes:** 

### Exp 3.2: Tuned Model Refit
**Date:** TBD  
**Phase:** P3  
**Description:** Refit tuned model on full 80 trips, evaluate on student-test  
**Config:** [Best params from Optuna]  
**Results:** [Tuned vs untuned comparison]  
**Artifacts:** `experiments/tuned_comparison.csv`  
**Notes:** 

---

## Phase 4: HMM + Viterbi (Future)

### Exp 4.1: HMM Emission Calibration
**Date:** TBD  
**Phase:** P4  
**Description:** Tune λ for emission: log P = model_logprob - λ·distance  
**Config:** [λ grid search on student-test]  
**Results:** [Best λ, emission quality]  
**Artifacts:** `hmm/emission.py`  
**Notes:** 

### Exp 4.2: Transition Matrix Construction
**Date:** TBD  
**Phase:** P4  
**Description:** Build adjacency + BFS hop distances per trip  
**Config:** [max_hops=3, hop_penalty, gap_penalty]  
**Results:** [Transition sparsity, connectivity stats]  
**Artifacts:** `hmm/transition.py`, per-trip adjacency.npz  
**Notes:** 

### Exp 4.3: Viterbi Decoding
**Date:** TBD  
**Phase:** P4  
**Description:** Run Viterbi on all student-test trips  
**Config:** [Emission + transition params]  
**Results:** [Decoded route accuracy, continuity]  
**Artifacts:** `hmm/viterbi.py`  
**Notes:** 

### Exp 4.4: Classifier vs HMM Comparison
**Date:** TBD  
**Phase:** P4  
**Description:** Compare classifier-only vs classifier+HMM  
**Config:** [Same features, same test trips]  
**Results:** [Δ Top-1 Acc, Δ Route Continuity, Δ Sequence F1]  
**Artifacts:** `experiments/hmm_comparison.csv`  
**Notes:** 

---

## Phase 5: API + UI (Future)

### Exp 5.1: API Latency
**Date:** TBD  
**Phase:** P5  
**Description:** Measure end-to-end latency on Laptop 1  
**Config:** [CPU, batch size=1]  
**Results:** [p50, p95, p99 latency per trip]  
**Artifacts:** `api/main.py`, latency logs  
**Notes:** Target <3s/trip

### Exp 5.2: Two-Laptop Integration
**Date:** TBD  
**Phase:** P5  
**Description:** LAN test with Laptop 2 UI → Laptop 1 API  
**Config:** [IP, port, CORS]  
**Results:** [10/10 trips render, no errors]  
**Artifacts:** `DEMO_CHECKLIST.md`, screenshots  
**Notes:** 

---

## Measurement Standards

| Metric | Definition | Target |
|--------|------------|--------|
| Top-1 Accuracy | `mean(pred_arc_top1 == gt_arc)` | >0.70 |
| Macro F1 | `f1_score(gt, pred, average='macro')` | >0.60 |
| Log Loss | `log_loss(gt_onehot, pred_proba)` | <1.0 |
| Route Continuity | `% consecutive pred_arcs connected` | >0.80 |
| Candidate Miss Rate | `mean(gt_arc not in candidate_set)` | <0.08 |
| Inference Latency | End-to-end ms per trip | <3000ms |

---

## Artifact Storage

```
experiments/
├── dataset_inventory.json          # Phase 1
├── smoke_test_results.json         # Phase 1
├── splits.json                     # Phase 1
├── model_comparison.csv            # Phase 2
├── student_test_eval.csv           # Phase 2
├── optuna_study.pkl                # Phase 3
├── tuned_comparison.csv            # Phase 3
├── hmm_comparison.csv              # Phase 4
└── latency_logs.json               # Phase 5
```

---

## Reproducibility Checklist

- [ ] Fixed random seeds (42) in all sklearn/optuna calls
- [ ] Requirements.txt with exact versions
- [ ] Run scripts with explicit configs (no hidden defaults)
- [ ] Artifacts saved with versioned names
- [ ] Experiment logs append-only (never overwrite)

---

*Last updated: 2026-11-18 — Phase 1 implementation complete, verification pending*