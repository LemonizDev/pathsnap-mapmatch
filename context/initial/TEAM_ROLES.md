# PathSnap — Team Roles & Delegation

**Milestone:** 11:30 Evaluation (Preprocessing & Feature Engineering Baseline)  
**Date:** 18 November 2026

---

## Team Composition

| Role | Person | Core Responsibilities | Current Focus |
|------|--------|----------------------|---------------|
| **ML & Data Lead** | *TBD* | Data loading, EDA, candidate generation, feature engineering, label construction, train/val splits, leakage prevention | **COMPLETED**: Data loader, candidate gen, features, splits |
| **Model & Algorithm Lead** | *TBD* | Model zoo (5+), CV, selection, Bayesian optimisation, HMM design (emissions/transitions), Viterbi, comparison experiments | **PENDING**: Phase 2+ |
| **Backend & Integration Lead** | *TBD* | FastAPI server, artifact loading, UI (Streamlit/Dash/HTML+JS), two-laptop deployment, evaluator-side accuracy, CORS, latency optimisation | **PENDING**: Phase 5+ |
| **Validation / Deployment / Demo Lead (Coordinator)** | **Nimit Jain** | Project coordination, timeline, risk management, report writing, GitHub repo hygiene, demo rehearsal, academic integrity compliance | **ACTIVE**: Milestone integration, documentation |

---

## Delegation Matrix (from Roadmap)

| Role | Must Understand | Handoffs (To → From) | Presentation Talking Points |
|------|-----------------|----------------------|----------------------------|
| **ML & Data Lead** | Raw file formats, coordinate systems, candidate-gen geometry, feature definitions, split strategy | → Model Lead: feature matrices + labels + splits<br>← Backend: API input schema for features | "How we turn raw GPS + road network into ML-ready features — our candidate generation finds K nearest road segments per point using perpendicular distance, features capture geometry + topology + temporal cues" |
| **Model & Algorithm Lead** | All model configs, CV scheme, Optuna search space, HMM math (emission/transition formulas, Viterbi recurrence), why chosen model works | → Backend: tuned model artifact + HMM params + Viterbi code<br>← ML Lead: features + labels + splits | "We compared 6 models across 3 algorithm families; LightGBM won on Macro-F1 + calibrated probabilities; Bayesian opt gave 2.3% F1 gain; HMM+Viterbi adds 4.1% route continuity by enforcing network connectivity" |
| **Backend & Integration Lead** | API contract (request/response), model/HMM serialization, UI→API flow, network topology (LAN IP, ports), Folium map rendering in UI | ← Model Lead: model + HMM artifacts + inference code<br>→ Coordinator: demo checklist, screenshots, run scripts | "Laptop 1 runs FastAPI with model+HMM loaded in memory; Laptop 2 UI fetches 10 evaluator trip IDs, on selection calls `/predict`, renders interactive Folium map with raw GPS (blue), predicted route (green), ground truth revealed on demand" |
| **Coordinator** | **Everything at integration level**: data→features→model→HMM→API→UI, evaluation metrics, leakage rules, submission checklist | ← All: task status, blockers<br>→ All: priorities, deadlines, integration checkpoints | "Our system reconstructs routes from noisy GPS in <2s/trip; we'll demo live on two laptops — evaluator picks any of 10 trips, sees predicted route vs ground truth, accuracy computed server-side. Report documents all experiments with reproducible code." |

---

## Sync Schedule (Review Checkpoints)

| Checkpoint | Target Date | Participants | Gate Criteria |
|------------|-------------|--------------|---------------|
| **CP1: Data Ready** | **Today (11:30)** | All | Feature matrices saved, splits verified no leakage, EDA plots committed |
| **CP2: Model Selected** | Day 3–4 | Model Lead + Coordinator | 5+ models compared, best chosen with evidence, tuned model saved |
| **CP3: HMM Working** | Day 7–8 | Model Lead + Backend Lead | Classifier-only vs +HMM comparison done, Viterbi decodes all student-test |
| **CP4: API+UI Integrated** | Day 10–11 | Backend Lead + Coordinator | Local API+UI works for all 10 student-test trips |
| **CP5: Two-Laptop Demo** | Day 12–13 | All | LAN demo works, latency <3s, all 10 evaluator trips render |
| **CP6: Submission Ready** | Day 14 | All | Report PDF done, GitHub public, reproducibility verified |

---

## Communication Protocol

- **Daily standup**: 15 min, async via shared board/notes
- **Blockers**: Post immediately to shared board with `HOLD` tag
- **Code reviews**: Required for all merges to main branch
- **Documentation**: Update `TASKS.md` and `EXPERIMENTS.md` with every completed task

---

## Contact Information

| Role | Name | Contact | Availability |
|------|------|---------|--------------|
| Coordinator | Nimit Jain | *TBD* | Full-time |
| ML & Data Lead | *TBD* | *TBD* | *TBD* |
| Model & Algorithm Lead | *TBD* | *TBD* | *TBD* |
| Backend & Integration Lead | *TBD* | *TBD* | *TBD* |

---

## Notes

- Roles marked *TBD* need team assignment
- This milestone (11:30) completed by Coordinator acting as ML/Data Lead
- Subsequent milestones require full team activation
- All work must comply with Academic Integrity Policy — independent implementation, documented decisions