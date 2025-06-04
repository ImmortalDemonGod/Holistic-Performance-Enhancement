---
**STATUS: ARCHIVED & SUPERSEDED**
This roadmap (`roadmap_vSigma.md`) has been superseded by the new canonical plan:
[`roadmap_Cultivation_Integrated_v1.0.md`](../../3_design_and_architecture/roadmap_Cultivation_Integrated_v1.0.md).
This document is retained for historical context and evolutionary reference only.
Please refer to `roadmap_Cultivation_Integrated_v1.0.md` for current project planning.
Date Archived: 2025-06-04
---

## Guiding DNA

| Principle | How v Σ implements it |
|-----------|----------------------|
| **Early Wins** | Ship *one runnable script or notebook per domain* in the first 90 days. |
| **Controlled Concurrency** | Never more than **3 active feature tracks** per phase; new tracks unlock only after passing a *Risk‑Gate* checklist. |
| **Capability Waves** | Each phase introduces a *math / tooling capability* that all active domains immediately use—maximising skill‑transfer. |
| **Formal Safety Net** | At least one Lean proof (or proof sketch) exits every phase, validating a result the code relies on. |
| **CI‑First** | Every new top‑level folder lands with a GitHub‑Actions job and a stubbed test to keep the tree green. |

---

## v Σ Road‑map Table

| Phase (≈ window) | Capability Wave 📐 | Active Feature Tracks (≤ 3) | **Milestones / Deliverables** | **Risk‑Gate ✓** *(all must pass to enter next phase)* |
|---|---|---|---|---|
| **P0 (0‑3 mo)**<br>**Bootstrap & Data Ingest** | *Linear Algebra + Time‑Series Stats* | ① Running data pipeline<br>② RNA raw‑data loader<br>③ Lean core utils | • `scripts/running/process_run_data.py` parses GPS/HR → weekly CSV<br>• `scripts/biology/load_rna_data.py` ingests FASTA/PDB → tidy parquet<br>• `Proofs/Core/Arithmetic.lean` (+ CI job) | □ Both ETL scripts emit sample datasets to `/data/` & tests pass<br>□ Lean file compiles on CI<br>□ README badge shows “Build ✔︎” |
| **P1 (3‑7 mo)**<br>**Dynamics & Geometry** | *ODE modelling + Numerical solvers* | ① Running VO₂/HR ODE fits<br>② RNA secondary‑structure EDA<br>③ Astro N‑body sandbox | • `notebooks/running/dynamics.ipynb` (logistic + biexponential recovery)<br>• `notebooks/biology/rna_geometry.ipynb` (base‑pair graphs + energy toy model)<br>• `scripts/space/two_body.py` (REBOUND demo + plot) | □ ODE fits replicate sample run within 5 % RMSE<br>□ RNA notebook builds without manual intervention<br>□ REBOUND job runs in CI with `pytest -q` |
| **P2 (7‑12 mo)**<br>**Control & Causal Coupling** | *PID / basic control + Causal DAGs* | ① Synergy PID scheduler<br>② Causal analysis notebook<br>③ Lean control lemmas | • `scripts/synergy/pid_scheduler.py` produces daily plan JSON<br>• `notebooks/synergy/causal_running_coding.ipynb` (DoWhy / DAGitty graph & ATE calc)<br>• `Proofs/Control/PID_stability.lean` | □ Scheduler passes 14‑day smoke test<br>□ ATE ≠ 0 with p < 0.1 (placeholder dataset)<br>□ Lean proof checked on CI |
| **P3 (12‑18 mo)**<br>**Optimization & Multivariate Stats** | *Convex optimisation + PCA/CCA* | ① cvxpy time‑allocator<br>② PCA dashboard (running + coding + bio)<br>③ RNA Bayesian param‑fit | • `scripts/synergy/optimize_time.py` (resource solver)<br>• `docs/4_analysis/pca_dashboard.ipynb` autodeployed with Voilà<br>• `notebooks/biology/bayes_rna_params.ipynb` (PyMC + trace plot) | □ Optimizer CI test hits ≤ 1 sec solve time on sample<br>□ Dashboard GitHub Pages auto‑publishes<br>□ Gelman‑Rubin Rˆ < 1.1 for RNA fit |
| **P4 (18‑24 mo)**<br>**ML & Formal Integration** | *Bayesian ML + RL + Stochastic calculus* | ① RL schedule agent<br>② RNA 3D coarse model (GNN stub)<br>③ PBH signal detector (Bayes change‑point) | • `scripts/synergy/rl_agent.py` (stable‑baselines PPO)<br>• `scripts/biology/rna_gnn.py` (PyTorch Geometric skeleton)<br>• `notebooks/space/pbh_detection.ipynb` (Bayesian blocks) | □ RL beats PID baseline ≥ 3 % on synthetic metric<br>□ GNN forward pass unit‑test green<br>□ Detector recall ≥ 0.8 on toy data |
| **P5 (24 mo +)**<br>**Grand Challenges** | *Astrodynamics, Game theory, High‑order ODE, ARC* | ① Full PBH encounter sim<br>② RNA 3D pipeline → AlphaFold‑style scorer<br>③ ARC solver prototypes | • `scripts/space/pbh_sim.py` (adaptive 15‑body)<br>• `pipelines/rna3d/` (diffusion + pairformer)<br>• `scripts/arc/solver_suite/` (pattern‑finder, circuit extractor)<br>• `Proofs/Astro/Orbit_error.lean` | □ CI passes w/ GPU stub runners<br>□ Any ARC sub‑benchmark ≥ 70 % solved<br>□ Lean orbit error lemma proven |

---

### How v Σ blends V1–V4

| Borrowed Strength | Source Version(s) | v Σ Implementation |
|-------------------|-------------------|--------------------|
| **Immediate multi‑domain excitement** | V1 / V3 | P0 ingests *both* Running & RNA data; P1 adds Astro sandbox. |
| **Risk‑gating & cognitive load control** | V4 | “≤ 3 active tracks” rule + explicit Risk‑Gate checklist each phase. |
| **Capability Waves (shared math per phase)** | V3 | Every phase headline is a maths/CS capability reused by all tracks. |
| **Early formal proofs for confidence** | V2 / V4 | Lean deliverable baked into *every* phase, starting with tiny utils. |
| **CI/CD first mindset** | hinted in V4 | New folder ⇒ new CI job & unit test before phase gate passes. |

---

## CI‑/DevOps add‑ons

1. **Matrix build**: `python 3.10` + `Lean nightly` + `CUDA‑off` runners.  
2. **Phase labels**: PRs must include `phase/PX` tag; GitHub Action blocks merging files from a *future* phase directory unless the tag matches.  
3. **Docs auto‑publish**: `mkdocs-material` + `gh‑pages` branch, updated each merge.

---

## Next Action (Week 0)

* Create branch `roadmap/vSigma`.
* Add this file to `docs/3_design/`.
* Stub CI workflows: `ci-ingest.yml`, `ci-lean.yml`, `ci-notebooks.yml`.
* Kick off **Phase P0 sprint‑planning** (issues + milestones on GitHub).

Once merged, v Σ becomes the source‑of‑truth plan—**greater than the sum of its parts** yet firmly anchored in the existing repo. Happy cultivating!
