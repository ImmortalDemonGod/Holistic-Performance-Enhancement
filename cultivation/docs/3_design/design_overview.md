# Cultivation – Architecture Overview (v Σ P0 → P1 reconciled)

> **Status:** Working draft • last updated 2025‑04‑18 after architecture review.  This replaces the empty `design_overview.md` and is **the canonical reference** for implementation until superseded by ADR‑series documents.

---

## 0  Reading Map
| Layer | Spec File |
|-------|-----------|
| **Vision & Theory** | `docs/1_background/` cluster |
| **Road‑map** | `docs/3_design/roadmap_vSigma.md` |
| **Math Stack** | `docs/2_requirements/math_stack.md` |
| **Architecture** | **`docs/3_design/architecture_overview.md`** *(this file)* |
| **Schemas** | `docs/2_requirements/schemas/*.json` |

---

## 1  Context Diagram (C4 level‑1)
```mermaid
flowchart LR
    subgraph Domains
        A[Running Data] -->|weekly CSV| ETL_R
        B[Software Repo] -->|git log JSON| ETL_S
        C[Bio Reading Notes] -->|papers CSV| ETL_B
    end

    subgraph ETL
        ETL_R(process_run_data.py)
        ETL_S(commit_metrics.py)
        ETL_B(analyze_literature.py)
    end

    ETL_R --> DS[(Domain Stores)]
    ETL_S --> DS
    ETL_B --> DS

    %% --- Core analytics pipeline ---
    DS --> SY(calc_synergy.py)
    SY -->|writes| SYDS[(synergy_score.parquet)]

    %% Potential engine is a **separate script in P2** but for P1 its logic is inside SY.
    SYDS --> PE[potential_engine.py*]
    PE --> POTDS[(potential_snapshot.parquet)]
    note right of PE
      *P1: stub inside SY  
      *P2: standalone script
    end note

    POTDS --> SCHED[
      PID Scheduler (P2) /  
      RL Agent (≥P4)
    ]
    SCHED --> PLAN[(daily_plan.json)]

    %% --- Observability layer ---
    DS --> DASH[Dashboards / Notebooks]
    SYDS --> DASH
    POTDS --> DASH
    PLAN --> DASH

    %% Lean / Formal verification (dotted for now)
    subgraph Formal [Lean 4 proofs]
        direction TB
        L1[Proofs/Control] -.-> SY
        L2[Proofs/ODE Fitness] -.-> ETL_R
    end

    %% Future domains (commented for now)
    %%  Astro[Astro N‑body] -->|csv| ETL_SPACE
```
*Dashed arrows* show future Lean validation hooks; greyed boxes (in mermaid comments) note Phase ≥P3 additions such as Astro & ARC.

---

## 2  Component Responsibilities (phased & concrete)
| ID | Script / Service | Phase | **Input** | **Output** | Notes |
|----|------------------|-------|-----------|------------|-------|
| **ETL_R** | `scripts/running/process_run_data.py` | **P0** | `.gpx` / `.fit` files in `data/raw/running/` | `running_weekly.csv` **and** `running_weekly.parquet` | Fast CSV is canonical; Parquet auto‑generated for analytics speed. ≤2 s per file *(FR‑R‑01)*. |
| **ETL_S** | `scripts/software/commit_metrics.py` | **P0** | Local git repo **or** GitHub API JSON | `commit_metrics.csv` / `.parquet` | Extracts LOC Δ, complexity (radon), Ruff score, pytest coverage.  Auto‑ranks commit difficulty. |
| **ETL_B** | `scripts/biology/analyze_literature.py` | **P1** | `bibtex/*.bib`, optional PDF meta JSON | `lit_reading.csv` | Fields: date, doi, field, minutes_spent, retention_quiz_score. |
| **DS** | `data/<domain>/` | P0 | Any ETL output | Partitioned **Parquet** | CSV kept for diff‑ability; Parquet for joins. |
| **SY** | `scripts/synergy/calculate_synergy.py` | **P1** | Joined Parquets (P + C) | `synergy_score.parquet` *(see schema §4.3)* | **P1 baseline** = rolling‑mean Δ; **P3 target** = SARIMA/Prophet.  Computes pair‑wise \(S_{A\to B}\). |
| **PE** | `scripts/synergy/potential_engine.py` | **P2** | `synergy_score.parquet`, domain KPI CSVs | `potential_snapshot.parquet` | Implements Π using _P_ + proxy‑C only until S/A ETLs exist (≥P3).  Monthly weight update via `update_potential_weights.py`. |
| **SCHED** | `scripts/synergy/pid_scheduler.py` | **P2** | Potential snapshot | `daily_plan.json` | Deterministic PID on first release; RL agent (`rl_agent.py`) swaps in at P4. |
| **DASH** | Streamlit app `dash_app.py` + Notebooks | P1 | All Parquets | Static HTML in `docs/4_analysis/assets/` | Built nightly in CI; friendly links in analysis docs. |
| **FORMAL CI** | `lake build/exec` | P0+ | Lean proofs | — | Proofs validate ODE stability, PID boundedness; failing proof == red CI. |

---

## 3  Formal Equations & Algorithmic Contracts
### 3.1  Synergy Score \(S_{A\to B}\)  *Eq (1)*
For ISO week *w*
\[
S_{A\to B}(w)=\Delta B_{\text{obs}}(w)-\Delta B_{\text{pred}}^{\text{baseline}}(w).
\]
**Baseline model evolution**
| Phase | Model |
|-------|-------|
| **P1** | 4‑week rolling mean of \(\Delta B\) |
| P2 | simple linear‑trend + seasonality |
| P3 | SARIMA / Prophet |
| ≥P4 | lightweight temporal GNN |
A **contract test** asserts that the baseline estimator used is declared in a YAML shebang inside `synergy_score.parquet` metadata.

### 3.2  Global Potential \(\Pi\)
\[
\Pi(P,C,S,A)=w_P P^{\alpha}+w_C C^{\beta}+\lambda \sum_{i<j}S_{i\to j}+\varepsilon.
\]
*P1/P2 note:* Only _P_ (physical) and _C_ (cognitive proxies from commits + lit) are populated; _S_ and _A_ vectors are zero‑padded until their ETLs exist.  Weights \(w,\alpha,\beta\) re‑learned monthly by ridge regression script `update_potential_weights.py` fed by a handcrafted KPI (e.g., mile‑pace delta).

---

## 4  Data Schemas (v0.2)
Full JSON schema files live in `docs/2_requirements/schemas/` (generated by `datamodel-codegen`). Below are excerpts.

### 4.1  `running_weekly.csv`
| column | dtype | description |
|--------|-------|-------------|
| `week` | string (`YYYY‑WW`) | ISO week id |
| `total_distance_km` | float | Σ distance |
| `avg_pace_sec_per_km` | float | harmonic mean |
| `avg_hr_bpm` | float | mean HR |
| `avg_power_w` | float | avg power |

### 4.2  `lit_reading.csv`
| column | dtype | description |
|--------|-------|-------------|
| `week` | string | ISO week |
| `doi` | string | paper DOI |
| `field` | string | taxonomy level‑1 |
| `minutes_spent` | int | active reading |
| `retention_quiz_score` | float | 0‑1 scaled |

### 4.3  `synergy_score.parquet`
| column | dtype | description |
|--------|-------|-------------|
| `week` | string | ISO week |
| `source_domain` | string | e.g. Running |
| `target_domain` | string | e.g. Software |
| `raw_delta` | float | \(\Delta B_{\text{obs}}\) |
| `baseline_delta` | float | baseline prediction |
| `synergy_score` | float | normalised [-1,1] |
| `baseline_model` | string | e.g. `rolling_mean_4w` |

---

## 5  Requirements Traceability (sample)
| ID | Requirement | Phase | Test Method |
|----|-------------|-------|-------------|
| **FR‑R‑01** | ETL_R processes 1 Gar‑min `.fit` file ≤2 s. | P0 | `pytest-benchmark` |
| **FR‑SY‑01** | SY writes ≥3 scored pairs per run. | P1 | integration test fixture |
| **NFR‑FMT‑01** | All Parquet files respect defined schema. | P0 | `great_expectations` in CI |
| **NFR‑LINT‑01** | `ruff check` passes zero errors. | P0 | CI step |

---

## 6  Testing & Quality Gates (phased)
| Layer | Tool | Starts in |
|-------|------|----------|
| Unit | `pytest` | P0 |
| Property‑based | `hypothesis` | **P1** |
| Data contract | `great_expectations` | P1 (once schemas stable) |
| Perf bench | `pytest‑benchmark` | P0‑selective |
| Formal proofs | Lean 4 | P0 |
| Notebook run | `nbconvert --execute` | P1 |
| Regression | `pytest‑regressions` | P2 |

_Lean in CI_: `lake build` ensures all proofs compile; targeted `lake exec` could run `#check` on theorems tied to algorithms (e.g., PID stability).  Failure == ❌ CI.

---

## 7  CI / CD (GH Actions v0.2)
```yaml
env:
  PYTHON_VERSION: '3.11'

name: CI
on: [push, pull_request]
jobs:
  lint-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: ${{ env.PYTHON_VERSION }}}
      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: pytest -q
  lean:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: leanprover/lean4-action@v1
      - run: lake build
  docs:
    runs-on: ubuntu-latest
    needs: lint-test
    steps:
      - uses: actions/checkout@v4
      - run: pip install mkdocs-material
      - run: mkdocs build --strict
```
Future matrix jobs: parquet schema validation, notebook execution, GPU RL training.

---

## 8  Open Questions & Upcoming ADRs
| # | Topic | Notes |
|---|-------|-------|
| 1 | **Data privacy** | GPS traces → keep only weekly aggregates in repo. Raw uploads auto‑purged (ADR‑0001). |
| 2 | **Baseline model upgrade path** | Rolling‑mean (P1) → SARIMA/Prophet (P3) — ADR‑0002 will record criteria. |
| 3 | **Π weights learning** | Ridge vs. Bayesian — ADR‑0003 (P2). |
| 4 | **RL agent state space** | Use Potential snapshot or raw features — ADR‑0004 (P3). |

---

### Next Steps
1. **Approve** this overview → merge into `main`.
2. **Open issues** auto‑generated from FR/NFR table.
3. **Implement** ETL_R & ETL_S with schemas + unit tests.
4. **Wire CI** (ruff + pytest + Lean) to keep the badge green.

