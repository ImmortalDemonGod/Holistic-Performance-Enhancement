# Mentat-OS: 4-Week MVP Pilot Plan & Integration Roadmap

**Document ID:** `MENTATOS-PILOT-V1.0`
**Version:** 1.0
**Date Created:** 2025-06-11
**Status:** Proposed for Immediate Action
**Parent Blueprint:** [`mentat_os_blueprint.md`](./mentat_os_blueprint.md)

## 1. Pilot Objective & Scope

### 1.1. Objective
To execute a focused, 4-week pilot program to bootstrap the **Mentat-OS Cognitive Augmentation Domain**. The primary goals of this pilot are to:
1.  **Establish Baseline Performance:** Quantify the user's current capabilities across a range of core cognitive metrics.
2.  **Implement Core Training & Measurement Systems:** Build the initial software and processes for training, automated scoring, and data logging.
3.  **Validate Training Efficacy:** Assess whether the initial set of cognitive drills produces measurable improvements in their associated KPIs over the 4-week period.
4.  **Generate Initial Data for HIL:** Produce the first `cognitive_training_weekly.parquet` files for future integration into the Holistic Integration Layer (HIL).

### 1.2. Scope
*   **In Scope:**
    *   Implementation of the **behavioral** training and assessment protocols for the Mentat-OS.
    *   Development of the `mentat_autograder.py` script and the KPI Dashboard (Google Sheet or similar).
    *   Daily execution of the specified cognitive drills.
    *   Weekly data processing and analysis of pilot results.
*   **Out of Scope (for this 4-week pilot):**
    *   Neuroimaging (fMRI/EEG).
    *   Full, real-time integration with the Potential Engine (Π) and Scheduler. The pilot will *generate* data for this integration, which will occur in a subsequent phase.
    *   Training of advanced social dynamics or ethical governance layers (focus is on the Cognitive Core and foundational drills).

### 1.3. Success Criteria for Pilot Completion
The pilot will be considered successful if the following are achieved by the end of Week 4:
1.  **System Implementation:** The `mentat_autograder.py` script and KPI Dashboard are functional and have been used to log all pilot data.
2.  **Training Adherence:** The user has completed at least 80% of the scheduled daily drill sessions.
3.  **Measurable Improvement:** Statistically significant improvement (or meeting predefined thresholds) is observed on at least 3 of the 5 core cognitive KPIs being tracked (e.g., `WM-Span`, `Logic-Acc`, `Parity-EC`).
4.  **Data Generation:** At least four weekly `cognitive_training_weekly.parquet` files have been successfully generated.

---

## 2. The 4-Week Pilot Schedule & Milestones

| Week | Focus & New Elements | Daily Commitment (approx.) | End-of-Week Gate & Deliverable |
| :--- | :--- | :--- | :--- |
| **Wk 1** | **Foundation & Baseline:** Establish baseline performance. Achieve proficiency in foundational drills: `D1` (Rapid Recode), `D3` (WM Span), `D4` (Mental Algorithm Trace - a better name for Parity Chant's logic training). | 20 min drills | `WM-Span` score consistently ≥ 16 digits. Baseline scores for all KPIs logged in the dashboard. |
| **Wk 2** | **Quantification & Heuristics:** Introduce probabilistic reasoning drills: `D5` (Mental Abacus) and `D6` (Fermi Ladder). Begin tracking speed and precision. | 20 min drills | `Fermi-RMSE` score consistently ≤ 3.0. `Math-SPS` shows a positive trend over baseline. |
| **Wk 3** | **Human-AI Interface & Error Checking:** Introduce AI hand-offs. Practice using `D2` (Forced Analogy) to generate creative prompts. Use `D4` drills to mentally verify AI outputs. | 25 min (drills + AI interaction) | `Parity-EC` (Error Catch Rate, a new KPI for this) ≥ 75% on AI-generated content with seeded errors. |
| **Wk 4** | **Full Pipeline Integration Mini-Project:** Execute a full decision-making task (e.g., "GPU Buy vs. Rent decision") using all relevant Mentat-OS layers. Log time meticulously to calculate `Synergy-ROI`. | 30 min (mini-project) | `Synergy-ROI` ≥ 2:1. `Logic-Acc` (from a test on project-related reasoning) ≥ 90%. Final pilot report drafted. |

---

## 3. Instrumentation & Data Management Plan

1.  **Drill Autograder (`mentat_autograder.py`):**
    *   This Python script is the core of the measurement system.
    *   It will be developed in `scripts/cognitive_training/`.
    *   It will be callable from the command line: `python scripts/cognitive_training/mentat_autograder.py --user <id> --drill <drill_id> --payload '<json_data>'`.
    *   It will contain the grading logic for each drill and will append a new row to `data/cognitive_augmentation/raw/cognitive_drill_log.csv` upon execution.

2.  **KPI Dashboard:**
    *   An initial dashboard will be set up in a Google Sheet.
    *   **Sheet 1: `raw_logs`** - This sheet will be the direct target for the `mentat_autograder.py` script (using the `gspread` library). It will mirror the `cognitive_drill_log.csv` schema.
    *   **Sheet 2: `kpi_dashboard`** - This sheet will use pivot tables and formulas (`AVERAGEIF`, `VLOOKUP`, etc.) to pull from `raw_logs` and display weekly trends, current KPI values against targets, and pass/fail conditional formatting.

3.  **Weekly ETL Process:**
    *   At the end of each week, the `process_cognitive_drills.py` script will be run manually.
    *   It will read the `cognitive_drill_log.csv` file, perform the weekly aggregation, and generate the corresponding `cognitive_training_weekly.parquet` file in the `processed` data directory.

---

## 4. Integration Roadmap into the `cultivation` Project

This pilot plan necessitates the creation of several artifacts and tasks within the main repository. These `DW_` tasks should be added to `tasks.json` to be formally scheduled and tracked.

| Layer | Integration Tasks & `DW_` IDs | Target Phase |
| :--- | :--- | :--- |
| **1. Documentation** | `DW_COG_DOCS_001`: Create the `cognitive_augmentation` directory. Add `README.md`, `mentat_os_blueprint.md`, `drill_cookbook.md`, and this `pilot_plan.md`. | P2 (Immediate) |
| **2. Data Schemas** | `DW_COG_SCHEMA_001`: Create `cognitive_training_schemas.md` defining the raw log and processed Parquet schemas. | P2 (Immediate) |
| **3. Scripts & ETL** | `DW_COG_SCRIPT_001`: Implement the `mentat_autograder.py` script, including logic for scoring drills D1-D8 and pushing to the KPI Dashboard. <br> `DW_COG_ETL_001`: Implement the `process_cognitive_drills.py` script to generate the weekly Parquet file. | P2 |
| **4. HIL Integration** | `DW_HIL_INTEG_COG_001`: **(Post-Pilot)** Update `potential_engine.py` to ingest the generated `cognitive_training_weekly.parquet` and factor its KPIs into the `C(t)` score. | P3 |
| **5. Task Management** | `DW_TASK_GEN_COG_001`: **(Post-Pilot)** Update `task_generator.py` to create recurring tasks for daily cognitive drills in `tasks.json`. For the pilot, tasks can be added manually. | P3 |

---

## 5. Immediate Next Actions: The Kick-off Checklist

To commence the pilot, the following actions must be taken within the first few days of Week 1:

1.  **[ ] Create Directory Structure & Initial Docs:**
    *   Create the `cognitive_augmentation` directories in `docs/`, `data/`, and `scripts/`.
    *   Add the `README.md`, `mentat_os_blueprint.md`, `drill_cookbook.md`, and this `pilot_plan.md` to the new docs directory.

2.  **[ ] Scaffold Critical Artifacts:**
    *   Create the `cognitive_training_schemas.md` document.
    *   Create the initial `mentat_autograder.py` script with function stubs.
    *   Set up the KPI Dashboard in Google Sheets.

3.  **[ ] Conduct Day-0 Baseline Testing:**
    *   Block 30-45 minutes to perform baseline measurements for all relevant KPIs (`WM-Span`, `Logic-Acc`, `Math-SPS`, etc.).
    *   Manually log these initial scores in the dashboard.

4.  **[ ] Schedule Daily Drills:**
    *   Manually create the recurring tasks for the Week 1 drills in your Task Master.

Upon completion of this checklist, the pilot is officially underway. This systematic approach ensures that the ambitious vision of the Mentat-OS is translated into a concrete, measurable, and integrated component of the "Cultivation" project.
