Okay, this is the culmination of our analysis. The goal is to produce a definitive, comprehensively revised phased development plan for the "Holistic Performance Enhancement (Cultivation)" project. This document will integrate the strengths of all previous versions and address their identified weaknesses, aiming to be "better than the sum of its parts." It will be verbose and detailed, suitable for technical documentation.

**Filename:** `roadmap_Cultivation_Integrated_v1.0.md`
**Proposed Location:** `cultivation/docs/3_design_and_architecture/roadmap_Cultivation_Integrated_v1.0.md`

---

# **Cultivation Project: Integrated Phased Development Plan v1.0**

**Document Version:** 1.0
**Date:** 2025-06-04
**Status:** Proposed Canonical Roadmap
**Lead Author (Synthesizer):** AI Language Model (based on user inputs and codebase analysis)

## 0. Introduction & Guiding Philosophy

This document outlines the revised, integrated phased development plan for the "Holistic Performance Enhancement (Cultivation)" project. It supersedes previous roadmap iterations and incorporates insights from a comprehensive analysis of the existing codebase, extensive design documentation, detailed deep work task plans, and external repository considerations.

**The Cultivation project's grand vision** is to create a deeply integrated, data-driven system for systemic personal development and research across diverse domains including, but not limited to: Running Performance, Biological & General Knowledge Acquisition, Software Engineering Ability, and Strength Training. The core philosophy emphasizes rigorous quantification ("if it isn’t measured, it effectively isn’t real"), the discovery and leveraging of synergistic interactions between domains, and a commitment to formal rigor where appropriate.

**This revised plan is guided by the following principles:**

1.  **Solid Foundations First:** Prioritize the establishment of a robust, project-wide Infrastructure & Automation (IA) Layer and meticulously defined Data Contracts before significant feature development.
2.  **Iterative Value Delivery:** Each phase must deliver tangible, usable increments of functionality or insight, enabling early feedback and continuous system utility.
3.  **Early Holistic Visibility:** Enable an end-to-end (though initially simplified) data flow from domains to a basic Holistic Integration Layer (HIL) and a user-facing dashboard as early as pragmatically possible.
4.  **Realistic Phasing & Scope Management:** Break down complex development into manageable phases with clear objectives and achievable deliverables within estimated timeframes.
5.  **Dependency-Aware Sequencing:** Structure task execution to respect critical inter-component and inter-domain dependencies.
6.  **User-Centric Development (N=1 Focus):** Ensure the system provides tangible value to the primary developer/user at each stage, facilitating their own "cultivation" process.
7.  **Testability, Maintainability, and Documentation:** Embed quality assurance and comprehensive documentation as integral, ongoing activities throughout the development lifecycle.
8.  **Strategic Integration of External Assets:** Leverage the powerful capabilities of associated external repositories (`DocInsight`, `RNA_PREDICT`, `pytest-fixer`, `Simplest_ARC_AGI`, `PrimordialEncounters`) at appropriate junctures, with clear integration points and consideration for their development status.
9.  **Risk Management:** Employ explicit Risk-Gate Checklists at the end of each phase to ensure objectives are met before proceeding.
10. **Alignment with Deep Work Plans:** Utilize the existing granular tasks detailed in the `deep_work_candidates/task_plans/*.json` files, re-sequencing and augmenting them as needed.

This plan aims to be a living document, subject to review and adaptation based on progress, new insights, and evolving project priorities, facilitated by the `DW_HIL_META_001` (Maintain Project Analysis & Deep Work Task Elicitation) process.

## 1. Core System Components Recap

The Cultivation system is envisioned to comprise several interconnected layers and components:

*   **Domain-Specific ETLs:** Pipelines for Running, Software Dev Metrics, Literature, Flashcards, Strength Training, Wellness, etc.
*   **Data Stores:** Standardized Parquet files and databases (DuckDB, SQLite) for processed domain data.
*   **Infrastructure & Automation (IA) Layer:** CI/CD, task runners, pre-commit hooks, logging, security, documentation site, environment management, data contract validation.
*   **Knowledge Systems:** Literature Pipeline (LPP with DocInsight), Flashcore (FSRS flashcards), Instrumented Reading App, Formal Study Curricula (Mathematical Biology, RNA Modeling CSM).
*   **Holistic Integration Layer (HIL):** Synergy Engine (S_A→B), Global Potential Engine (Π), Adaptive Scheduler (PID, then RL), Focus Predictor (biosensors).
*   **User Interfaces:** Domain-specific dashboards, a Unified Holistic Dashboard, CLIs.
*   **Formal Methods Layer:** Lean 4 proofs for critical algorithms.
*   **Knowledge Creation & Validation (KCV) Layer:** Future advanced R&D capabilities (Knowledge Graph, Hypothesis Engine, Simulation Environment).

## 2. Pre-Phase 0: Critical Prerequisites & Initial Setup

Before commencing Phase 0, the following must be in place or addressed:

1.  **Project Repository:** The `Holistic-Performance-Enhancement` Git repository initialized and accessible.
2.  **Core Documentation Review:** Familiarity with key vision and design documents (`The_Dao_of_Cultivation...`, `project_philosophy_and_core_concepts.md`, `architecture_overview.md`, `roadmap_vSigma.md`).
3.  **Deep Work Task Plans:** All `deep_work_candidates/task_plans/*.json` files reviewed and understood as the primary source of granular tasks.
4.  **External Repository Status Check (Initial):**
    *   **DocInsight Backend:** Confirm accessibility and basic functionality of a development/test instance of the DocInsight RAG service (from the separate `ImmortalDemonGod/DocInsight` repository). If not ready, a mock server interface must be prioritized in P0 for LPP client development.
    *   **Task Master AI CLI:** Ensure the `task-master-ai` CLI tool is installable and fundamentally operational for basic task listing and status updates, as `pid_scheduler.py` and other components plan to interact with it.
5.  **Development Environment:** Primary development machine setup with Python (target 3.11+), Git, and preferred IDE (VS Code recommended for Lean 4 integration).
6.  **Secrets Management Strategy (Initial):** Decision on how local `.env` files and GitHub repository secrets will be initially managed.

---

## 3. Phased Development Plan

### **Phase 0: "Operational Bedrock" - IA Foundations & Core Data Contracts (Est. Duration: 2-3 Months)**

*   **Theme:** Establish an unbreakable, automated, and well-documented engineering foundation for the entire project. Define how data will flow and be validated.
*   **Capability Wave Focus (Alignment with `roadmap_vSigma.md`):** Foundational Tooling & Process Standardization.
*   **Key Objectives:**
    1.  Implement a comprehensive, project-wide IA Layer MVP (CI, Task Runner, Pre-commit, Docs Site, Logging, Security, Env Management, Centralized Config).
    2.  Define and document formal Data Contracts (schemas and validation rules) for all key inter-component data artifacts planned for P0 and P1.
    3.  Establish the Lean 4 formal methods project structure and CI build.
    4.  Populate critical missing requirements documentation.
*   **Major Development Tracks & Specific Tasks:**

    1.  **Infrastructure & Automation (IA) Layer - Foundational Setup:**
        *   **(Ref: `ia_layer_plan.json`)**
        *   `DW_IA_CI_001`: Establish Comprehensive Project-Wide CI/CD Strategy and Reusable Workflow Templates (Python setup, linting, basic testing, artifact handling).
        *   `DW_IA_TOOLING_001`: Implement Standardized Project Task Runner (Decision: **Taskfile.yml** based on Go Task for cross-platform compatibility and YAML syntax).
        *   `DW_IA_PRECOMMIT_001`: Implement Comprehensive Pre-Commit Hook Framework (Python formatters/linters [Black, Ruff, isort], YAML/MD linters, secrets scan [detect-secrets], large file check, end-of-file/trailing whitespace).
        *   `DW_INFRA_DOCS_TOOLING_001`: Setup Automated Documentation Generation and Publishing (Decision: **MkDocs with Material theme** and `generate_nav.py` integration, deploying to GitHub Pages).
        *   `DW_INFRA_LOGGING_001`: Implement Standardized Project-Wide Python Logging Framework (e.g., `cultivation.utils.logger_setup`).
        *   `DW_INFRA_SECURITY_001`: Standardize Secrets Management (formalize `.env` usage with `.env.template`, GitHub Secrets for CI). Integrate `detect-secrets` into pre-commit/CI. Create `SECURITY.md`.
        *   `DW_INFRA_ENV_MGMT_001`: Define and Document Comprehensive Multi-Environment Management Strategy (Python `.venv`, Node.js via `nvm`, Elan/Lake for Lean, Docker for services).
        *   **`NEW_TASK: DW_IA_CONFIG_001`**: Design and implement a basic centralized configuration loader (e.g., Python module loading a root `config.yaml`, allowing environment-specific overrides and per-script parameter access).
        *   **Rationale:** This provides a consistent, high-quality development environment from day one, accelerating all future work and reducing errors.

    2.  **Data Contracts & Schema Definition:**
        *   **(Ref: `DW_INFRA_TESTING_001` part, adapted from V1/V5)**
        *   **`NEW_TASK: DW_DC_DEFINE_SCHEMAS_001`**: Create and populate `cultivation/docs/2_requirements_and_specifications/data_contracts.md`. Define formal data schemas (using Pandera for Parquet/CSV, JSON Schema for JSON) for ALL primary data artifacts generated by ETLs and consumed by HIL components in P0 and P1. This includes:
            *   `daily_wellness.parquet` (from HabitDash)
            *   `running_weekly.parquet`, `run_summary_YYYYMMDD_*.csv` (from Running ETL)
            *   `strength_summary.parquet`, `strength_sessions.parquet`, `strength_exercises_log.parquet` (from Strength ETL)
            *   `commit_metrics.parquet` or `dev_metrics_YYYY-MM-DD.csv` (from DevDailyReflect ETL)
            *   `reading_stats.parquet`, `literature/metadata/*.json`, `literature/db.sqlite` key tables (from Literature Pipeline)
            *   `flashcore_analytics.parquet`, `flash.db` schema (from Flashcore)
            *   HIL outputs: `synergy_score.parquet`, `potential_snapshot.parquet`, `daily_plan.json`.
        *   **Rationale:** Clear, version-controlled data contracts are essential for decoupled development, robust data pipelines, and reliable HIL operation. Defining these upfront guides ETL implementation.

    3.  **Formal Methods - Foundational Setup:**
        *   **(Ref: `formal_methods_plan.json`)**
        *   `DW_FM_001`: Setup Core Lean 4 Project (`cultivation/lean/`) with `lakefile.lean`, `lean-toolchain` (pinning versions), and CI workflow (via `DW_IA_CI_001` templates) for `lake build` and mathlib caching. Create `Cultivation/Core/Common.lean` with placeholder definitions or very basic utility proofs.
        *   **Rationale:** Establishes the "Formal Safety Net" principle early.

    4.  **Critical Documentation - Requirements:**
        *   **`NEW_TASK: DW_DOCS_POPULATE_REQS_001`**: Populate `cultivation/docs/2_requirements_and_specifications/functional_requirements.md` and `non_functional_requirements.md` with at least the high-level P0-P2 requirements derived from existing design documents (`project_philosophy_and_core_concepts.md`, `architecture_overview.md`, `roadmap_vSigma.md`).
        *   **Rationale:** Addresses a critical documentation gap, providing a formal basis for system validation.

*   **Phase 0 Risk-Gate Checklist (Exit Criteria):**
    *   ✅ All P0 IA Layer tasks (CI strategy/templates, Task Runner, Pre-commit hooks, Docs Site auto-build/deploy, Logging framework, Secrets Mgt policy, Env Mgmt guide, Centralized Config loader) implemented and documented.
    *   ✅ `data_contracts.md` created and populated with initial schemas for P0/P1 data artifacts.
    *   ✅ Lean 4 project skeleton initialized in `cultivation/lean/`, `lakefile.lean` configured, and basic CI build passes.
    *   ✅ `functional_requirements.md` and `non_functional_requirements.md` populated with initial high-level content.
    *   ✅ All P0 code passes CI checks (linting, formatting).

---

### **Phase 1: "Domain Activation & Initial HIL Data Flow" - Getting Data for Holistic Insight (Est. Duration: 3-5 Months)**

*   **Theme:** Implement robust ETL pipelines for all core domains, ensuring outputs conform to data contracts. Develop initial analytics for each. Begin flowing this data into a simplified HIL.
*   **Capability Wave Focus:** Data Engineering Excellence, Initial Domain Analytics.
*   **Key Objectives:**
    1.  Operationalize schema-validated ETL pipelines for Running, DevDailyReflect, Strength Training, basic Literature processing, and core Flashcard data.
    2.  Implement MVP analytics for each domain, outputting standardized Parquet files.
    3.  Develop initial, simplified versions of the Synergy and Potential engines capable of consuming this data.
    4.  Launch a basic, read-only Holistic Dashboard MVP.
*   **Major Development Tracks & Specific Tasks:**

    1.  **Domain ETL Implementation & Maturation (Adhering to P0 Data Contracts):**
        *   **Running System:**
            *   Solidify and thoroughly test existing scripts (`process_all_runs.py`, `parse_run_files.py`, `metrics.py`, `walk_utils.py`, `weather_utils.py`).
            *   Ensure `running_weekly.parquet` (via `aggregate_weekly_runs.py`) output is stable, conforms to its P0-defined schema, and schema validation (Pandera) is integrated into its CI test.
            *   Operationalize `fatigue_watch.py` (GitHub issue creation from `daily_wellness.parquet` and `subjective.csv`) and `pid_scheduler.py` (for *running plans only*, reading `training_plans/*.csv`, interacting with Task Master CLI as per `DW_TASKMASTER_CLI_WRAP_001`, handling phase gating via GitHub API from `DW_HIL_SCHED_001`'s existing logic).
        *   **Software Dev Metrics (DevDailyReflect):**
            *   `(Ref: dev_daily_reflect_plan.json)`
            *   `DW_DDR_CORE_001`: Robustify core Git ingestion (`ingest_git.py`) and metric processing (`commit_processor.py`). Ensure `dev_metrics_YYYY-MM-DD.csv` (or a `commit_metrics.parquet` rollup) conforms to its P0-defined schema and is validated in CI. Ensure daily report generation and commit via `.github/workflows/daily_dev_review.yml` is reliable.
        *   **Strength Training System (New ETL & Basic Analytics):**
            *   `(Ref: strength_data_schemas.md, deep_work_gaps.md)`
            *   **`NEW_TASK: DW_STRENGTH_ETL_001`**: Robustly implement `ingest_yaml_log.py` and `log_strength_session.py`. Ensure `strength_sessions.parquet` and `strength_exercises_log.parquet` conform to P0-defined schemas.
            *   **`NEW_TASK: DW_STRENGTH_ANALYTICS_001`**: Develop script to process detailed strength logs into a weekly summary `strength_summary.parquet` (e.g., total volume, sets per muscle group, avg RPEs), conforming to its P0-defined schema and validated in CI.
        *   **Knowledge - Flashcore (Core Ingestion & Basic Analytics):**
            *   `(Ref: flashcore_plan.json)`
            *   `DW_FC_CORE_001` (Pydantic Models), `DW_FC_CORE_002` (Database Layer), `DW_FC_YAML_001` (YAML Processor): Complete final audit, hardening, and exhaustive testing of these already substantially developed components.
            *   `DW_FC_CLI_001`: Implement `tm-fc ingest` CLI (wrapping `ingest_flashcards.py`).
            *   **`NEW_TASK: DW_FC_ANALYTICS_MVP_001`**: Implement basic analytics functions in `flashcore.analytics.py` to query `flash.db` (e.g., cards due, new cards, basic retention estimate if FSRS is not yet implemented), outputting `flashcore_analytics.parquet` conforming to P0-defined schema and validated in CI.
        *   **Knowledge - Literature Pipeline (Core Ingestion & Metrics):**
            *   `(Ref: literature_pipeline_plan.json)`
            *   `DW_LIT_CLIENT_001_FINALIZE`: Finalize `docinsight_client.py`.
                *   **External Dependency:** Requires a stable, developer-managed instance or a robust mock of the DocInsight backend service for testing.
            *   `DW_LIT_INGEST_001_FETCH_PAPER`: Finalize `fetch_paper.py` for single paper ingest (download PDF, extract arXiv metadata, create note, submit job to DocInsight, store `docinsight_job_id` in metadata JSON). *No polling for results in this script.*
            *   `DW_LIT_INGEST_002_DOCINSIGHT_POLLER`: Finalize `process_docinsight_results.py` as an asynchronous worker to poll DocInsight for results of submitted jobs, updating metadata JSONs and note files.
            *   `DW_LIT_METRICS_001_WEEKLY_AGGREGATE`: Refactor `metrics_literature.py` (using `literature/db.sqlite` from `reading_session.py` for session telemetry, and metadata JSONs for paper info) to produce weekly aggregated `reading_stats.parquet` conforming to P0-defined schema and validated in CI. (Initial `reading_session.py` telemetry capture can be basic for P1).

    2.  **Holistic Integration Layer (HIL) - Initial Data Consumption & Dashboard:**
        *   `(Ref: hil_kcv_plan.json)`
        *   `DW_HIL_CORE_001` (Synergy Engine - P1 Simplified): Implement a *simplified* `calculate_synergy.py`. For P1, focus on calculating and storing *domain-specific KPI trends* (e.g., 4-week rolling average of Running EF, DevDailyReflect Net LOC, LPP papers read) rather than complex `S_A→B` calculations. Output `domain_kpi_trends.parquet`.
        *   `DW_HIL_CORE_002` (Potential Engine - P1 Simplified): Implement a *simplified* `potential_engine.py`. For P1, calculate Π as a weighted sum of the latest values from `running_weekly.parquet`, `strength_summary.parquet`, `commit_metrics.parquet`, `reading_stats.parquet`, `flashcore_analytics.parquet`. Use manually configured weights. Output `potential_snapshot.parquet`.
        *   **`NEW_TASK: DW_UX_DASHBOARD_MVP_001`**: Develop an MVP "Cultivation Dashboard" (e.g., using Streamlit or Dash) to display:
            *   Latest Π score and its trend (from `potential_snapshot.parquet`).
            *   Key KPIs from each active domain (from their respective weekly/daily Parquets).
            *   Trends from `domain_kpi_trends.parquet`.
            *   (Read-only, no control/scheduling features yet).

    3.  **Infrastructure & Automation (IA) Layer - Testing & Data Validation:**
        *   `(Ref: ia_layer_plan.json)`
        *   `DW_INFRA_TESTING_001` (Full for P1 ETLs): Ensure Pandera schema validation is implemented and enforced in CI for all Parquet files produced by P1 ETLs and consumed by HIL. Establish initial code coverage targets (>70%) for ETL scripts.
        *   `DW_IA_IDEMPOTENCY_001` (Initial Audit): Audit and enforce idempotency for P1 data ingestion scripts that modify state (e.g., `sync_habitdash.py`, `fetch_arxiv_batch.py` state file).

*   **Phase 1 Risk-Gate Checklist (Exit Criteria):**
    *   ✅ ETL pipelines for Running, DevDailyReflect, Strength, Literature (ingest, DocInsight poll, weekly metrics), and Flashcore (ingest, basic analytics) are functional, automated via Task Runner, and their Parquet outputs are schema-validated in CI.
    *   ✅ `pid_scheduler.py` (running plans) and `fatigue_watch.py` are operational.
    *   ✅ Simplified Synergy Engine (KPI trends) and Potential Engine (weighted KPI sum) produce `domain_kpi_trends.parquet` and `potential_snapshot.parquet` respectively.
    *   ✅ Basic Unified Dashboard MVP displays Π and key domain KPIs.
    *   ✅ All P1 code meets initial test coverage targets.
    *   ✅ `data_contracts.md` updated with all P1 schemas.

---

### **Phase 2: "Holistic Control & Core User Experience" - Maturing HIL and Domain Functionality (Est. Duration: 4-6 Months)**

*   **Theme:** Implement adaptive scheduling via a generalized PID controller, deliver core user-facing functionalities for knowledge systems, and enhance domain-specific analytics and insights.
*   **Capability Wave Focus:** PID/Basic Control, ODE Modeling Formalization.
*   **Key Objectives:**
    1.  Implement a functional, multi-domain PID Scheduler driven by Π.
    2.  Deliver full MVP for Flashcore (FSRS scheduling, CLI review, exporters) and Literature Pipeline (automated batch fetching, instrumented reader, search CLI).
    3.  Formalize initial ODE models in Lean 4.
    4.  Enhance DevDailyReflect with coaching insights.
    5.  Integrate curriculum parsing for task management.
*   **Major Development Tracks & Specific Tasks:**

    1.  **Holistic Integration Layer (HIL) - Scheduling & Refined Synergy/Potential:**
        *   `(Ref: hil_kcv_plan.json)`
        *   **Refine `DW_HIL_CORE_001` (Synergy Engine - P1 Full):** Implement `S_A→B(w) = ΔB_obs(w) - ΔB_pred_baseline(w)` using a 4-week rolling mean for `ΔB_pred_baseline(w)`. Focus on (Running ↔ Software Dev) and (Running ↔ Cognitive Metrics) initially. Output `synergy_score.parquet`.
        *   **Refine `DW_HIL_CORE_002` (Potential Engine - P2 Full):** Update Π calculation to use P, C, and the new `synergy_score.parquet` (S_A→B terms). Zero-pad other conceptual domains (Social, Aptitude).
        *   `DW_HIL_SCHED_001`: Generalize `pid_scheduler.py` into a Multi-Domain PID Scheduler driven by Π. Define how PID error signals translate to actionable plan adjustments for each domain (e.g., adjust time allocation for learning blocks, modify intensity parameters for running/strength).
        *   `DW_HIL_SCHED_002`: Develop `daily_hpe_planner.py` to orchestrate the PID Scheduler outputs and the learning block schedulers (`active_learning_block_scheduler.py`, `passive_learning_block_scheduler.py`) into a unified daily plan (e.g., `daily_plan.json` or Markdown output).
        *   `DW_HIL_INTEGRATION_001` (C(t) into Π): Ensure cognitive metrics (`C(t)`) from LPP (`reading_stats.parquet`) and Flashcore (`flashcore_analytics.parquet`) are robustly integrated into the Π calculation.

    2.  **Knowledge Systems - Full MVP Functionality:**
        *   **Flashcore:**
            *   `(Ref: flashcore_plan.json)`
            *   `DW_FC_SCHED_001`: Implement core FSRS algorithm in `flashcore.scheduler`.
            *   `DW_FC_REVIEW_001`: Implement `ReviewSessionManager` core logic.
            *   `DW_FC_REVIEW_002` (CLI part): Develop functional CLI-based review experience.
            *   `DW_FC_EXPORT_001`: Implement Anki Exporter (`tm-fc export anki`).
            *   `DW_FC_CLI_002` (`add`), `DW_FC_CLI_003` (`vet`), `DW_FC_CLI_004` (`review` wrapper), `DW_FC_CLI_005` (`export` wrapper).
            *   `DW_FC_AUTO_001` (Pre-commit hook for `vet`), `DW_FC_AUTO_002` (Nightly CI for ingest/export).
        *   **Literature Pipeline:**
            *   `(Ref: literature_pipeline_plan.json)`
            *   `DW_LIT_AUTOMATION_001_BATCH_FETCH`: Deploy nightly CI for arXiv batch fetching and DocInsight result processing, committing artifacts.
            *   `DW_LIT_SEARCH_002_CLI`: Implement `lit-search` CLI.
            *   `DW_LIT_READERAPP_001_BACKEND` & `DW_LIT_READERAPP_002_FRONTEND` (MVP): Implement the Instrumented Reader web app (FastAPI backend with WebSocket telemetry to SQLite, basic PDF.js frontend for event capture).
            *   `DW_LIT_READERAPP_003_CLI_SESSION`: Refine `reading_session.py` CLI to log subjective metrics via ReaderApp API.
            *   `DW_LIT_TELEMETRYVIZ_001_CLI`: Develop `plot_reading_metrics.py` for telemetry visualization and flashcard candidate extraction.

    3.  **Software Dev Metrics (DevDailyReflect) - Enhanced Insights:**
        *   `(Ref: dev_daily_reflect_plan.json)`
        *   `DW_DDR_COACH_001` (Insight Engine MVP): Implement `coach/rules_engine.py` for identifying Risky Commits and TODOs/FIXMEs from enriched commit data.
        *   `DW_DDR_REPORT_001` (Overhaul Markdown Report): Integrate insights from `rules_engine.py`.

    4.  **Task Management & Curriculum Integration:**
        *   `(Ref: hil_kcv_plan.json, task_management/ docs)`
        *   `DW_HIL_TASKMGMT_001`: Develop `curriculum_parser.py` and `task_generator.py` (MVP for RNA Modeling CSM) to populate `tasks.json` with HPE metadata.
        *   Refine `active_learning_block_scheduler.py` and `passive_learning_block_scheduler.py` for robust operation with enriched `tasks.json`.

    5.  **Formal Methods - Initial Domain Proofs (P1 Roadmap Alignment):**
        *   `(Ref: formal_methods_plan.json)`
        *   `DW_FM_005`: Formalize Running domain ODE models (e.g., HR recovery, VO2 kinetics) in Lean.
        *   `DW_FM_006`: Formalize Biological domain ODE models (e.g., Logistic growth, Budworm model) from Math Bio curriculum in Lean.

    6.  **Infrastructure & Automation (IA) Layer - Maturation:**
        *   `(Ref: ia_layer_plan.json)`
        *   `DW_INFRA_ERROR_RESILIENCE_001`: Implement project-wide error handling, retry mechanisms (e.g., Python `tenacity` library for API calls), and alerting strategy for critical CI/automation failures.
        *   `DW_HIL_INFRA_001`: Set up CI/CD for HIL components (Synergy, Potential, Schedulers), including data validation.

*   **Phase 2 Risk-Gate Checklist (Exit Criteria):**
    *   ✅ HIL Synergy Engine (P1 Full) & Potential Engine (P2 Full) are operational, calculating `synergy_score.parquet` and `potential_snapshot.parquet`.
    *   ✅ Multi-Domain PID Scheduler MVP generates daily plans. `daily_hpe_planner.py` orchestrates learning block schedulers with PID outputs.
    *   ✅ Flashcore system allows YAML ingest, FSRS-based CLI review, and Anki export. Nightly builds run.
    *   ✅ Literature Pipeline features automated nightly fetching. Instrumented Reader MVP logs telemetry. `lit-search` CLI is functional.
    *   ✅ DevDailyReflect provides MVP insights (Risky Commits, TODOs) in daily reports.
    *   ✅ Curriculum Parser MVP ingests at least one CSM (e.g., RNA Week 1) into `tasks.json`.
    *   ✅ Lean proofs for initial Running & Biology ODEs are complete and CI-verified.
    *   ✅ Data contract validation and error resilience are enhanced across key pipelines.
    *   ✅ Unified Dashboard MVP is updated with P2 HIL outputs and richer domain KPIs.

---

### **Phase 3: "Predictive Augmentation & KCV Incubation" - Advanced Capabilities (Est. Duration: 6-9 Months)**

*   **Theme:** Introduce predictive modeling, enhance HIL with real-time biosignals (Focus Predictor), mature domain analytics, and lay concrete foundations for the Knowledge Creation & Validation (KCV) layer.
*   **Capability Wave Focus:** Advanced ML (Π weights, Focus Predictor), Initial Knowledge Graph, UI/UX Refinement.
*   **Key Objectives:**
    1.  Implement adaptive Π weight learning.
    2.  Develop and integrate Focus Predictor MVP.
    3.  Introduce predictive models for Running performance/injury.
    4.  Expand DevDailyReflect with richer data sources and Task Master integration.
    5.  Mature Flashcore/LPP with analytics dashboards and advanced features.
    6.  Begin KCV layer implementation (KG Core, Hypothesis Formalization, Sim Env MVP).
    7.  Formalize PID controller logic and explore AI-assisted proving in Lean 4.
*   **Major Development Tracks & Specific Tasks:** *(Selected highlights, drawing from all `deep_work_plan.json` files)*

    1.  **Holistic Integration Layer (HIL) - Advanced & Predictive:**
        *   `DW_HIL_CORE_003`: Implement `update_potential_weights.py` for Π recalibration (regression-based).
        *   **Focus Predictor MVP (`DW_HIL_FOCUS_001` to `_004`):** Assemble and validate DIY sensor hardware (EEG-B, EDA). Implement software stack (LSL ingestion, preprocessing, feature extraction). Train initial XGBoost + Kalman fusion model. Implement Focus Score API. *This is a major parallel R&D effort.*
        *   **Running Predictive Models (Ref: `running_performance_prediction_concepts.md`):**
            *   **`NEW_TASK: DW_RUN_PREDICT_PERFORMANCE_001`**: Implement Gradient Boosting Models (GBMs) for short-term run performance/recovery prediction.
            *   **`NEW_TASK: DW_RUN_PREDICT_INJURY_001`**: Implement Survival Analysis models for injury risk/timing.

    2.  **Knowledge Systems - Advanced Features & Analytics:**
        *   **Flashcore:** `DW_FC_EXPORT_002` (Markdown), `DW_FC_INT_001` (Task Master `[[fc]]` Hook), `DW_FC_ANALYTICS_001` (Core Analytics), `DW_FC_ANALYTICS_002` (Streamlit Dashboard MVP).
        *   **Literature Pipeline:** Refine Instrumented Reader UI/UX. `DW_LIT_AUTOMATION_002_CORPUS_MGMT` (ADR for DocInsight corpus management).
        *   **`NEW_TASK: DW_KNOWLEDGE_DASHBOARD_001`**: Design and implement an MVP "Holistic Knowledge Dashboard" (Streamlit) integrating insights from LPP, Flashcore, Instrumented Reading, and Math Bio progress.

    3.  **Software Dev Metrics (DevDailyReflect) - Full Capabilities:**
        *   `DW_DDR_DATASRC_001` (PR/Issue Data), `DW_DDR_DATASRC_002` (CI/CD Coverage Data).
        *   `DW_DDR_COACH_002` (Extend Insight Engine for PR/Issue/CI alerts).
        *   `DW_DDR_INTEGRATION_001` (Task Master integration for actionable items).
        *   `DW_DDR_ANALYSIS_001` (Historical Trend Analysis Dashboard).

    4.  **Knowledge Creation & Validation (KCV) - Foundational MVPs:**
        *   `DW_KCV_001`: Design & Implement Knowledge Graph Core (ontology, backend, initial population scripts).
        *   `DW_KCV_002`: Develop Hypothesis Formalization & Testability Assessment Module (MVP).
        *   `DW_KCV_003`: Implement Simulation Environment & Model Management (MVP: ODEs from Math Bio, Snakemake).
        *   `DW_KCV_007`: Design & Implement Ethical and Epistemic Safeguards Framework (KCV MVP).

    5.  **Formal Methods - Advanced Tooling & Core HIL Proofs:**
        *   `DW_FM_004` (Lean-Python Cross-Verification Framework).
        *   `DW_FM_007` / `DW_HIL_FM_001`: Formalize PID Controller algorithm and prove stability/boundedness.
        *   `DW_FM_012`: Formalize Synergy & Π Equations in Lean.
        *   `DW_FM_008` (AI Prover Integration - DeepSeek).

    6.  **IA Layer - Scalability & Orchestration Prep:**
        *   `DW_INFRA_LARGE_DATA_MGMT_001` (Implement chosen strategy for DVC/LFS).
        *   `DW_INFRA_ORCH_001` (Research advanced workflow orchestration - Airflow, Prefect for KCV).
        *   All documentation tasks (`DW_FC_DOCS_001`, `_002`, `DW_LIT_DOCS_001_SYSTEM_GUIDE`, `DW_DDR_DOCS_001`, `DW_HIL_DOCS_001`).

*   **Phase 3 Risk-Gate Checklist (Exit Criteria):**
    *   ✅ HIL Π recalibration is functional. Focus Predictor MVP provides a usable Focus Score via API, integrated into Π/Scheduler.
    *   ✅ Initial predictive models for Running performance/injury are operational.
    *   ✅ Holistic Knowledge Dashboard MVP is live. Flashcore/LPP analytics dashboards are available.
    *   ✅ DevDailyReflect integrates PR/Issue/CI data and generates actionable Task Master items.
    *   ✅ KCV KG Core is implemented. Hypothesis Formalization and Simulation Environment MVPs are functional.
    *   ✅ PID Scheduler logic, Synergy & Π equations are formally defined/proven in Lean. AI Prover integration initiated.
    *   ✅ Large data management solution (DVC/LFS) is in place. ADR for advanced orchestration complete.
    *   ✅ All P0-P3 components are comprehensively documented.

---

### **Phase 4 & Beyond: "Grand Challenges & System Self-Evolution" (Ongoing, Long-Term)**

*   **Theme:** Mature and expand KCV layer capabilities, transition HIL scheduler to Reinforcement Learning, achieve comprehensive formal verification, and apply the Cultivation system to its most ambitious long-term research goals.
*   **Capability Wave Focus:** Reinforcement Learning, Generative AI, Advanced Knowledge Engineering, Deep Formal Proofs, System Autonomy.
*   **Key Development Areas (Examples, drawing from P4+ `roadmap_vSigma.md` and EPIC tasks):**
    *   **HIL - RL Scheduler:** Implement RL-based scheduler (PPO or similar) to replace/augment PID, aiming to optimize Π or related reward functions directly.
    *   **KCV - Full Scale:**
        *   Mature Simulation Environment (diverse models: ABM, PDE, etc.).
        *   `DW_KCV_004`: Analogical Reasoning Assistant.
        *   `DW_KCV_005`: Conceptual Knowledge Versioning within KG.
        *   `DW_KCV_006`: External Impact Tracking (Patent Office/Journal).
        *   Integrate external specialized project outputs (from `RNA_PREDICT`, `PrimordialEncounters`, `Simplest_ARC_AGI`) as KCV "Laboratory" components or "Think Tank" knowledge sources.
    *   **Formal Methods - Advanced & EPIC Research:**
        *   `DW_FM_009` (Benchmark AI Provers), `DW_FM_010` (LoRA Fine-Tuning).
        *   `DW_FM_011` (Systematic Audit & Formalization of `math_stack.md`).
        *   `DW_FM_013` (Automate `math_stack.md` documentation).
        *   `DW_FM_014` (Proof Maintenance Strategy).
        *   EPIC Research (`DW_FM_015` to `_017`): Potential Theorems, Formal Logic for Boundaries, Unified Theory of Potential.
    *   **Flashcore EPICs (Full Implementation):** `DW_FC_EPIC_SAT` (Self-Assessment Tool), `DW_FC_EPIC_KD` (Knowledge Dimensions), `DW_FC_EPIC_ANKI_SYNC`.
    *   **System Self-Improvement & Autonomy:** Design mechanisms for Cultivation to analyze its own performance (e.g., using DevDailyReflect on its codebase, KCV to critique its models) and suggest or automate improvements.

*   **Reasoning for P4+:** This phase represents the project operating at its full R&D potential, directly tackling its most ambitious scientific and transhumanist goals.

---

## 4. Cross-Cutting Concerns & Ongoing Activities (All Phases)

*   **Documentation (`DW_..._DOCS_...`):** Continuous creation, refinement, and synchronization of user guides, developer documentation, API specifications, and design documents. The MkDocs site (`DW_INFRA_DOCS_TOOLING_001`) is the central hub.
*   **Testing (`DW_..._TEST_...`, `DW_INFRA_TESTING_001`):** Rigorous unit, integration, and end-to-end (E2E) testing for all components. Data contract validation. Test coverage monitoring and improvement.
*   **Security (`DW_INFRA_SECURITY_001`):** Ongoing vigilance, secure coding practices, dependency scanning, secrets management, and regular security audits.
*   **Idempotency & Error Resilience (`DW_INFRA_IDEMPOTENCY_001`, `DW_INFRA_ERROR_RESILIENCE_001`):** Ensuring all automated processes are robust, fault-tolerant, and safely re-runnable.
*   **User Experience (UX) & Usability:** Even for an N=1 system, ensuring CLIs are intuitive, dashboards are informative, and workflows are efficient is crucial for sustained engagement and system effectiveness.
*   **Task Management & Project Tracking (`DW_HIL_META_001`):** All development work (including these roadmap tasks) should be tracked in Task Master (or `tasks.json`), linked to relevant design documents and code. The `DW_HIL_META_001` process is used for periodic roadmap review, task breakdown, and re-prioritization.
*   **External Dependency Management:** Active monitoring of external services (DocInsight, HabitDash, GitHub API, LLM APIs) and libraries. Maintain fallbacks or abstraction layers where feasible. Regularly update pinned dependencies.

## 5. Integration with External Repositories (Overall Strategy)

The `external_systems_analysis.md` outlines a strategy of leveraging several powerful, likely author-developed, external repositories:

*   **`DocInsight` (Backend RAG Service):**
    *   **Critical Dependency:** For Literature Pipeline functionality (semantic search, summarization, novelty scoring).
    *   **Integration:** Assumed to be deployed as a service (local Docker or remote). Cultivation's `docinsight_client.py` interacts with its API.
    *   **Phasing:** P0/P1 for core client integration. `DW_LIT_AUTOMATION_002_CORPUS_MGMT` (P3) addresses its corpus management.
*   **`RNA_PREDICT` (RNA 3D Structure Prediction):**
    *   **Role:** Advanced modeling tool for the RNA Biology curriculum (Pillar 1 projects in `SKILL_MAP_CSM_pillar1.md`) and a key component for KCV "Laboratory" research.
    *   **Integration:** Outputs (predicted structures, metrics) can be ETL'd into Cultivation's data stores. Training/inference jobs can be managed by Cultivation's (future advanced) schedulers.
    *   **Phasing:** Initial use in P3/P4 learning projects; deeper KCV integration P4+.
*   **`pytest-fixer` (AI-Driven Test Fixing):**
    *   **Role:** Enhances Software Engineering domain capabilities and can be used on the Cultivation codebase itself.
    *   **Integration:** Can be run locally from P1. `DW_PYFIX_ETL` (P3) plans to ingest its logs for `ETL_Software`. CI integration (`DW_PYFIX_CI_INTEG`) is a research task.
    *   **Licensing:** Needs resolution as per `DW_PYFIX_LICENSE`.
*   **`Simplest_ARC_AGI` (Neural Circuit Extraction for ARC):**
    *   **Role:** Foundation for the "Abstract Reasoning (ARC)" domain and KCV research into model interpretability/composition.
    *   **Integration:** `DW_ARC_ETL` (P3) for ingesting ARC model performance metrics. Advanced circuit extraction/composition becomes KCV research projects (`DW-ARC-KCV-...` in P4+).
*   **`PrimordialEncounters` (Astrophysics Simulation):**
    *   **Role:** Engine for the "Astrophysics" domain and a KCV "Laboratory" component for N-body simulations.
    *   **Integration:** P1 roadmap mentions "Astro N-body sandbox." This implies early operationalization. `ETL_Astro` (P3+) will bring its simulation outputs into Cultivation.

**Strategy:** These external repositories are treated as specialized microservices or libraries. Cultivation focuses on the integration layer, data ETL, and holistic analysis, leveraging their domain-specific power.

## 6. Conclusion and Next Steps (Post-Approval of this Plan)

This Integrated Phased Development Plan v1.0 provides a detailed, structured, and ambitious yet pragmatically sequenced roadmap for realizing the Cultivation project's vision. It prioritizes establishing robust engineering foundations before layering on complex functionalities and advanced research capabilities.

**Immediate Next Steps (Upon Approval):**

1.  **Formalize P0 Tasks in Task Master:** Break down the P0 deliverables from this plan into specific, actionable tasks in `tasks.json` with appropriate HPE metadata.
2.  **Initiate P0 Development:** Begin work on the IA Layer core components and Data Contract definitions.
3.  **Establish P0 Risk-Gate Review Process:** Define how and when the P0 Risk-Gate Checklist will be formally reviewed and signed off.
4.  **Address Critical Documentation Gaps:** Start populating `functional_requirements.md` and `non_functional_requirements.md`.
5.  **Set Up Initial Project Boards/Tracking:** Configure Task Master (or chosen PM tool) to reflect this roadmap's phases and priorities.

This plan, while comprehensive, is a living document. Regular review, adaptation, and re-prioritization via the `DW_HIL_META_001` process will be essential for navigating the complexities of this project and ensuring its long-term success in achieving its transformative goals.