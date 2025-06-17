## **Onboarding Document: Holistic Performance Enhancement (Cultivation) Project**

**Version:** 2.1 (Post-Validation Iteration)
**Date:** 2025-06-17

**Objective:** This document provides a definitive, comprehensive analysis of the "Holistic-Performance-Enhancement" (HPE) or "Cultivation" repository. It is designed to give a new project member a deep understanding of the project's vision, architecture, current state, and development workflows, enabling them to quickly become a productive contributor to deep work tasks.

### **1. High-Level Vision & Core Philosophy**

Welcome to the **Holistic Performance Enhancement (HPE)** project, internally codenamed **"Cultivation"**. This is not a standard software project but a long-term, N=1 research and development initiative to build a comprehensive, data-driven "operating system for the self."

**1.1. The Mission**

The project's mission is to systematically enhance human performance by integrating and analyzing data across multiple, interconnected life domains. The ultimate goals that guide the project's architecture include radical life extension, accumulating intellectual power, and deeply understanding natural laws, as detailed in `The_Dao_of_Cultivation_A_Founding_Vision.md`.

**1.2. Core Philosophy**

The project is built on a set of core principles that every contributor must understand:

*   **Radical Quantification:** A foundational belief that **"if a benefit isn’t measured, it effectively isn’t real."** This drives the project's obsession with capturing metrics for everything.
*   **Synergy as a First-Class Citizen:** The system is explicitly designed to model and leverage the synergistic effects between domains (e.g., `S(Running → Software)`).
*   **Systematic, Iterative Improvement:** The project itself is a "cultivation" process, following a phased roadmap (`roadmap_Cultivation_Integrated_v1.0.md`) with explicit goals and risk-gates.
*   **Formal Rigor:** A long-term aspiration is to use formal methods (Lean 4) to mathematically prove the correctness of critical system algorithms.
*   **Verification over Assumption:** All changes—code, tests, automation, or documentation—must be locally verified. An exit code of 0 is not sufficient proof of correctness. When in doubt, ask for clarification or write a test to prove your assumption.

**1.3. The Language of Cultivation**

To navigate this repository, you must understand these central concepts:

*   **Domains:** The distinct areas of performance being measured (Running, Strength, Knowledge, Wellness, etc.).
*   **Potential (Π):** A core composite metric representing the theoretical maximum performance capacity of the entire system.
*   **Holistic Integration Layer (HIL):** The central "brain" of the system (Synergy Engine, Potential Engine) that synthesizes data to drive scheduling.
*   **Knowledge Creation & Validation (KCV) Layer:** A visionary component for generating and validating new hypotheses.

---

### **2. System Architecture & Data Flow**

The project follows a multi-layered data processing and feedback loop.

```mermaid
graph TD
    subgraph A [Data Sources]
        A1[Wearables & GPX/FIT Files]
        A2[Git Repository History]
        A3[Academic Papers & Notes]
        A4[Strength Logs (.yaml/.md)]
        A5[Flashcard Reviews (.yaml)]
        A6[HabitDash API - Wellness Metrics]
    end

    subgraph B [ETL & Processing Layer: `cultivation/scripts/`]
        B1[Running Pipeline]
        B2[DevDailyReflect Pipeline]
        B3[Literature Pipeline & DocInsight]
        B4[Strength Pipeline]
        B5[Flashcore Pipeline]
        B6[Wellness Sync Pipeline]
    end

    subgraph C [Data Stores: `cultivation/data/`]
        C1[running_*.parquet, weekly_metrics.parquet]
        C2[dev_metrics_*.csv, pr_summaries*.md]
        C3[reading_stats.parquet, literature/db.sqlite]
        C4[strength_*.parquet]
        C5[flash.db]
        C6[daily_wellness.parquet]
    end

    subgraph D [Holistic Integration Layer (HIL)]
        D1[Synergy Engine]
        D2[Global Potential (Π) Engine]
        D3[Adaptive Scheduler (PID/RL)]
    end

    subgraph E [User-Facing Outputs & Actuators]
        E1[Task Master: `tasks.json`]
        E2[Dashboards & Reports]
        E3[CLI Tools]
        E4[Anki Decks & Markdown Exports]
    end

    A1 --> B1 --> C1; A2 --> B2 --> C2; A3 --> B3 --> C3;
    A4 --> B4 --> C4; A5 --> B5 --> C5; A6 --> B6 --> C6;
    C1 & C2 & C3 & C4 & C5 & C6 --> D1 --> D2 --> D3 --> E1 & E2;
    B1 & B2 & B3 & B4 & B5 --> E2 & E3; B5 --> E4;
```

---

### **3. Key Domains & Sub-Systems: Status & Analysis**

The project is organized into several domains, each with its own systems and varying levels of maturity.

#### **3.1. Running Performance (Physical Domain)**
*   **Maturity:** **High.** The most mature domain.
*   **Key Scripts:** The main entry point is `process_all_runs.py`.
*   **How to Use:** Drop `.gpx` or `.fit` files into `cultivation/data/raw/` and run `task run:process-runs`.

#### **3.2. Wellness & Recovery (Physiological Domain)**
*   **Maturity:** **High.** Fully automated and integrated.
*   **Key Scripts:** `utilities/sync_habitdash.py` (for data sync), `running/fatigue_watch.py` (for alerts).
*   **How to Use:** Runs automatically via GitHub Actions. Manual sync: `task run:sync-wellness`.

#### **3.3. Knowledge Acquisition (Cognitive Domain)**
*   **Maturity:** **Medium.** Robust backend; user-facing components in development.
*   **Sub-Systems:** Literature Pipeline (DocInsight), Flashcore (FSRS flashcards), Instrumented Reading App, and the **conceptual** Mentat-OS.

#### **3.4. Software Engineering (DevDailyReflect)**
*   **Maturity:** **High.** Functional and automated.
*   **Key Scripts:** Pipeline in `scripts/software/dev_daily_reflect/`.
*   **How to Use:** Runs automatically via CI. Manual run: `task run:dev-reflect`.

#### **3.5. Strength Training**
*   **Maturity:** **Low-Medium.** Data schemas and ingestion scripts exist but analytics are underdeveloped. A prime area for contribution.
*   **Key Scripts:** `strength/ingest_yaml_log.py`, `strength/log_strength_session.py`.

#### **3.6. Abstract Reasoning & AI (`jarc_reactor`)**
*   **Maturity:** **High.** A mature, self-contained MLOps project.
*   **Architecture:** PyTorch Lightning, Hydra for config, Optuna for HPO.
*   **How to Use:** `task arc:run-first-light` for a quick test. See `README.md` in its directory for more.

---

### **4. Development Workflow & Tooling**

A new contributor must understand and adhere to the following core practices.

**4.1. The Task Runner (`Taskfile.yml`): Your Primary Interface**
The project uses **Go Task** as the universal command runner. **All development tasks, including testing and linting, SHOULD be run via the `Taskfile.yml`.**

*   **Key Commands:**
    *   `task setup`: Sets up the Python virtual environment and installs dependencies.
    *   `task test`: Runs the entire `pytest` suite.
    *   `task lint`: Runs all code and documentation linters.
    *   `task docs:serve`: Serves the documentation site locally.
*   **To see all available commands, run `task --list-all`**.

**4.2. Automation & CI/CD (`.github/workflows/`)**
The project is heavily automated with GitHub Actions. Review the YAML files in this directory to understand how quality is maintained and data is kept fresh.

**4.3. The Principle of Verification: Trust, but Verify**
This project values correctness and robustness. Adherence to the following verification principles is mandatory.

*   **Read the Logs:** An exit code of 0 from a command is **not sufficient** to assume success. You must always inspect the full log output to check for underlying errors, warnings, or unexpected behavior.
*   **Know What Success Looks Like:** A successful `task test` run, for example, will show a summary from `pytest` at the end indicating the number of tests passed. It will look something like this:
    ```
    ================== XX passed in YYs ==================
    ```
    Any output that does not contain this summary, or contains tracebacks and error messages, indicates a failure that must be investigated.
*   **Verify Your Automations:** If you add or change a command in `Taskfile.yml` (e.g., you add `run:my-new-script`), you must run it locally (`task run:my-new-script`) to confirm it executes correctly before committing.

**4.4. Troubleshooting Common Errors**
If you encounter issues, especially after pulling new changes, follow these steps first.

*   **Problem:** `ModuleNotFoundError`, `ImportError`, or test failures due to dependency conflicts.
    *   **Solution:** Your local environment may be out of sync. Activate the virtual environment (`source .venv/bin/activate`) and re-run the setup task to install the latest pinned dependencies: `task setup`.
*   **Problem:** A `task` command fails with unexpected errors (e.g., `task test` reports a `pytest` error).
    *   **Solution:** First, carefully read the full error log in your terminal to understand the specific error message. Then, for a systematic approach to debugging, consult the project's guide:
    *   **Canonical Guide:** **[`cultivation/docs/7_user_guides_and_sops/comprehensive_debugging_workflow.md`](./docs/7_user_guides_and_sops/comprehensive_debugging_workflow.md)**

---

### **5. Onboarding Guide for a New Contributor**

Follow these steps to get up and running:

1.  **Understand the Vision:** Read the key documents listed in Section 1.
2.  **Set Up Your Environment:**
    *   Install **Go Task** (see `Taskfile.yml` for instructions).
    *   Clone the repository.
    *   From the project root, run `task setup`.
    *   Activate the environment: `source .venv/bin/activate`.
    *   Copy `.env.template` to `.env` and fill in any API keys.
3.  **Learn the Workflow:**
    *   Familiarize yourself with the commands in `Taskfile.yml` (`task --list-all`).
    *   Run the core quality checks (`task lint`, `task test`) and **verify their output logs**.
    *   Serve the documentation locally to explore it: `task docs:serve`.
4.  **Explore a Mature Domain:**
    *   The **Running** domain is the best starting point.
    *   Trace its workflow: examine raw data, run the pipeline (`task run:process-runs`), inspect the processed Parquet/CSV files and the output reports/figures. Read the corresponding scripts to understand the logic.
5.  **Find a First Task & Contribute:**
    *   A good starting point is to contribute to a P0/P1 task from the roadmap (`roadmap_Cultivation_Integrated_v1.0.md`).
    *   **Example Areas:**
        *   Enhancing analytics for the **Strength Training domain**.
        *   Implementing a new CLI command for the **Flashcore system**.
        *   Adding test cases to an existing data pipeline.
    *   Create a feature branch, make yourchanges, add tests, and verify locally with `task lint` and `task test` before submitting a Pull Request.

    **Task Completion Verification Checklist (especially for AI Agents):**
    Before considering a development task complete and ready for PR, ensure ALL of the following steps are VERIFIED (do not rely on exit codes alone; inspect logs thoroughly):

        1.  **Execution of New/Modified Components & Log Review:**
            *   Any newly created or modified executable components (e.g., scripts, `Taskfile.yml` commands) have been run locally.
            *   Their output logs MUST be meticulously reviewed for correctness, expected behavior, and the absence of errors or warnings.
        2.  **Automated Tests Passed & Log Review:**
            *   All relevant automated tests, particularly any newly added ones, pass when executed via the project's standard testing command (e.g., `task test`).
            *   Output logs MUST be meticulously reviewed.
        3.  **Code Quality Checks Passed & Log Review:**
            *   All code and documentation conform to project standards, as verified by linters and other quality checks (e.g., `task lint`).
            *   Output logs MUST be meticulously reviewed.
        4.  **Core Task Objective Achieved & Verified:**
            *   The primary functional goal of the task has been confirmed. This may involve manual inspection, reviewing generated files, or other verification methods if not fully covered by automated tests.

---

### **6. Overall Assessment & Strategic Recommendations**

*   **Strengths:** Exceptional vision & documentation; high degree of automation; systematic and data-driven philosophy; strong technical foundations in several domains.
*   **Challenges & Risks:** The project's scope is vast for a small team. There is a significant gap between the visionary documentation and the current implementation, particularly for the HIL and KCV layers.
*   **Strategic Direction:** The project is building out its core domain-specific systems. The highest-value next steps involve implementing the HIL to connect the mature data pipelines, solidifying data contracts, and bringing less mature domains (like Strength) to parity with mature ones (like Running).

This project is a marathon, not a sprint. Welcome to the team, and happy cultivating.