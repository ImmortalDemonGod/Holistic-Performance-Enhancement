The "Daily Dev Review" (internally named `DevDailyReflect`) is an automated system designed to analyze software development activity within the `Holistic-Performance-Enhancement` repository. It processes Git commit history to generate daily reports summarizing productivity and code quality metrics. This system aims to provide developers with insights into their recent work, flag potential risks, and inform their planning for the next day.

Here's a detailed breakdown of how it works and what it does, in the context of the repository:

**I. Core Goal & Philosophy**

*   **Automated Reflection:** The system automates the collection and processing of development metrics, reducing manual effort for daily review.
*   **Data-Driven Insights:** It provides quantitative data on commit activity (LOC, frequency) and code quality (Cyclomatic Complexity, Maintainability Index, Ruff linting errors).
*   **Actionable Feedback (Implicit):** While the current version primarily reports metrics, the structure (seen in `cultivation/scripts/software/dev_daily_reflect/README.md`'s "Program: DevDailyReflect" outline) is designed to evolve towards providing more direct suggestions (e.g., identifying risky commits, TODOs).
*   **Contribution to Holistic Performance:** This system focuses on the "Software Engineering" subdomain of the broader "Cultivation" project. The metrics generated are intended to feed into the `ETL_S` (Software ETL) and subsequently the `Synergy Engine` and `Potential Engine (Π)`, contributing to the "Cognitive" or "Technical" aspects of holistic performance.

**II. Automated Workflow Orchestration**

The entire process is orchestrated by a GitHub Action workflow:

1.  **Trigger:**
    *   Defined in `.github/workflows/daily_dev_review.yml`.
    *   Runs daily at 7:00 UTC (via `cron: '0 7 * * *'`).
    *   Can also be triggered on pushes to any branch or manually (`workflow_dispatch`).
    *   It skips runs triggered by the `github-actions[bot]` to avoid loops.

2.  **Setup:**
    *   Checks out the repository with full history (`fetch-depth: 0`) to allow `git log` to access all necessary commits.
    *   Sets up Python 3.11.
    *   Installs dependencies from `requirements.txt` (which now includes `radon`, `ruff`, `GitPython`, `PyYAML`).

3.  **Testing (Pre-pipeline):**
    *   Runs `pytest` to ensure the general health of the Python environment and scripts.

4.  **Pipeline Execution:**
    *   Executes `cultivation/scripts/software/dev_daily_reflect/test_dev_daily_reflect.sh`. This shell script orchestrates the core logic:
        1.  `ingest_git.py`: Fetches and enriches commit data.
        2.  `aggregate_daily.py`: Aggregates metrics per author.
        3.  `report_md.py`: Generates the Markdown report.

5.  **Report Handling:**
    *   **Commit & Push:** The generated Markdown report (e.g., `dev_report_YYYY-MM-DD.md`) is committed back to the repository in the `cultivation/outputs/software/dev_daily_reflect/reports/` directory by the `github-actions[bot]`. The commit message includes `[CI skip]` to prevent triggering other CI workflows.
    *   **Artifact Upload:** The same report is uploaded as a build artifact for easy access from the GitHub Actions run page.

**III. Core Python Scripts & Data Flow**

The system uses a series of Python scripts located in `cultivation/scripts/software/dev_daily_reflect/`:

1.  **Configuration Loading (`config_loader.py`):**
    *   **Input:** `cultivation/scripts/software/dev_daily_reflect/config/daily_review.yaml`.
    *   **Functionality:** Loads configuration settings, providing defaults and allowing overrides via environment variables. Key settings include `repository_path` (points to the root of the `Holistic-Performance-Enhancement` repo), `lookback_days` (how many days of commits to analyze, default 1), `rollup_dir`, and `report_output_dir`.
    *   **Output:** A configuration dictionary used by other scripts.

2.  **Git Commit Ingestion (`ingest_git.py`):**
    *   **Input:** Configuration (especially `repository_path` and `lookback_days`).
    *   **Functionality:**
        *   Determines the time window for fetching commits (e.g., last 24 hours if `lookback_days: 1`).
        *   Uses `subprocess` to run `git log` with `--numstat` to get commit details (SHA, author, timestamp, message, lines added/deleted per file).
        *   Parses the `git log` output.
        *   **Output (1):** Saves this raw commit data as `cultivation/outputs/software/dev_daily_reflect/raw/git_commits_YYYY-MM-DD.json`. (Example: `git_commits_2025-05-17.json`)
        *   **Enrichment:** Calls `analyze_commits_code_quality` from `metrics/commit_processor.py`.
        *   **Output (2):** Saves the enriched commit data as `cultivation/outputs/software/dev_daily_reflect/raw/git_commits_enriched_YYYY-MM-DD.json`. (Example: `git_commits_enriched_2025-05-17.json`)

3.  **Commit Metrics Calculation (`metrics/commit_processor.py`):**
    *   **Input:** Repository path and the list of raw commit dictionaries from `ingest_git.py`.
    *   **Functionality:** For each commit:
        *   Identifies all Python (`.py`) files changed in that commit.
        *   For each changed Python file:
            *   Retrieves the actual content of the file *at that specific commit's version* using `repo.git.show(f'{sha}:{fname}')`.
            *   Calculates Cyclomatic Complexity (CC) using `radon.complexity.cc_visit`.
            *   Calculates Maintainability Index (MI) using `radon.metrics.mi_visit`.
            *   Runs `ruff` (a Python linter) on the file content (saved to a temporary file) and counts the number of reported issues.
        *   Aggregates these metrics per commit: `py_files_changed_count`, `total_cc` (sum of CC for all changed .py files), `avg_mi` (average MI of changed .py files), and `ruff_errors` (sum of Ruff issues).
    *   **Output:** The list of commit dictionaries, now augmented with these code quality metrics.

4.  **Daily Aggregation (`aggregate_daily.py`):**
    *   **Input:** The latest enriched commit JSON file (e.g., `git_commits_enriched_YYYY-MM-DD.json`).
    *   **Functionality:**
        *   Reads the enriched JSON data into a Pandas DataFrame.
        *   Groups the data by `author`.
        *   Aggregates metrics per author:
            *   `commits`: Count of commits.
            *   `loc_add`, `loc_del`: Sum of lines added/deleted.
            *   `loc_net`: Calculated as `loc_add - loc_del`.
            *   `py_files_changed_count`: Sum.
            *   `total_cc`: Sum.
            *   `avg_mi`: Mean.
            *   `ruff_errors`: Sum.
    *   **Output:** Saves the aggregated data to `cultivation/outputs/software/dev_daily_reflect/rollup/dev_metrics_YYYY-MM-DD.csv`. (Example: `dev_metrics_2025-05-17.csv`)

5.  **Markdown Report Generation (`report_md.py`):**
    *   **Input:** The latest aggregated metrics CSV file (e.g., `dev_metrics_YYYY-MM-DD.csv`) and, if available, the corresponding enriched commit JSON file.
    *   **Functionality:**
        *   Reads the rollup CSV.
        *   Constructs a Markdown report:
            *   A header with the date.
            *   Overall summary statistics (total commits, total net LOC, total Python files changed, total CC, average MI across authors, total Ruff errors).
            *   A table summarizing these metrics for each author.
            *   If the `git_commits_enriched_YYYY-MM-DD.json` file is found, it adds a "Per-Commit Metrics" section with a detailed table showing SHA (shortened), Author, Added, Deleted, Py Files changed, CC, Avg MI, and Ruff errors for each commit in the period.
    *   **Output:** Saves the report as `cultivation/outputs/software/dev_daily_reflect/reports/dev_report_YYYY-MM-DD.md`. (Examples: `dev_report_2025-05-16.md`, `dev_report_2025-05-17.md`)

6.  **Utilities (`utils.py`):**
    *   Currently contains `get_repo_root()`, which robustly determines the project's root directory. This is crucial for scripts to locate configuration files and output directories correctly, regardless of where they are invoked from.

**IV. Data Produced**

*   **Raw Commits:** `cultivation/outputs/software/dev_daily_reflect/raw/git_commits_YYYY-MM-DD.json`
    *   Basic git log information (SHA, author, timestamp, message, LOC added/deleted).
*   **Enriched Commits:** `cultivation/outputs/software/dev_daily_reflect/raw/git_commits_enriched_YYYY-MM-DD.json`
    *   Raw commit data plus per-commit code quality metrics (CC, MI, Ruff errors, Python files changed).
*   **Daily Rollup:** `cultivation/outputs/software/dev_daily_reflect/rollup/dev_metrics_YYYY-MM-DD.csv`
    *   Metrics aggregated per author for the day.
*   **Markdown Report:** `cultivation/outputs/software/dev_daily_reflect/reports/dev_report_YYYY-MM-DD.md`
    *   Human-readable summary of the daily development activity and quality metrics.

**V. Context and Integration within the "Cultivation" Project**

The "Daily Dev Review" system serves as a concrete implementation of the software metrics arm of the "Cultivation" project:

1.  **ETL for Software Domain (ETL_S):**
    *   The `ingest_git.py` and `metrics/commit_processor.py` scripts act as the Extract, Transform, and Load (ETL) pipeline for software development data. This aligns with the `ETL_S` component in `cultivation/docs/3_design/design_overview.md`.
    *   The output `dev_metrics_YYYY-MM-DD.csv` (or potentially the more granular `git_commits_enriched_...json`) becomes the structured data source (`softwareDB[(software.parquet)]` in the design overview, though currently CSV/JSON).

2.  **Input to Synergy and Potential Engines:**
    *   As per `cultivation/docs/1_background/synergy_concept.md` and `cultivation/docs/3_design/design_overview.md`, metrics from software engineering (like commit frequency, code quality from CC/MI/Ruff) are intended to feed the `calculate_synergy.py` and `potential_engine.py`. The daily rollup CSV provides these quantitative inputs.

3.  **Alignment with Road-map (v Σ):**
    *   This system directly addresses Phase P0's goal: "② Software Repo --> git log JSON --> ETL_S" from the C4 diagram in `cultivation/docs/3_design/design_overview.md`.
    *   It contributes to the "Software-Engineering Metrics" math stack defined in `cultivation/docs/2_requirements/math_stack.md` (discrete math, graph theory for CC, statistical process control for trends).

4.  **Operational Playbook:**
    *   The daily generated report becomes a key artifact for the "Developer" persona's "Daily Loop" as envisioned in `cultivation/docs/4_analysis/operational_playbook.md`. It provides the data for reflection and planning.

5.  **Analysis Layer:**
    *   The reports and raw data stored in `cultivation/outputs/software/dev_daily_reflect/` fit the purpose of `cultivation/docs/4_analysis/` as a "results vault." These outputs are concrete analytical products. The `commit_metrics_prototyping.py` notebook was an early exploration, and this system is its operationalization.

6.  **Knowledge Management & Learning (`flashcards*.md`, `knowledge_creation_*.md`):**
    *   While not directly implemented yet, the "Program: DevDailyReflect" outline (in prompt, referenced from `cultivation/scripts/software/dev_daily_reflect/README.md`) suggests an "Insight & Guidance Engine Layer (`coach/rules_engine.py`)". This engine could identify patterns (e.g., consistently low MI in certain modules, frequent TODOs) that become learning points or flashcards for the developer, aligning with the knowledge retention and creation goals.

7.  **Task Master Integration (`task_master_integration.md`):**
    *   Future enhancements to the "Insight & Guidance Engine" could automatically generate tasks in Task Master (e.g., "Review commit X due to high CC," "Address FIXME in file Y").

In summary, the Daily Dev Review system is a foundational data collection and reporting pipeline for the software engineering aspect of the "Cultivation" project. It transforms raw Git history into structured metrics and human-readable reports, enabling daily reflection, tracking code quality trends, and providing essential data for higher-level synergy and potential calculations within the holistic performance enhancement framework.
====
You can run the "Daily Dev Review" (DevDailyReflect) pipeline manually by executing the shell script that the GitHub Action uses, or by running the individual Python scripts in sequence.

Here's how:

**Prerequisites:**

1.  **Git:** Ensure Git is installed and your repository (`Holistic-Performance-Enhancement`) has some commit history.
2.  **Python:** Python 3.11 (as specified in the CI workflow) should be installed.
3.  **Dependencies:** You need to install the Python packages listed in `requirements.txt`. It's highly recommended to do this in a virtual environment.
    *   `python3 -m venv .venv` (if you haven't already created a venv in your project root)
    *   `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)
    *   `pip install -r requirements.txt`
4.  **Ruff:** Ensure `ruff` is installed and accessible in your PATH if you want Ruff metrics (it's listed as a dependency, so `pip install -r requirements.txt` should cover it).
5.  **Repository Root:** All commands should generally be run from the root of your `Holistic-Performance-Enhancement` repository.

---

**Method 1: Using the Orchestration Shell Script (Recommended for ease)**

The `cultivation/scripts/software/dev_daily_reflect/test_dev_daily_reflect.sh` script is designed to run the entire pipeline.

1.  **Navigate to the Repository Root:**
    ```bash
    cd /path/to/your/Holistic-Performance-Enhancement
    ```

2.  **Ensure the script is executable (if needed):**
    ```bash
    chmod +x cultivation/scripts/software/dev_daily_reflect/test_dev_daily_reflect.sh
    ```

3.  **Run the script:**
    *   Make sure your virtual environment is activated.
    *   The script sets `PYTHONPATH=.` internally for its Python calls, assuming it's run from the project root.
    ```bash
    bash cultivation/scripts/software/dev_daily_reflect/test_dev_daily_reflect.sh
    ```
    This will:
    *   Run `ingest_git.py` to fetch raw and enriched commit data.
    *   Run `aggregate_daily.py` to create the daily metrics rollup CSV.
    *   Run `report_md.py` to generate the Markdown report.
    *   List the output files.

---

**Method 2: Running Individual Python Scripts Manually**

If you want more control or need to debug a specific step, you can run the Python scripts one by one.

1.  **Navigate to the Repository Root:**
    ```bash
    cd /path/to/your/Holistic-Performance-Enhancement
    ```

2.  **Activate your virtual environment:**
    ```bash
    source .venv/bin/activate
    ```

3.  **Run the scripts in order (ensure `PYTHONPATH` allows finding the `cultivation` package):**

    *   **Step 1: Ingest Git Commits (Raw + Enriched)**
        ```bash
        PYTHONPATH=. python3 cultivation/scripts/software/dev_daily_reflect/ingest_git.py
        ```

    *   **Step 2: Aggregate Daily Metrics**
        ```bash
        PYTHONPATH=. python3 cultivation/scripts/software/dev_daily_reflect/aggregate_daily.py
        ```

    *   **Step 3: Generate Markdown Report**
        ```bash
        PYTHONPATH=. python3 cultivation/scripts/software/dev_daily_reflect/report_md.py
        ```

    **Note on `PYTHONPATH=.`**: This tells Python to include the current directory (your project root) in the module search path, which is necessary for imports like `from cultivation.scripts...` to work correctly.

---

**Configuration:**

*   The behavior of the scripts, especially `ingest_git.py`, is controlled by `cultivation/scripts/software/dev_daily_reflect/config/daily_review.yaml`.
*   **`lookback_days`**: By default, it's set to `1`. If you want to analyze more than the last day's commits, you can temporarily change this value in the YAML file before running the scripts. For example, change `lookback_days: 1` to `lookback_days: 7` to get data for the last week.
*   **`repository_path`**: This is set to `../../../..` relative to the config file, which should correctly point to your project root if the scripts are invoked as intended by `test_dev_daily_reflect.sh` (from the project root with `PYTHONPATH=.`).

---

**Expected Output:**

After running the pipeline, you will find the generated files in `cultivation/outputs/software/dev_daily_reflect/`:

*   `raw/git_commits_YYYY-MM-DD.json`: Raw commit data.
*   `raw/git_commits_enriched_YYYY-MM-DD.json`: Commits enriched with CC, MI, and Ruff metrics.
*   `rollup/dev_metrics_YYYY-MM-DD.csv`: Daily metrics aggregated per author.
*   `reports/dev_report_YYYY-MM-DD.md`: The final Markdown report.

Replace `YYYY-MM-DD` with the current date.

---

**Troubleshooting:**

*   **Permissions:** If `test_dev_daily_reflect.sh` doesn't run, ensure it has execute permissions (`chmod +x ...`).
*   **Dependencies:** Double-check that all packages in `requirements.txt` are installed in your active Python environment.
*   **Git History:** The scripts rely on `git log`. If you have a very shallow clone or no commits within the `lookback_days` window, the output might be empty or minimal.
*   **Ruff:** If `ruff` is not found, the enrichment step for Ruff errors will be skipped or might show warnings. Ensure it's installed and in your PATH.
*   **Relative Paths:** The scripts and config use relative paths. Running them from the correct directory (project root) with the correct `PYTHONPATH` is crucial. The shell script handles this well.