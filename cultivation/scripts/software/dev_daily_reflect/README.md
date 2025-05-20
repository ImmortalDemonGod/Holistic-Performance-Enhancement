# DevDailyReflect Pipeline

This module provides scripts for daily developer reflection and metrics aggregation.

- **ingest_git.py**: Fetch commit data from the git repo.
- **aggregate_daily.py**: Aggregate commit data per author/day.
- **report_md.py**: Generate a Markdown daily report.

Outputs are written to `cultivation/outputs/software/dev_daily_reflect/`.
Okay, this is a fantastic set of information! You've got a working prototype (`commit_metrics_prototyping.py`), a more structured implementation (`dev_daily_reflect` scripts), a gap analysis, and a wealth of ideas from exploring existing tools.

Let's outline a program, which we can call **"DevDailyReflect"** (building on your `dev_daily_reflect` directory), that consolidates these efforts into a cohesive system for daily work review and next-day planning.

## Program: DevDailyReflect

### Goal
To provide a daily automated review of software development activity, highlighting achievements, potential risks, quality trends, and suggesting actionable next steps to inform the developer’s next day.

**Core Philosophy:**
*   **Automated Data Collection:** Minimize manual input.
*   **Actionable Insights:** Focus on what the user can *do* with the information.
*   **Holistic View (MVP):** Start with code-centric metrics, but design for future integration of other "Holistic Performance" domains.
*   **Self-Contained:** Leverage existing Python scripts and GitHub capabilities as much as possible.

---

**I. Overall Architecture (Leveraging your "Draft Architecture" and `dev_daily_reflect` structure):**

1.  **Configuration (`cultivation/scripts/software/dev_daily_reflect/config/daily_review.yaml`):**
    *   `repository_path`: Path to the Git repository to analyze.
    *   `lookback_days`: How far back to fetch commits for daily analysis (e.g., 1 for "yesterday", 2 for "last 48h").
    *   `output_base_dir`: Base for all outputs (e.g., `cultivation/outputs/software/dev_daily_reflect`).
    *   `user_aliases`: Mapping of git author names/emails to a consistent display name.
    *   `code_quality_thresholds`:
        *   `risk_loc_cc_factor_high`: e.g., 5000
        *   `risk_loc_cc_factor_medium`: e.g., 1000
        *   `max_ruff_errors_per_commit`: e.g., 5
        *   `min_mi_acceptable`: e.g., 60
        *   `max_cc_per_function_warn`: e.g., 15 (harder to get per commit easily without deep AST diffing)
    *   `pr_staleness_threshold_days`: (Future) e.g., 2
    *   `issue_staleness_threshold_days`: (Future) e.g., 7
    *   `file_extensions_to_analyze`: List of extensions for code metrics (e.g., `['.py', '.js']`).
    *   `exclude_patterns`: List of file/directory patterns to exclude from analysis.

2.  **Data Ingestion Layer (`cultivation/scripts/software/dev_daily_reflect/ingest/`)**
    *   **`git_commit_ingestor.py`** (Adapting `ingest_git.py`):
        *   Uses `git log` for specified `lookback_days`.
        *   Collects: SHA, author, date, message, files changed, lines added/deleted per file.
        *   **Output:** Raw commit data (JSON) per day (e.g., `raw/git_commits_YYYY-MM-DD.json`).
    *   **`github_api_ingestor.py`** (Future Enhancement - v2):
        *   Fetches PR data (open, merged, review times, comments).
        *   Fetches Issue data (open, closed, age).
        *   Fetches CI/CD run statuses (pass/fail, duration) via GitHub Actions API.
        *   **Output:** `raw/github_prs_YYYY-MM-DD.json`, `raw/github_issues_YYYY-MM-DD.json`, etc.

3.  **Metric Calculation & Enrichment Layer (`cultivation/scripts/software/dev_daily_reflect/metrics/`)**
    *   **`commit_enricher.py`** (Adapting `metrics/commit_processor.py`):
        *   Takes raw commit data from `git_commit_ingestor.py`.
        *   For each commit and each relevant changed file (`file_extensions_to_analyze`, not in `exclude_patterns`):
            *   Retrieves file content at that commit (`git show SHA:path/to/file`).
            *   Calculates Cyclomatic Complexity (Radon).
            *   Calculates Maintainability Index (Radon).
            *   Runs Ruff (on the file content at that commit) and counts errors/warnings.
            *   (Future) Test Coverage Delta: If `coverage.xml` is available from CI for this commit or its parent, calculate the change.
        *   Aggregates per commit: `total_cc`, `avg_mi` (weighted by LOC or simple mean), `total_ruff_errors`, `py_files_changed_count`.
        *   **Output:** Enriched commit data (JSON) (e.g., `processed/commits_enriched_YYYY-MM-DD.json`).

4.  **Aggregation Layer (`cultivation/scripts/software/dev_daily_reflect/aggregate/`)**
    *   **`daily_aggregator.py`** (Adapting `aggregate_daily.py`):
        *   Takes enriched commit data (and future PR/Issue data).
        *   **Per-Author Daily Summary:**
            *   Total commits.
            *   Total LOC added, deleted, net.
            *   Total/Average CC, MI.
            *   Total Ruff errors.
            *   Number of Python (or configured) files touched.
            *   (Future) PRs opened/merged/reviewed, Issues closed.
        *   **Overall Daily Summary:**
            *   Aggregates of the above across all authors.
        *   **Output:** `daily_rollup/summary_YYYY-MM-DD.csv` (or Parquet) and `daily_rollup/author_summary_YYYY-MM-DD.csv`.

5.  **Insight & Guidance Engine Layer (`cultivation/scripts/software/dev_daily_reflect/coach/`)**
    *   **`rules_engine.py`**:
        *   Takes aggregated daily data and enriched commit data.
        *   Applies rules based on `config.yaml` thresholds:
            *   **Identify Risky Commits:**
                *   (LOC net * Total CC) > threshold.
                *   Total Ruff errors > threshold.
                *   MI < threshold.
                *   (Future) Significant test coverage drop.
            *   **Identify Blockers (Future - requires PR/Issue data):**
                *   Stale PRs (open > `pr_staleness_threshold_days` without activity).
                *   Stale Issues.
            *   **Identify TODOs/FIXMEs:**
                *   Scans diffs of today's commits for "TODO", "FIXME".
            *   **Identify CI Failures (Future - requires CI data):**
                *   List commits that caused CI failures.
        *   **Output:** A structured list of insights/alerts/suggestions (e.g., `insights_YYYY-MM-DD.json`).
            *   `[{ "type": "RISKY_COMMIT", "sha": "abc", "reason": "High CC and LOC churn", "severity": "HIGH" }, ...]`
            *   `[{ "type": "ACTION_ITEM", "description": "Address TODO in file X.py: 'Refactor this'", "source_commit": "def" }, ...]`

6.  **Reporting Layer (`cultivation/scripts/software/dev_daily_reflect/report/`)**
    *   **`markdown_reporter.py`** (Adapting `report_md.py`):
        *   Takes aggregated summaries and insights.
        *   Generates a `dev_report_YYYY-MM-DD.md` file.
        *   **Report Structure:**
            1.  **Date & Overall Summary:** Total commits, Net LOC, Overall Quality Score (e.g., avg MI, total Ruff errors).
            2.  **Highlights & Key Alerts (from `rules_engine.py`):**
                *   🚨 High-Risk Commits (SHA, author, reason).
                *   ⚠️ Quality Concerns (e.g., commit with many Ruff errors, low MI).
                *   📉 (Future) Coverage Drops.
            3.  **Productivity Snapshot (from `daily_aggregator.py`):**
                *   Table: Author | Commits | LOC Added | LOC Del | Net LOC | Files Touched
            4.  **Code Quality Snapshot (from `daily_aggregator.py`):**
                *   Table: Author | Avg CC | Avg MI | Ruff Errors
            5.  **Blockers & Attention Needed (Future - from `rules_engine.py`):**
                *   Stalled PRs.
                *   CI Failures for the day.
            6.  **💡 Suggested Next Actions (from `rules_engine.py`):**
                *   "Review risky commit `abc123` by Alice."
                *   "Address TODO in `file.py`: 'Implement XYZ'."
                *   "Investigate CI failure for commit `def456`."
                *   (Future) "Follow up on PR #123 - stalled for 3 days."
            7.  **(Optional) Detailed Commit Log:** Table of all enriched commits for the day.
        *   **Output:** `reports/dev_report_YYYY-MM-DD.md`.

7.  **Scheduler & Orchestrator (`.github/workflows/daily_dev_review.yml` & `cultivation/scripts/software/dev_daily_reflect/main_pipeline.sh` or a Python orchestrator):**
    *   GitHub Action runs daily (e.g., early morning UTC).
    *   Calls the shell script (or Python orchestrator) which executes steps 2-6 in order.
    *   Commits the generated Markdown report back to the repository.
    *   Uploads the report as a build artifact.

---

**II. Daily Workflow for the User:**

1.  **Morning:** User checks the newly committed `dev_report_YYYY-MM-DD.md` in their repository (or receives a notification if Slack/email integration is added later).
2.  **Review Highlights & Alerts:** Quickly see if any commits were flagged as high-risk or if there are major quality concerns.
3.  **Check Productivity & Quality Snapshots:** Understand own and team's (if applicable) output and quality trends for the previous day.
4.  **Address Suggested Next Actions:**
    *   Prioritize reviewing any flagged risky commits.
    *   Look into TODOs identified.
    *   Plan to fix CI failures or address quality dips.
5.  **Plan Today's Work:** The report provides context on what was completed, what issues arose, and what needs immediate attention, helping to structure the current day's tasks.

---

**III. Implementation Steps (Leveraging your existing `dev_daily_reflect`):**

1.  **Refine Configuration (`daily_review.yaml`):** Ensure all necessary paths and thresholds are there. Implement `config_loader.py` robustly.
2.  **Solidify Ingestion (`ingest_git.py`):** Ensure it correctly uses `lookback_days` and handles `git log` parsing robustly.
3.  **Enhance Metric Calculation (`commit_processor.py`):**
    *   Ensure Ruff is run on the *content of the file at the specific commit*, not just the latest version. (Your `git show SHA:path` approach is good).
    *   Add error handling for Radon/Ruff failures on specific files (e.g., if a file is not valid Python).
    *   Integrate test coverage parsing (e.g., from `coverage xml`). This might require a separate step after CI runs tests. For a daily review, you might look for `coverage.xml` generated by the *latest* CI run on the main branch or the user's primary development branch.
4.  **Robust Aggregation (`aggregate_daily.py`):** Ensure it handles missing metrics gracefully (e.g., if Ruff errors couldn't be calculated for some commits).
5.  **Develop Insight Engine (`rules_engine.py`):** This is a key new piece. Start with simple rules for risky commits and TODOs.
6.  **Improve Reporting (`report_md.py`):** Structure it clearly as outlined above, pulling data from the aggregator and the new insights engine.
7.  **Test GitHub Action (`daily_dev_review.yml`):** Ensure it runs reliably, checks out the code, installs dependencies, executes the pipeline, and commits the report. Handle permissions for committing back.
8.  **Testing:** Add more unit and integration tests for each component (as you've started).

---

This outline provides a comprehensive program that builds directly on your existing work and incorporates the best ideas from your research. It focuses on providing tangible daily value to the developer.