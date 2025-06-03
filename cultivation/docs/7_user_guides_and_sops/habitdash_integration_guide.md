# Habit Dash API Integration & Wellness Context Pipeline

## Overview
This document describes the integration of Habit Dash API data into the cultivation project, including:
- Automated fetching and caching of daily wellness metrics
- Use of these metrics in run summaries and fatigue monitoring
- Automation of the sync process via GitHub Actions

---

## 1. Data Flow & Files

- **`cultivation/scripts/sync_habitdash.py`**: Fetches daily wellness metrics (HRV, RHR, Recovery, Sleep, etc.) from the Habit Dash API and saves them to `cultivation/data/daily_wellness.parquet`.
- **`cultivation/data/daily_wellness.parquet`**: Parquet file containing daily metrics for Whoop and Garmin, indexed by date.
- **`cultivation/scripts/utilities/habitdash_api.py`**: Contains the API client and the FIELD_IDS mapping for all supported metrics.
- **`cultivation/scripts/utilities/field_id_discovery.py`**: Script to discover and print available metric field IDs from the Habit Dash API.
- **`cultivation/scripts/utilities/field_id_discovery_output.txt`**: Output of the above script for reference.

---

## 2. Usage Instructions

### A. Fetching and Caching Wellness Data
- Run the sync script manually:
  ```sh
  HABITDASH_API_KEY=your_key python cultivation/scripts/sync_habitdash.py
  ```
- This will update `cultivation/data/daily_wellness.parquet` with the latest metrics.

### B. Integrating Wellness Data in Run Summaries
- **`cultivation/scripts/running/run_performance_analysis.py`** now automatically loads the Parquet cache and inserts a pre-run wellness context block into each `run_summary.txt`.
    - Metrics shown: HRV (Whoop), RHR (Whoop), Recovery Score (Whoop), Sleep Score (Whoop), Body Battery (Garmin), and previous day's Avg Stress (Garmin).
    - Missing data is handled gracefully (shows `n/a`).

### C. Fatigue Alerts with Wellness Data
- **`cultivation/scripts/running/fatigue_watch.py`** now loads RHR, HRV, and Recovery Score from the Parquet cache (if available) for the last 7 days.
    - If objective values are missing, the script falls back to subjective values from `subjective.csv`.
    - The most recent Recovery Score is included in the GitHub issue body if present.

---

## 3. Automation via GitHub Actions

- **Workflow file:** `.github/workflows/sync-habitdash.yml`
- **What it does:**
    - Runs daily at 5:00 AM UTC (or on manual trigger)
    - Installs dependencies, runs the sync script, and commits/pushes updates to the Parquet cache
    - Requires the API key to be set as a repository secret: `HABITDASH_API_KEY`
- **How to enable:**
    1. Add your Habit Dash API key as a secret in your GitHub repo settings (`HABITDASH_API_KEY`)
    2. Ensure the workflow file exists and is committed
    3. The workflow will keep your wellness data up-to-date automatically

---

## 4. Field ID Discovery

- To audit or expand available metrics, run:
  ```sh
  PYTHONPATH=cultivation/scripts python cultivation/scripts/utilities/field_id_discovery.py > cultivation/scripts/utilities/field_id_discovery_output.txt
  ```
- Reference the output file to update `FIELD_IDS` as needed.

---

## 5. Troubleshooting & Best Practices
- If the sync script fails, check your API key and network connectivity.
- Always ensure `requirements.txt` is up-to-date (`python-dotenv` required).
- If adding new metrics, update both `FIELD_IDS` and `METRICS_TO_FETCH` in the sync script.
- Use the Parquet cache (`daily_wellness.parquet`) as the single source of truth for all downstream analytics and reporting scripts.

---

## 6. Summary of Key Files

| File/Script                                              | Purpose                                                      |
|----------------------------------------------------------|--------------------------------------------------------------|
| `cultivation/scripts/sync_habitdash.py`                  | Fetch & cache daily wellness metrics                         |
| `cultivation/data/daily_wellness.parquet`                | Cached daily metrics (Parquet)                               |
| `cultivation/scripts/utilities/habitdash_api.py`         | API client, FIELD_IDS mapping                                |
| `cultivation/scripts/utilities/field_id_discovery.py`    | Discover available metric field IDs                          |
| `cultivation/scripts/utilities/field_id_discovery_output.txt` | Output of discovery script                              |
| `cultivation/scripts/running/run_performance_analysis.py`| Adds wellness context to run summaries                       |
| `cultivation/scripts/running/fatigue_watch.py`           | Uses wellness cache for fatigue alerts                       |
| `.github/workflows/sync-habitdash.yml`                   | Automates daily wellness data sync                           |

---

For further questions or to extend this integration, see the code comments or contact the maintainer.
