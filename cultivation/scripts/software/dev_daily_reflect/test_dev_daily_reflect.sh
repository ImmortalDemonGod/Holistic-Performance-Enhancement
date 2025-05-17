#!/bin/bash
set -e
VENV_PY="/Users/tomriddle1/Holistic-Performance-Enhancement/.venv/bin/python"

# Step 1: Ingest git commits (raw + enriched)
echo "[Step 1] Running ingest_git.py..."
$VENV_PY cultivation/scripts/software/dev_daily_reflect/ingest_git.py

# Step 2: Aggregate daily metrics
echo "[Step 2] Running aggregate_daily.py..."
$VENV_PY cultivation/scripts/software/dev_daily_reflect/aggregate_daily.py

# Step 3: Generate Markdown report
echo "[Step 3] Running report_md.py..."
$VENV_PY cultivation/scripts/software/dev_daily_reflect/report_md.py

# Step 4: Check for expected outputs
echo "\n[Summary] Checking for outputs..."
ls -lh cultivation/outputs/software/dev_daily_reflect/raw/git_commits_*.json || echo "[WARN] No commit JSONs found."
ls -lh cultivation/outputs/software/dev_daily_reflect/rollup/dev_metrics_*.csv || echo "[WARN] No rollup CSV found."
ls -lh cultivation/outputs/software/dev_daily_reflect/reports/dev_report_*.md || echo "[WARN] No Markdown report found."
