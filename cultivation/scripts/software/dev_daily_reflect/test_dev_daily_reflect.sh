#!/bin/bash
set -e
VENV_PY="$(dirname "$0")/../../../../.venv/bin/python"

# Ensure the virtual environment exists
if [ ! -f "$VENV_PY" ]; then
    echo "[ERROR] Python virtual environment not found at $VENV_PY. Please set it up using 'python3 -m venv .venv' in the project root, then install dependencies with 'source .venv/bin/activate && pip install -r requirements.txt'." >&2
    exit 1
fi

SCRIPT_DIR="$(dirname "$0")"
REPORTS_DIR="${SCRIPT_DIR}/../../outputs/software/dev_daily_reflect/reports"

# Ensure PYTHONPATH includes the project root for imports like 'from cultivation...'
export PYTHONPATH=".:$PYTHONPATH"

# --- Configuration for Backfill ---
BACKFILL_DAYS=7 # Number of past days to check for missing reports (e.g., 7 for a week)

# --- Backfill Logic ---
echo "\n[Phase 1] Checking for and backfilling missing reports from the last ${BACKFILL_DAYS} days..."

# Determine the OS for date command compatibility
if [[ "$(uname)" == "Darwin" ]]; then # macOS
    DATE_CMD="gdate" # Requires GNU date, install with 'brew install coreutils'
    # Check if gdate is installed
    if ! command -v gdate &> /dev/null; then
        echo "[ERROR] GNU date (gdate) not found. On macOS, please install with 'brew install coreutils'."
        echo "Skipping backfill."
        BACKFILL_DAYS=0 # Skip backfill if gdate is not available
    fi
else # Linux
    DATE_CMD="date"
fi

if [[ ${BACKFILL_DAYS} -gt 0 ]]; then
    for i in $(seq ${BACKFILL_DAYS} -1 1); do # Loop from N days ago to 1 day ago
        TARGET_DATE=$(${DATE_CMD} -d "-${i} days" +%Y-%m-%d)
        REPORT_FILE="${REPORTS_DIR}/dev_report_${TARGET_DATE}.md"

        if [ ! -f "${REPORT_FILE}" ]; then
            echo "\n[Backfill] Report for ${TARGET_DATE} not found. Generating..."
            echo "  [Step 1 - Backfill ${TARGET_DATE}] Running ingest_git.py --date ${TARGET_DATE}..."
            $VENV_PY "${SCRIPT_DIR}/ingest_git.py" --date "${TARGET_DATE}"
            if [ $? -ne 0 ]; then echo "[ERROR] ingest_git.py failed for ${TARGET_DATE}"; continue; fi

            echo "  [Step 2 - Backfill ${TARGET_DATE}] Running aggregate_daily.py --date ${TARGET_DATE}..."
            $VENV_PY "${SCRIPT_DIR}/aggregate_daily.py" --date "${TARGET_DATE}"
            if [ $? -ne 0 ]; then echo "[ERROR] aggregate_daily.py failed for ${TARGET_DATE}"; continue; fi

            echo "  [Step 3 - Backfill ${TARGET_DATE}] Running report_md.py --date ${TARGET_DATE}..."
            $VENV_PY "${SCRIPT_DIR}/report_md.py" --date "${TARGET_DATE}"
            if [ $? -ne 0 ]; then echo "[ERROR] report_md.py failed for ${TARGET_DATE}"; fi
        else
            echo "[Backfill] Report for ${TARGET_DATE} already exists. Skipping."
        fi
    done
fi
echo "\n[Phase 1] Backfill check complete."

# --- Original Pipeline Run (for current/latest period) ---
echo "\n[Phase 2] Running pipeline for current/latest period..."

# Step 1: Ingest git commits (raw + enriched)
echo "[Step 1] Running ingest_git.py..."
PYTHONPATH=. $VENV_PY cultivation/scripts/software/dev_daily_reflect/ingest_git.py

# Step 2: Aggregate daily metrics
echo "[Step 2] Running aggregate_daily.py..."
PYTHONPATH=. $VENV_PY cultivation/scripts/software/dev_daily_reflect/aggregate_daily.py

# Step 3: Generate Markdown report
echo "[Step 3] Running report_md.py..."
PYTHONPATH=. $VENV_PY cultivation/scripts/software/dev_daily_reflect/report_md.py

# Step 4: Check for expected outputs
echo "\n[Summary] Checking for outputs..."
ls -lh cultivation/outputs/software/dev_daily_reflect/raw/git_commits_*.json || echo "[WARN] No commit JSONs found."
ls -lh cultivation/outputs/software/dev_daily_reflect/rollup/dev_metrics_*.csv || echo "[WARN] No rollup CSV found."
ls -lh cultivation/outputs/software/dev_daily_reflect/reports/dev_report_*.md || echo "[WARN] No Markdown report found."
