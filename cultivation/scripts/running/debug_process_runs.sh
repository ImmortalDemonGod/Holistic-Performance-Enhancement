#!/bin/bash
# Robust debug script to clean ALL processed data and output figures, then reprocess all runs using the venv Python.
# Usage: ./cultivation/scripts/running/debug_process_runs.sh

set -euo pipefail
set -x  # Print all executed commands for easier debugging

# Get the project root using git (bulletproof)
BASE_DIR="$(git rev-parse --show-toplevel)"
cd "$BASE_DIR" || exit 1

# Clean ALL output figures and processed data
rm -rf cultivation/outputs/figures/*
rm -rf cultivation/data/processed/*

# Optionally, clean any markers or cache files (uncomment if needed)
# rm -rf cultivation/outputs/cache/*
# rm -f cultivation/outputs/figures/*/weather_failed.marker

# Run the process_all_runs.py script using the venv Python for consistency
VENV_PY="$BASE_DIR/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "[ERROR] Python venv not found at $VENV_PY. Please create or activate your venv."
  exit 2
fi

# Run the pipeline
"$VENV_PY" "$BASE_DIR/cultivation/scripts/running/process_all_runs.py"

set +x

echo "Done!"
