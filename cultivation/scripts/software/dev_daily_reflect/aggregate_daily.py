# aggregate_daily.py
# Aggregates commit data per author/day for DevDailyReflect (MVP)
# Reads raw commit JSON and writes daily rollup CSV

#!/usr/bin/env python3
import pandas as pd
import sys
import argparse
import datetime
from cultivation.scripts.software.dev_daily_reflect.config_loader import load_config
from pathlib import Path

# --- Configuration ---
config = load_config()
REPO_ROOT = (Path(__file__).parent / config["repository_path"]).resolve()
RAW_DIR = REPO_ROOT / 'cultivation' / 'outputs' / 'software' / 'dev_daily_reflect' / 'raw'
ROLLUP_DIR = REPO_ROOT / config["rollup_dir"]
ROLLUP_DIR.mkdir(parents=True, exist_ok=True)

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='Aggregate daily dev metrics for a specific date or the latest available data.')
parser.add_argument('--date', type=str, help='Target date in YYYY-MM-DD format. If not provided, processes the latest raw data file.')
args = parser.parse_args()

def main():
    # --- Determine input file and date_tag ---
    """
    Aggregates daily commit data per author and writes a rollup CSV file.
    
    Determines the appropriate commit JSON file (enriched or raw) for a specified date or the latest available, loads the data, and aggregates metrics such as commit count, lines added, lines deleted, net lines changed, and optional metrics if present. Outputs the aggregated results to a CSV file in the configured rollup directory. Handles missing files, invalid dates, and empty data gracefully with informative messages and appropriate exit codes.
    """
    date_tag = None
    json_file = None

    if args.date:
        try:
            # Validate date format, though we mainly use it for filenames here
            datetime.datetime.strptime(args.date, '%Y-%m-%d') 
            date_tag = args.date
            # Prefer enriched, fall back to raw for the specific date
            specific_enriched_file = RAW_DIR / f'git_commits_enriched_{date_tag}.json'
            specific_raw_file = RAW_DIR / f'git_commits_{date_tag}.json'
            if specific_enriched_file.exists():
                json_file = specific_enriched_file
            elif specific_raw_file.exists():
                json_file = specific_raw_file
            else:
                print(f'[ERROR] No commit JSON file found for date {date_tag} in {RAW_DIR}. Exiting.')
                sys.exit(2)
            print(f"[INFO] Processing data for specific date: {date_tag} from {json_file.name}")
        except ValueError:
            print(f"[ERROR] Invalid date format for --date: {args.date}. Please use YYYY-MM-DD.")
            sys.exit(1)
    else:
        # --- Find latest enriched or raw JSON file (Original behavior) ---
        enriched_files = sorted(RAW_DIR.glob('git_commits_enriched_*.json'))
        if enriched_files:
            json_file = enriched_files[-1]
            date_tag = json_file.stem.split('_')[-1] # Extract date from filename like git_commits_enriched_YYYY-MM-DD
        else:
            raw_files = sorted(RAW_DIR.glob('git_commits_*.json'))
            if not raw_files:
                print(f'[ERROR] No commit JSON files found in {RAW_DIR}. Exiting.')
                sys.exit(2)
            json_file = raw_files[-1]
            date_tag = json_file.stem.split('_')[-1] # Extract date from filename like git_commits_YYYY-MM-DD
        print(f"[INFO] Processing latest available data: {date_tag} from {json_file.name}")

    # --- Load commit data ---
    try:
        df = pd.read_json(json_file)
    except Exception as e:
        print(f'[ERROR] Failed to read {json_file}: {e}')
        sys.exit(1)

    if df.empty:
        print('[INFO] No commits found in the last 24h.')
        # Still write an empty rollup with headers
        rollup_file = ROLLUP_DIR / f'dev_metrics_{date_tag}.csv'
        pd.DataFrame(columns=[
            'author','commits','loc_add','loc_del','loc_net',
            'py_files_changed_count','total_cc','avg_mi','ruff_errors'
        ]).to_csv(rollup_file, index=False)
        print(f'[✓] wrote empty {rollup_file}')
        sys.exit(0)

    # --- Aggregate per author ---
    agg_dict = {
        'sha': 'count',
        'added': 'sum',
        'deleted': 'sum',
    }
    if 'py_files_changed_count' in df.columns:
        agg_dict['py_files_changed_count'] = 'sum'
    if 'total_cc' in df.columns:
        agg_dict['total_cc'] = 'sum'
    if 'avg_mi' in df.columns:
        agg_dict['avg_mi'] = 'mean'

    grouped = df.groupby('author').agg(agg_dict).rename(columns={
        'sha': 'commits',
        'added': 'loc_add',
        'deleted': 'loc_del'
    })
    grouped['loc_net'] = grouped['loc_add'] - grouped['loc_del']
    # Handle ruff_errors column
    if 'ruff_errors' in df.columns:
        grouped['ruff_errors'] = df.groupby('author')['ruff_errors'].sum()
    # Preserve all columns
    grouped = grouped.reset_index()
    rollup_file = ROLLUP_DIR / f'dev_metrics_{date_tag}.csv'
    grouped.to_csv(rollup_file, index=False)
    print(f'[✓] wrote {rollup_file}')

if __name__ == "__main__":
    main()
