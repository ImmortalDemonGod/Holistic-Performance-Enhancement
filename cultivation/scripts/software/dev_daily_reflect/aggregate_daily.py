# aggregate_daily.py
# Aggregates commit data per author/day for DevDailyReflect (MVP)
# Reads raw commit JSON and writes daily rollup CSV

#!/usr/bin/env python3
import pandas as pd
import sys
from cultivation.scripts.software.dev_daily_reflect.utils import get_repo_root

# --- Configuration ---
REPO_ROOT = get_repo_root()
RAW_DIR = REPO_ROOT / 'cultivation' / 'outputs' / 'software' / 'dev_daily_reflect' / 'raw'
ROLLUP_DIR = REPO_ROOT / 'cultivation' / 'outputs' / 'software' / 'dev_daily_reflect' / 'rollup'
ROLLUP_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # --- Find latest enriched or raw JSON file ---
    date_tag = None
    enriched_files = sorted(RAW_DIR.glob('git_commits_enriched_*.json'))
    if enriched_files:
        json_file = enriched_files[-1]
        date_tag = json_file.stem[-10:]
    else:
        raw_files = sorted(RAW_DIR.glob('git_commits_*.json'))
        if not raw_files:
            print(f'[WARN] No commit JSON file found for {date_tag}. Exiting.')
            sys.exit(2)
        json_file = raw_files[-1]
        date_tag = json_file.stem[-10:]
    date_tag = json_file.stem[-10:]

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
