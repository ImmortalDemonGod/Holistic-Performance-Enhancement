# aggregate_daily.py
# Aggregates commit data per author/day for DevDailyReflect (MVP)
# Reads raw commit JSON and writes daily rollup CSV

import pandas as pd
import pathlib
import datetime
import sys

# --- Configuration ---
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
RAW_DIR = REPO_ROOT / 'cultivation' / 'outputs' / 'software' / 'dev_daily_reflect' / 'raw'
ROLLUP_DIR = REPO_ROOT / 'cultivation' / 'outputs' / 'software' / 'dev_daily_reflect' / 'rollup'
ROLLUP_DIR.mkdir(parents=True, exist_ok=True)

# --- Find latest raw JSON file ---
raw_files = sorted(RAW_DIR.glob('git_commits_*.json'))
if not raw_files:
    print('[WARN] No raw commit JSON found. Exiting.')
    sys.exit(0)
raw_file = raw_files[-1]

# --- Load commit data ---
try:
    df = pd.read_json(raw_file)
except Exception as e:
    print(f'[ERROR] Failed to read {raw_file}: {e}')
    sys.exit(1)

if df.empty:
    print('[INFO] No commits found in the last 24h.')
    # Still write an empty rollup with headers
    rollup_file = ROLLUP_DIR / f'dev_metrics_{raw_file.stem[-10:]}.csv'
    pd.DataFrame(columns=['author','commits','loc_add','loc_del','loc_net']).to_csv(rollup_file, index=False)
    print(f'[✓] wrote empty {rollup_file}')
    sys.exit(0)

# --- Aggregate per author ---
gb = df.groupby('author').agg(
    commits=('sha', 'count'),
    loc_add=('added', 'sum'),
    loc_del=('deleted', 'sum')
).reset_index()
gb['loc_net'] = gb['loc_add'] - gb['loc_del']

# --- Write rollup CSV ---
date_tag = raw_file.stem[-10:]
rollup_file = ROLLUP_DIR / f'dev_metrics_{date_tag}.csv'
gb.to_csv(rollup_file, index=False)
print(f'[✓] wrote {rollup_file}')
