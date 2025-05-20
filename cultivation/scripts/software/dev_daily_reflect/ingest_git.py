#!/usr/bin/env python3
# ingest_git.py
# Fetches commit data for DevDailyReflect pipeline (MVP)
# Writes raw commit data as JSON to outputs/software/dev_daily_reflect/raw/

import subprocess
import datetime
import json
import sys
import re
import argparse
from cultivation.scripts.software.dev_daily_reflect.config_loader import load_config
from pathlib import Path

# --- Configuration ---
config = load_config()
REPO_ROOT = (Path(__file__).parent / config["repository_path"]).resolve()
OUTPUT_DIR = REPO_ROOT / 'cultivation' / 'outputs' / 'software' / 'dev_daily_reflect' / 'raw'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='Ingest git commits for a specific date or the configured lookback period.')
parser.add_argument('--date', type=str, help='Target date in YYYY-MM-DD format. If not provided, uses lookback_days from config.')
args = parser.parse_args()

# --- Calculate time window --- 
if args.date:
    try:
        target_date_obj = datetime.datetime.strptime(args.date, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc)
        # For a specific date, we want commits *on* that day.
        # SINCE is the beginning of the target_date, UNTIL is the beginning of the next day.
        SINCE_DT = target_date_obj.replace(hour=0, minute=0, second=0, microsecond=0)
        UNTIL_DT = SINCE_DT + datetime.timedelta(days=1)
        SINCE_ISO = SINCE_DT.isoformat()
        UNTIL_ISO = UNTIL_DT.isoformat()
        DATE_TAG = args.date
        git_cmd_time_args = [f'--since={SINCE_ISO}', f'--until={UNTIL_ISO}']
        print(f"[INFO] Processing git logs for specific date: {DATE_TAG} (from {SINCE_ISO} to {UNTIL_ISO})")
    except ValueError:
        print(f"[ERROR] Invalid date format for --date: {args.date}. Please use YYYY-MM-DD.")
        sys.exit(1)
else:
    # --- Calculate time window (configurable lookback_days) --- (Original behavior)
    now = datetime.datetime.now(datetime.timezone.utc)
    try:
        lookback_days = int(config.get("lookback_days", 1))
    except ValueError:
        print(f"[ERROR] Invalid lookback_days value in config: {config.get('lookback_days')}")
        lookback_days = 1  # Default to 1 day if invalid
    start_dt = now - datetime.timedelta(days=lookback_days)
    SINCE_ISO = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ') # Original format for lookback
    DATE_TAG = now.strftime('%Y-%m-%d')
    git_cmd_time_args = [f'--since={SINCE_ISO}']
    print(f"[INFO] Processing git logs since {SINCE_ISO} (lookback: {lookback_days} days, current date tag: {DATE_TAG})")

# --- Run git log to get commits --- 
cmd = [
    'git', 'log', 
    *git_cmd_time_args, # Use the determined time arguments
    '--pretty=format:%H|%an|%ai|%s', '--numstat'
]
try:
    raw = subprocess.check_output(cmd, text=True, cwd=REPO_ROOT)
except subprocess.CalledProcessError as e:
    print(f"[ERROR] git log failed: {e}", file=sys.stderr)
    sys.exit(1)

records = []
sha = author = ts = msg = None
cur = None
for line in raw.splitlines():
    if '|' in line:
        if cur:
            records.append(cur)
        sha, author, ts, msg = line.split('|', 3)
        cur = dict(sha=sha, author=author, timestamp=ts, message=msg, added=0, deleted=0)
    elif line.strip():
        m = re.match(r'^(\d+|-)\t(\d+|-)\t', line)
        if m and cur:
            add, delt = m.group(1), m.group(2)
            if add.isdigit(): 
                cur['added'] += int(add)
            if delt.isdigit(): 
                cur['deleted'] += int(delt)
if cur:
    records.append(cur)

# --- Write output ---
outfile = OUTPUT_DIR / f'git_commits_{DATE_TAG}.json'
with open(outfile, 'w') as f:
    json.dump(records, f, indent=2)
print(f'[✓] wrote {outfile} ({len(records)} commits)')

# --- Enrich with code quality metrics ---
try:
    from cultivation.scripts.software.dev_daily_reflect.metrics.commit_processor import analyze_commits_code_quality
    enriched = analyze_commits_code_quality(str(REPO_ROOT), records)
    enriched_outfile = OUTPUT_DIR / f'git_commits_enriched_{DATE_TAG}.json'
    with open(enriched_outfile, 'w') as f:
        json.dump(enriched, f, indent=2)
    print(f'[✓] wrote {enriched_outfile} (enriched with code metrics)')
except Exception as e:
    print(f'[WARN] Could not enrich with code metrics: {e.__class__.__name__}: {e}')
    import traceback
    print(f'Traceback: {traceback.format_exc()}')
