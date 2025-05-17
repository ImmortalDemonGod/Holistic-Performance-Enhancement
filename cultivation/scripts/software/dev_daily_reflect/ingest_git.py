#!/usr/bin/env python3
# ingest_git.py
# Fetches commit data for DevDailyReflect pipeline (MVP)
# Writes raw commit data as JSON to outputs/software/dev_daily_reflect/raw/

import subprocess
import datetime
import json
import sys
import re
from cultivation.scripts.software.dev_daily_reflect.utils import get_repo_root

# --- Configuration ---
REPO_ROOT = get_repo_root()  # repo root
OUTPUT_DIR = REPO_ROOT / 'cultivation' / 'outputs' / 'software' / 'dev_daily_reflect' / 'raw'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Calculate time window (last 24h, UTC) ---
now = datetime.datetime.now(datetime.timezone.utc)
start = now - datetime.timedelta(days=1)
SINCE = start.strftime('%Y-%m-%dT%H:%M:%SZ')
DATE_TAG = now.strftime('%Y-%m-%d')

# --- Run git log to get commits since SINCE ---
cmd = [
    'git', 'log', f'--since={SINCE}', '--pretty=format:%H|%an|%ai|%s', '--numstat'
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
    from metrics.commit_processor import analyze_commits_code_quality
    enriched = analyze_commits_code_quality(str(REPO_ROOT), records)
    enriched_outfile = OUTPUT_DIR / f'git_commits_enriched_{DATE_TAG}.json'
    with open(enriched_outfile, 'w') as f:
        json.dump(enriched, f, indent=2)
    print(f'[✓] wrote {enriched_outfile} (enriched with code metrics)')
except Exception as e:
    print(f'[WARN] Could not enrich with code metrics: {e}')
