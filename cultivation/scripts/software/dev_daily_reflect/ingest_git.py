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
from cultivation.scripts.software.dev_daily_reflect.utils import get_repo_root
from pathlib import Path
from typing import TypedDict, List, Optional, cast, Dict, Any
import os

class CommitData(TypedDict):
    sha: str
    author: str
    timestamp: str
    message: str
    added: int
    deleted: int

def run_ingestion_logic(repo_root_path: Path, output_dir_path: Path, script_args):
    """Encapsulates the main logic of the script."""
    config = load_config()

    if script_args.date:
        try:
            target_date_obj = datetime.datetime.strptime(script_args.date, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc)
            SINCE_DT = target_date_obj.replace(hour=0, minute=0, second=0, microsecond=0)
            UNTIL_DT = SINCE_DT + datetime.timedelta(days=1)
            SINCE_ISO = SINCE_DT.isoformat()
            UNTIL_ISO = UNTIL_DT.isoformat()
            DATE_TAG = script_args.date
            git_cmd_time_args = [f'--since={SINCE_ISO}', f'--until={UNTIL_ISO}']
            print(f"[INFO] Processing git logs for specific date: {DATE_TAG} (from {SINCE_ISO} to {UNTIL_ISO})")
        except ValueError:
            print(f"[ERROR] Invalid date format for --date: {script_args.date}. Please use YYYY-MM-DD.")
            sys.exit(1)
    else:
        now = datetime.datetime.now(datetime.timezone.utc)
        try:
            lookback_days = int(config.get("lookback_days", 1))
        except ValueError:
            print(f"[ERROR] Invalid lookback_days value in config: {config.get('lookback_days')}")
            lookback_days = 1
        start_dt = now - datetime.timedelta(days=lookback_days)
        SINCE_ISO = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        DATE_TAG = now.strftime('%Y-%m-%d')
        git_cmd_time_args = [f'--since={SINCE_ISO}']
        print(f"[INFO] Processing git logs since {SINCE_ISO} (lookback: {lookback_days} days, current date tag: {DATE_TAG})")

    cmd = [
        'git', 'log',
        *git_cmd_time_args,
        '--pretty=format:%H|%an|%ai|%s', '--numstat'
    ]
    try:
        raw = subprocess.check_output(cmd, text=True, cwd=repo_root_path)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] git log failed: {e}", file=sys.stderr)
        sys.exit(1)

    records: List[CommitData] = []
    cur: Optional[CommitData] = None
    
    sha_str: str
    author_str: str
    ts_str: str
    msg_str: str

    for line in raw.splitlines():
        if '|' in line:
            if cur is not None:
                records.append(cur)
            sha_str, author_str, ts_str, msg_str = line.split('|', 3)
            cur = {'sha': sha_str, 'author': author_str, 'timestamp': ts_str, 'message': msg_str, 'added': 0, 'deleted': 0}
        elif line.strip() and cur is not None:
            m = re.match(r'^(\d+|-)\t(\d+|-)\t', line)
            if m:
                add_val_str, delt_val_str = m.group(1), m.group(2)
                if add_val_str.isdigit():
                    cur['added'] += int(add_val_str)
                if delt_val_str.isdigit():
                    cur['deleted'] += int(delt_val_str)
    if cur is not None:
        records.append(cur)

    output_dir_path.mkdir(parents=True, exist_ok=True)
    outfile = output_dir_path / f'git_commits_{DATE_TAG}.json'
    with open(outfile, 'w') as f:
        json.dump(records, f, indent=2)
    print(f'[✓] wrote {outfile} ({len(records)} commits)')

    try:
        from cultivation.scripts.software.dev_daily_reflect.metrics.commit_processor import analyze_commits_code_quality
        # Cast records to satisfy analyze_commits_code_quality if it expects List[Dict[Any, Any]]
        enriched = analyze_commits_code_quality(str(repo_root_path), cast(List[Dict[Any, Any]], records))
        enriched_outfile = output_dir_path / f'git_commits_enriched_{DATE_TAG}.json'
        with open(enriched_outfile, 'w') as f:
            json.dump(enriched, f, indent=2)
        print(f'[✓] wrote {enriched_outfile} (enriched with code metrics)')
    except Exception as e:
        print(f'[WARN] Could not enrich with code metrics: {e.__class__.__name__}: {e}')
        import traceback
        print(f'Traceback: {traceback.format_exc()}')

def main_script_runner(custom_argv=None):
    parser = argparse.ArgumentParser(description='Ingest git commits for a specific date or the configured lookback period.')
    parser.add_argument('--date', type=str, help='Target date in YYYY-MM-DD format. If not provided, uses lookback_days from config.')
    script_args = parser.parse_args(custom_argv)

    # Determine REPO_ROOT, allowing override for testing
    repo_root_override = os.environ.get('CULTIVATION_REPO_ROOT_OVERRIDE')
    if repo_root_override:
        actual_repo_root = Path(repo_root_override)
        print(f"[INFO] Using overridden REPO_ROOT: {actual_repo_root}") # For test visibility
    else:
        actual_repo_root = get_repo_root()
    
    actual_output_dir = actual_repo_root / 'cultivation' / 'outputs' / 'software' / 'dev_daily_reflect' / 'raw'
    
    run_ingestion_logic(actual_repo_root, actual_output_dir, script_args)

if __name__ == "__main__":
    main_script_runner()
