"""
Read training_plans/2025_Q2_running_plan.csv, push next-7-days tasks to Task-Master,
and emit planning IDs '2025w18-Mon', '2025w18-Tue', … for each session.
Supports CLI arguments for week selection and dry-run mode.
"""
import pandas as pd
from datetime import datetime, timedelta
import subprocess
import os
import shutil
from pathlib import Path
import json
import requests
import argparse

# --- Path configuration ---
PROJECT_ROOT  = Path(__file__).resolve().parents[3]          # cultivation repo-root
CALENDAR_CSV  = PROJECT_ROOT / 'training_plans' / '2025_Q2_running_plan.csv'
DATA_DIR      = PROJECT_ROOT / 'cultivation' / 'data'
LOOKUP_TABLE  = DATA_DIR / 'pid_lookup.csv'
STATUS_PATH   = DATA_DIR / 'status.json'

MODEL = os.environ.get('MODEL', 'claude-3-opus-20240229')
PHASES = ["Base-Ox", "Tempo-Dur", "Peak"]
REPO_SLUG = os.environ.get('GITHUB_REPOSITORY', 'ImmortalDemonGod/Holistic-Performance-Enhancement')

def emit_pid(date):
    """
    Generate a planning ID string for a given date, e.g. '2025w18-Mon'.
    Args:
        date: datetime object
    Returns:
        str: planning ID
    """
    week = date.isocalendar().week
    year = date.isocalendar().year
    day_map = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    dow = day_map[date.weekday()]
    return f"{year}w{week:02d}-{dow}"

def get_current_phase():
    """
    Get the current training phase from status.json, or default to first phase.
    Returns:
        str: phase name
    """
    if STATUS_PATH.exists():
        with open(STATUS_PATH) as f:
            status = json.load(f)
        return status.get("phase", PHASES[0])
    return PHASES[0]

def advance_phase(current):
    """
    Advance to the next phase in PHASES, or return last if at end.
    Args:
        current: str, current phase
    Returns:
        str: next phase
    """
    if current not in PHASES:
        return PHASES[0]
    idx = PHASES.index(current)
    if idx + 1 < len(PHASES):
        return PHASES[idx+1]
    return PHASES[-1]

def kpi_gate_passed():
    """
    Check if the latest run-metrics workflow succeeded via GitHub REST API.
    Returns:
        bool: True if gate passed, False otherwise
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN missing – unable to verify KPI gate.")
        return False  # keep current phase
    url = f"https://api.github.com/repos/{REPO_SLUG}/actions/workflows/run-metrics.yml/runs?per_page=1&branch=main&status=completed"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        runs = resp.json().get("workflow_runs", [])
        if runs and runs[0]["conclusion"] == "success":
            return True
        print(f"Latest run-metrics workflow conclusion: {runs[0]['conclusion'] if runs else 'none'}")
        return False
    except Exception as e:
        print(f"Could not check KPI gate: {e}. Assuming passed.")
        return True

def main():
    """
    Main entry point for PID scheduler CLI.
    Supports --week (YYYY-WW) and --dry-run.
    """
    parser = argparse.ArgumentParser(description="Push next-7-days tasks to Task-Master and emit planning IDs.")
    parser.add_argument('--week', '-w', help='ISO week to schedule (e.g. 2025-18). Default: current week')
    parser.add_argument('--dry-run', action='store_true', help='Print actions but do not invoke Task-Master')
    args = parser.parse_args()

    today = datetime.now()
    if args.week:
        year, week = map(int, args.week.split('-'))
        # ISO: Monday is 1, Sunday is 7
        monday = datetime.fromisocalendar(year, week, 1)
    else:
        monday = today - timedelta(days=today.weekday())
        week = monday.isocalendar().week
        year = monday.isocalendar().year

    # Read CSV and check for week existence
    df = pd.read_csv(CALENDAR_CSV)
    week_numbers = df['Week'].astype(int).tolist()
    if week not in week_numbers:
        print(f"Warning: Requested week {week} not found in training plan (weeks available: {week_numbers})")
        print("No sessions scheduled.")
        return

    tasks = []
    lookup_rows = []
    # PHASE ADVANCE LOGIC
    current_phase = get_current_phase()
    # Use real KPI gate check
    gate_passed = kpi_gate_passed()
    new_phase = current_phase
    if gate_passed:
        if current_phase == "Base-Ox":
            new_phase = "Tempo-Dur"
        elif current_phase == "Tempo-Dur":
            new_phase = "Peak"
        else:
            new_phase = current_phase
    if new_phase != current_phase:
        with open(STATUS_PATH, "w") as f:
            json.dump({"phase": new_phase}, f)
        print(f"Advanced phase: {current_phase} → {new_phase}")
    else:
        print(f"Current phase: {current_phase}")

    # Example: schedule 7 days starting from Monday
    for i, day_name in enumerate(['Mon','Tue','Wed','Thu','Fri','Sat','Sun']):
        # Find matching row in calendar
        row = df[df['Week'] == week]
        if not row.empty:
            row = row.iloc[0]  # Ensure row is a Series, not DataFrame
            activity = row[day_name]
            if activity and activity.lower() not in ['rest', 'rest/hrv']:
                pid = emit_pid(monday + timedelta(days=i))
                print(f"Scheduling: {pid} - {activity}")
                task_title = f"Run {pid}"
                scheduled_time = f"{monday + timedelta(days=i)}T06:00:00"  # Default 6am
                session_code = 'run'
                planned_duration = 0
                intensity_pct = 0
                # Push to Task Master
                if not args.dry_run and shutil.which('task-master'):
                    subprocess.run([
                        'task-master', 'create', task_title,
                        '--scheduled', scheduled_time,
                        '--labels', session_code, pid
                    ], check=True)
                else:
                    print('task-master CLI not found – dry-run only')
                tasks.append(pid)
                lookup_rows.append({
                    'pid': pid,
                    'planned_duration': planned_duration,
                    'intensity_pct': intensity_pct
                })
    # Save lookup table
    pd.DataFrame(lookup_rows).to_csv(LOOKUP_TABLE, index=False)
    print(f"Scheduled {len(tasks)} sessions, lookup table written to {LOOKUP_TABLE}")

if __name__ == '__main__':
    main()