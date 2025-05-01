"""
Read training_plans/2025_Q2_running_plan.csv, push next-7-days tasks to Task-Master,
and emit planning IDs '2025w18-Mon', '2025w18-Tue', … for each session.
"""
import pandas as pd
from datetime import datetime, timedelta
import subprocess
import os
import shutil
from pathlib import Path
import json
import requests

CALENDAR_CSV = Path(__file__).parents[3] / 'training_plans' / '2025_Q2_running_plan.csv'
LOOKUP_TABLE = Path(__file__).parent.parent.parent / 'data' / 'pid_lookup.csv'
STATUS_PATH = Path(__file__).parent.parent.parent / 'data' / 'status.json'

MODEL = os.getenv('MODEL', 'claude-3-opus-20240229')

PHASES = ["Base-Ox", "Tempo-Dur", "Peak"]

REPO_SLUG = os.environ.get('GITHUB_REPOSITORY', 'ImmortalDemonGod/Holistic-Performance-Enhancement')

def emit_pid(date):
    week = date.isocalendar().week
    year = date.isocalendar().year
    day_map = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    dow = day_map[date.weekday()]
    return f"{year}w{week:02d}-{dow}"

def get_current_phase():
    if STATUS_PATH.exists():
        with open(STATUS_PATH) as f:
            status = json.load(f)
        return status.get("phase", PHASES[0])
    return PHASES[0]

def advance_phase(current):
    if current not in PHASES:
        return PHASES[0]
    idx = PHASES.index(current)
    if idx + 1 < len(PHASES):
        return PHASES[idx+1]
    return PHASES[-1]

def kpi_gate_passed():
    """Check if the latest run-metrics workflow succeeded via GitHub REST API."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Warning: GITHUB_TOKEN not set; assuming gate passed.")
        return True
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
    df = pd.read_csv(CALENDAR_CSV)
    today = datetime.now().date()
    next_7 = [today + timedelta(days=i) for i in range(7)]
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
    for day in next_7:
        # Find matching row in calendar
        if 'date' in df.columns:
            row = df[df['date'] == str(day)]
            if row.empty:
                continue
            row = row.iloc[0]
        else:
            week = day.isocalendar().week
            dow = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day.weekday()]
            row = df.loc[df['Week'] == week, dow]
            if row.empty or row.iloc[0] in ('Rest',''):
                continue
            session = row.iloc[0]
            row = pd.DataFrame({'session_title': [session]})
        pid = emit_pid(day)
        task_title = row.get('session_title', f"Run {pid}")
        scheduled_time = f"{day}T06:00:00"  # Default 6am
        session_code = row.get('session_code', 'run')
        planned_duration = row.get('planned_duration', 0)
        intensity_pct = row.get('intensity_pct', 0)
        # Push to Task Master
        if shutil.which('task-master'):
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