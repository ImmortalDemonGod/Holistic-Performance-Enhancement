"""
Read training_plans/2025_Q2_BaseOx.csv, push next-7-days tasks to Task-Master,
and emit planning IDs '2025w18-Mon', '2025w18-Tue', … for each session.
"""
import pandas as pd
from datetime import datetime, timedelta
import subprocess
import os
from pathlib import Path

CALENDAR_CSV = Path(__file__).parent.parent.parent / 'training_plans' / '2025_Q2_BaseOx.csv'
LOOKUP_TABLE = Path(__file__).parent.parent.parent / 'data' / 'pid_lookup.csv'

MODEL = os.getenv('MODEL', 'claude-3-opus-20240229')

def emit_pid(date):
    week = date.isocalendar().week
    year = date.isocalendar().year
    day_map = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    dow = day_map[date.weekday()]
    return f"{year}w{week:02d}-{dow}"

def main():
    df = pd.read_csv(CALENDAR_CSV)
    today = datetime.now().date()
    next_7 = [today + timedelta(days=i) for i in range(7)]
    tasks = []
    lookup_rows = []
    for day in next_7:
        # Find matching row in calendar
        row = df[df['date'] == str(day)]
        if row.empty:
            continue
        row = row.iloc[0]
        pid = emit_pid(day)
        task_title = row.get('session_title', f"Run {pid}")
        scheduled_time = f"{day}T06:00:00"  # Default 6am
        session_code = row.get('session_code', 'run')
        planned_duration = row.get('planned_duration', 0)
        intensity_pct = row.get('intensity_pct', 0)
        # Push to Task Master
        subprocess.run([
            'task-master', 'create', task_title,
            '--scheduled', scheduled_time,
            '--labels', session_code, pid
        ], check=True)
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