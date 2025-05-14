#!/usr/bin/env python3
"""
Interactive CLI to log a strength session into Parquet files.
"""
import argparse
import os
import uuid
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / 'data' / 'strength' / 'processed'
SESSIONS_PATH = PROCESSED_DIR / 'strength_sessions.parquet'
EXERCISES_PATH = PROCESSED_DIR / 'strength_exercises_log.parquet'
LIB_PATH = PROCESSED_DIR / 'exercise_library.csv'


def prompt_session_info(default_dt):
    print('Enter session-level information (press Enter to accept default):')
    dt_str = input(f'  Session datetime UTC [{default_dt}]: ') or default_dt
    session_dt = pd.to_datetime(dt_str)
    plan_id = input('  Plan ID (optional): ').strip() or None
    wellness = input('  Wellness light [Green/Amber/Red]: ').strip() or ''
    rpe_ub = float(input('  Overall RPE upper body (0-10): ') or 0)
    rpe_lb = float(input('  Overall RPE lower body (0-10): ') or 0)
    rpe_core = input('  Overall RPE core (0-10, optional): ').strip()
    rpe_core = float(rpe_core) if rpe_core else None
    dur_actual = int(input('  Actual duration (min): ') or 0)
    notes = input('  Session notes: ').strip()
    return {
        'session_id': f"{session_dt.strftime('%Y%m%d_%H%M%S')}_{plan_id or 'unspecified'}",
        'session_datetime_utc': session_dt,
        'plan_id': plan_id,
        'wellness_light': wellness,
        'overall_rpe_upper_body': rpe_ub,
        'overall_rpe_lower_body': rpe_lb,
        'overall_rpe_core': rpe_core,
        'session_duration_actual_min': dur_actual,
        'session_notes': notes
    }


def prompt_exercises(lib_df):
    entries = []
    print('\nEnter exercises. Leave name blank to finish.')
    idx = 1
    while True:
        name = input(f'  Exercise #{idx} name: ').strip()
        if not name:
            break
        if name not in lib_df['exercise_name'].values:
            print(f"    Warning: '{name}' not in library.")
        set_num = int(input('    Set number: ') or 1)
        reps_actual = int(input('    Reps actual: ') or 0)
        weight_actual = float(input('    Weight kg actual: ') or 0)
        duration_s = input('    Duration s actual (if timed, else blank): ').strip()
        duration_s = int(duration_s) if duration_s else None
        distance_m = input('    Distance m actual (if applicable): ').strip()
        distance_m = float(distance_m) if distance_m else None
        rpe_set = float(input('    RPE for set (0-10): ') or 0)
        rir_set = input('    RIR for set (optional): ').strip()
        rir_set = int(rir_set) if rir_set else None
        notes_set = input('    Set notes: ').strip()
        entry = {
            'log_id': str(uuid.uuid4()),
            'exercise_name': name,
            'set_number': set_num,
            'reps_actual': reps_actual,
            'weight_kg_actual': weight_actual,
            'duration_s_actual': duration_s,
            'distance_m_actual': distance_m,
            'rpe_set': rpe_set,
            'rir_set': rir_set,
            'set_notes': notes_set
        }
        entries.append(entry)
        idx += 1
    return entries


def main():
    parser = argparse.ArgumentParser(description='Log a strength session interactively.')
    parser.add_argument('--session_datetime_utc', type=str, help='Session start in UTC (ISO format)')
    parser.add_argument('--plan_id', type=str, help='Optional plan identifier')
    args = parser.parse_args()

    default_dt = args.session_datetime_utc or datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    session_info = prompt_session_info(default_dt)
    lib_df = pd.read_csv(LIB_PATH)
    exercises = prompt_exercises(lib_df)

    # Prepare DataFrames
    sessions_df = pd.DataFrame([session_info])
    for col in ['session_id']:
        sessions_df[col] = sessions_df[col].astype(str)
    exercises_df = pd.DataFrame(exercises)
    exercises_df['session_id'] = session_info['session_id']

    # Append or create Parquet files
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if SESSIONS_PATH.exists():
        old = pd.read_parquet(SESSIONS_PATH)
        sessions_df = pd.concat([old, sessions_df], ignore_index=True)
    sessions_df.to_parquet(SESSIONS_PATH, index=False)

    if EXERCISES_PATH.exists():
        old_ex = pd.read_parquet(EXERCISES_PATH)
        exercises_df = pd.concat([old_ex, exercises_df], ignore_index=True)
    exercises_df.to_parquet(EXERCISES_PATH, index=False)

    print('\nLogged session to:', SESSIONS_PATH)
    print('Logged exercises to:', EXERCISES_PATH)


if __name__ == '__main__':
    main()
