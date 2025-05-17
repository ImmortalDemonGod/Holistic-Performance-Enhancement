#!/usr/bin/env python3
"""
Interactive CLI to log a strength session into Parquet files.
"""
import argparse
import uuid
import pandas as pd  # type: ignore
from pathlib import Path
from datetime import datetime

def safe_input(prompt, default=None):
    try:
        val = input(prompt)
        return val or default
    except EOFError:
        return default

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / 'data' / 'strength' / 'processed'
SESSIONS_PATH = PROCESSED_DIR / 'strength_sessions.parquet'
EXERCISES_PATH = PROCESSED_DIR / 'strength_exercises_log.parquet'
LIB_PATH = PROCESSED_DIR / 'exercise_library.csv'


def prompt_session_info(default_dt, default_plan_id=None):
    print('Enter session-level information (press Enter to accept default):')
    dt_str = safe_input(f'  Session datetime UTC [{default_dt}]: ', default_dt)
    session_dt = pd.to_datetime(dt_str)
    plan_id = safe_input(f'  Plan ID [{default_plan_id}]: ', default_plan_id).strip() or None
    wellness = safe_input('  Wellness light [Green/Amber/Red]: ', '').strip()
    rpe_ub = float(safe_input('  Overall RPE upper body (0-10): ', '0'))
    rpe_lb = float(safe_input('  Overall RPE lower body (0-10): ', '0'))
    rpe_core = safe_input('  Overall RPE core (0-10, optional): ', '').strip()
    rpe_core = float(rpe_core) if rpe_core else None
    dur_actual = int(safe_input('  Actual duration (min): ', '0'))
    notes = safe_input('  Session notes: ', '').strip()
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
        name = safe_input(f'  Exercise #{idx} name: ', '').strip()
        if not name:
            break
        if name not in lib_df['exercise_name'].values:
            print(f"    Warning: '{name}' not in library.")
        set_num = int(safe_input('    Set number: ', '1'))
        reps_actual = int(safe_input('    Reps actual: ', '0'))
        weight_actual = float(safe_input('    Weight kg actual: ', '0'))
        duration_s = safe_input('    Duration s actual (if timed, else blank): ', '').strip()
        duration_s = int(duration_s) if duration_s else None
        distance_m = safe_input('    Distance m actual (if applicable): ', '').strip()
        distance_m = float(distance_m) if distance_m else None
        rpe_set = float(safe_input('    RPE for set (0-10): ', '0'))
        rir_set = safe_input('    RIR for set (optional): ', '').strip()
        rir_set = int(rir_set) if rir_set else None
        notes_set = safe_input('    Set notes: ', '').strip()
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
    session_info = prompt_session_info(default_dt, args.plan_id)
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
