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
    # wellness light must be empty or one of Green/Amber/Red
    while True:
        wellness = safe_input('  Wellness light [Green/Amber/Red]: ', '').strip()
        if not wellness or wellness.lower() in ('green', 'amber', 'red'):
            break
        print("  Please enter 'Green', 'Amber', or 'Red'")
    wellness = wellness.capitalize() if wellness else None

    # overall RPE upper body (0–10)
    while True:
        rpe_ub_str = safe_input('  Overall RPE upper body (0-10): ', '0')
        try:
            rpe_ub = float(rpe_ub_str)
            if 0 <= rpe_ub <= 10:
                break
            print("  RPE must be between 0 and 10")
        except ValueError:
            print("  Please enter a valid number")

    # overall RPE lower body (0–10)
    while True:
        rpe_lb_str = safe_input('  Overall RPE lower body (0-10): ', '0')
        try:
            rpe_lb = float(rpe_lb_str)
            if 0 <= rpe_lb <= 10:
                break
            print("  RPE must be between 0 and 10")
        except ValueError:
            print("  Please enter a valid number")

    # optional core RPE (0–10)
    rpe_core = safe_input('  Overall RPE core (0-10, optional): ', '').strip()
    if rpe_core:
        try:
            rpe_core = float(rpe_core)
            if not (0 <= rpe_core <= 10):
                print("  RPE core outside valid range (0-10), setting to null")
                rpe_core = None
        except ValueError:
            print("  Invalid RPE core value, setting to null")
            rpe_core = None
    else:
        rpe_core = None
    # planned duration (min)
    dur_planned = int(safe_input('  Planned duration (min): ', '0'))
    dur_actual = int(safe_input('  Actual duration (min): ', '0'))
    # environment temperature (°C)
    env_temp_str = safe_input('  Environment temperature (°C): ', '').strip()
    env_temp = None
    if env_temp_str:
        try:
            env_temp = float(env_temp_str)
        except ValueError:
            print('  Invalid temperature, setting to null')
            env_temp = None
    # location type
    location = safe_input('  Location type: ', '').strip()
    # video captured
    video = safe_input('  Video captured (true/false): ', 'false').lower() in ('true', 't', 'yes', 'y', '1')
    notes = safe_input('  Session notes: ', '').strip()
    return {
        'session_id': f"{session_dt.strftime('%Y%m%d_%H%M%S')}_{plan_id or 'unspecified'}",
        'session_datetime_utc': session_dt,
        'plan_id': plan_id,
        'wellness_light': wellness,
        'overall_rpe_upper_body': rpe_ub,
        'overall_rpe_lower_body': rpe_lb,
        'overall_rpe_core': rpe_core,
        'session_duration_planned_min': dur_planned,
        'session_duration_actual_min': dur_actual,
        'environment_temp_c': env_temp,
        'location_type': location,
        'video_captured': video,
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
        # Build set of canonical names and aliases (case-insensitive)
        canon = set(lib_df['exercise_name'].str.lower())
        if 'exercise_alias' in lib_df.columns:
            lib_df['exercise_alias'].fillna('', inplace=True)
            aliases = set()
            for a in lib_df['exercise_alias']:
                if isinstance(a, str):
                    for alias in a.split(';'):
                        if alias.strip():
                            aliases.add(alias.strip().lower())
            valid = canon.union(aliases)
            if name.lower() not in valid:
                print(f"    Warning: '{name}' not in library.")
                continue_anyway = safe_input("    Continue anyway? (y/n): ", "n").strip().lower()
                if continue_anyway != 'y':
                    continue
        else:
            if name.lower() not in set(lib_df['exercise_name'].str.lower()):
                print(f"    Warning: '{name}' not in library.")
                continue_anyway = safe_input("    Continue anyway? (y/n): ", "n").strip().lower()
                if continue_anyway != 'y':
                    continue
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
    try:
        lib_df = pd.read_csv(LIB_PATH)
        exercises = prompt_exercises(lib_df)
    except FileNotFoundError:
        print(f"Error: Exercise library not found at {LIB_PATH}")
        print("Please ensure the library file exists before logging a session.")
        return
    except Exception as e:
        print(f"Error loading exercise library: {e}")
        return

    # Prepare DataFrames
    sessions_df = pd.DataFrame([session_info])
    for col in ['session_id']:
        sessions_df[col] = sessions_df[col].astype(str)
    exercises_df = pd.DataFrame(exercises)
    exercises_df['session_id'] = session_info['session_id']

    # Append or create Parquet files with error handling
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if SESSIONS_PATH.exists():
            old = pd.read_parquet(SESSIONS_PATH)
            # Check for schema compatibility
            missing_cols = set(sessions_df.columns) - set(old.columns)
            if missing_cols:
                print(f"Warning: New session data contains columns not in existing file: {missing_cols}")
                print("Adding these columns to the existing data with null values.")
                for col in missing_cols:
                    old[col] = None
            sessions_df = pd.concat([old, sessions_df], ignore_index=True)
        sessions_df.to_parquet(SESSIONS_PATH, index=False)

        if EXERCISES_PATH.exists():
            old_ex = pd.read_parquet(EXERCISES_PATH)
            # Check for schema compatibility
            missing_cols = set(exercises_df.columns) - set(old_ex.columns)
            if missing_cols:
                print(f"Warning: New exercise data contains columns not in existing file: {missing_cols}")
                print("Adding these columns to the existing data with null values.")
                for col in missing_cols:
                    old_ex[col] = None
            exercises_df = pd.concat([old_ex, exercises_df], ignore_index=True)
        exercises_df.to_parquet(EXERCISES_PATH, index=False)

        print('\nLogged session to:', SESSIONS_PATH)
        print('Logged exercises to:', EXERCISES_PATH)
    except Exception as e:
        print(f"Error saving data: {e}")


if __name__ == '__main__':
    main()
