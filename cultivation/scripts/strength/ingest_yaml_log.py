#!/usr/bin/env python3
"""
Ingest a YAML workout log into Parquet session and exercise logs.
"""
import argparse
import yaml
import pandas as pd
from pathlib import Path

# Paths
dir_base = Path(__file__).resolve().parents[2]
processed_dir = dir_base / 'data' / 'strength' / 'processed'
sessions_path = processed_dir / 'strength_sessions.parquet'
exercises_path = processed_dir / 'strength_exercises_log.parquet'


def main():
    parser = argparse.ArgumentParser(description='Ingest YAML workout log to Parquet')
    parser.add_argument('yaml_file', help='Path to the YAML log file')
    args = parser.parse_args()

    # Load YAML
    with open(args.yaml_file) as f:
        data = yaml.safe_load(f)

    # Session-level
    session = {
        'session_id': data.get('session_id'),
        'session_datetime_utc': pd.to_datetime(data.get('session_datetime_utc')),
        'plan_id': data.get('plan_id'),
        'wellness_light': data.get('wellness_light'),
        'overall_rpe_upper_body': data.get('overall_rpe_upper_body'),
        'overall_rpe_lower_body': data.get('overall_rpe_lower_body'),
        'overall_rpe_core': data.get('overall_rpe_core'),
        'session_duration_planned_min': data.get('session_duration_planned_min'),
        'session_duration_actual_min': data.get('session_duration_actual_min'),
        'environment_temp_c': data.get('environment_temp_c'),
        'location_type': data.get('location_type'),
        'video_captured': data.get('video_captured'),
        'session_notes': data.get('session_notes')
    }

    # Exercises-level
    exercises = data.get('exercises', [])
    ex_df = pd.DataFrame(exercises)
    if not ex_df.empty:
        ex_df['session_id'] = session['session_id']

    # Write Parquet (append or create)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Sessions
    sess_df = pd.DataFrame([session])
    if sessions_path.exists():
        old = pd.read_parquet(sessions_path)
        sess_df = pd.concat([old, sess_df], ignore_index=True)
    sess_df.to_parquet(sessions_path, index=False)

    # Exercises
    if not ex_df.empty:
        if exercises_path.exists():
            old_ex = pd.read_parquet(exercises_path)
            ex_df = pd.concat([old_ex, ex_df], ignore_index=True)
        ex_df.to_parquet(exercises_path, index=False)

    print(f'Successfully ingested {args.yaml_file}')
    print(f'- Sessions logged to: {sessions_path}')
    print(f'- Exercises logged to: {exercises_path}')


if __name__ == '__main__':
    main()
