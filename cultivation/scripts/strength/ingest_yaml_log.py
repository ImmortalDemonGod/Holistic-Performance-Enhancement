#!/usr/bin/env python3
"""
Ingest a YAML workout log into Parquet session and exercise logs.
"""
import argparse
import yaml  # type: ignore
from convert_markdown_to_yaml import parse_markdown
import pandas as pd  # type: ignore
from pathlib import Path

# Paths
dir_base = Path(__file__).resolve().parents[2]
processed_dir = dir_base / 'data' / 'strength' / 'processed'
sessions_path = processed_dir / 'strength_sessions.parquet'
exercises_path = processed_dir / 'strength_exercises_log.parquet'
lib_path = processed_dir / 'exercise_library.csv'


def main():
    parser = argparse.ArgumentParser(description='Ingest YAML workout log to Parquet')
    parser.add_argument('input_file', help='Path to the Markdown or YAML log file')
    args = parser.parse_args()

    # Load YAML or Markdown
    in_path = Path(args.input_file)
    if in_path.suffix.lower() == '.md':
        data = parse_markdown(str(in_path))
    else:
        with open(in_path) as f:
            data = yaml.safe_load(f)

    # Skip template or files without session_id
    if 'session_id' not in data or not data.get('session_id'):
        print(f"Skipping {args.input_file}: no session_id found.")
        return
    # -- Schema validation: session fields
    required_session_keys = [
        'session_id', 'session_datetime_utc', 'plan_id', 'wellness_light',
        'overall_rpe_upper_body', 'overall_rpe_lower_body', 'overall_rpe_core',
        'session_duration_planned_min', 'session_duration_actual_min',
        'environment_temp_c', 'location_type', 'video_captured', 'session_notes'
    ]
    for key in required_session_keys:
        if key not in data:
            raise KeyError(f'Missing required session field: {key}')

    # -- Validate exercise names against library (case-insensitive, including aliases)
    lib_df = pd.read_csv(lib_path)
    # Build set of canonical names and aliases
    canon = set(lib_df['exercise_name'].str.lower())
    aliases = set()
    lib_df['exercise_alias'].fillna('', inplace=True)
    for a in lib_df['exercise_alias']:
        for alias in a.split(';'):
            if alias.strip():
                aliases.add(alias.strip().lower())
    valid = canon.union(aliases)
    # Check each exercise name lowercased
    ex_names = [ex.get('exercise_name','').strip() for ex in data.get('exercises', [])]
    unknown = {n for n in ex_names if n.lower() not in valid}
    if unknown:
        raise ValueError(f'Unknown exercise names in log: {unknown}')

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
        # Remove any existing records for this session to avoid duplicates
        old = old[old['session_id'] != session['session_id']]
        sess_df = pd.concat([old, sess_df], ignore_index=True)
    sess_df.to_parquet(sessions_path, index=False)

    # Exercises
    if not ex_df.empty:
        if exercises_path.exists():
            old_ex = pd.read_parquet(exercises_path)
            # Remove existing exercise rows for this session
            old_ex = old_ex[old_ex['session_id'] != session['session_id']]
            ex_df = pd.concat([old_ex, ex_df], ignore_index=True)
        ex_df.to_parquet(exercises_path, index=False)

    print(f'Successfully ingested {args.input_file}')
    print(f'- Sessions logged to: {sessions_path}')
    print(f'- Exercises logged to: {exercises_path}')


if __name__ == '__main__':
    main()
