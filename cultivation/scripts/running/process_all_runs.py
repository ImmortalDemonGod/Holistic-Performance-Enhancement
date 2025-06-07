import os
import argparse
from pathlib import Path
import subprocess
import pandas as pd
import sys
from datetime import date
# (imports removed – handled inside called scripts)

# Use sys.executable for subprocesses
PYTHON_EXEC = os.environ.get('VENV_PYTHON', sys.executable)
# Set PROJECT_ROOT to the top-level project directory (not cultivation)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / 'cultivation' / 'scripts' / 'running'
UTILITIES_DIR = PROJECT_ROOT / 'cultivation' / 'scripts' / 'utilities'

# --- Step 0.5: Ensure wellness data is up to date before processing runs ---
WELLNESS_PARQUET = PROJECT_ROOT / 'cultivation' / 'data' / 'daily_wellness.parquet'
SYNC_SCRIPT = UTILITIES_DIR / 'sync_habitdash.py'

def ensure_wellness_uptodate():
    today = date.today()
    try:
        df = pd.read_parquet(WELLNESS_PARQUET)
        df.index = pd.to_datetime(df.index).date
        if today not in df.index:
            print("[INFO] Today's wellness data missing, syncing Habit Dash...")
            subprocess.run(
    [PYTHON_EXEC, str(SYNC_SCRIPT)],
    cwd=str(PROJECT_ROOT),
    check=True,
    env={**os.environ, 'PYTHONPATH': str(PROJECT_ROOT / 'cultivation' / 'scripts')}
)
    except FileNotFoundError:
        print("[INFO] Wellness file not found, syncing Habit Dash...")
        subprocess.run(
    [PYTHON_EXEC, str(SYNC_SCRIPT)],
    cwd=str(PROJECT_ROOT),
    check=True,
    env={**os.environ, 'PYTHONPATH': str(PROJECT_ROOT / 'cultivation' / 'scripts')}
)


def get_run_files(raw_dir):
    """Return list of all .gpx files in the raw data directory, but skip base .gpx if an _hr_override.gpx exists."""
    raw_dir = Path(raw_dir)
    all_gpx = sorted([f for f in raw_dir.glob('*.gpx')])
    override_stems = {f.stem.replace('_hr_override', '') for f in all_gpx if f.stem.endswith('_hr_override')}
    filtered = []
    for f in all_gpx:
        # If this is a base file and an override exists, skip it
        if not f.stem.endswith('_hr_override') and f.stem in override_stems:
            continue
        filtered.append(f)
    return filtered

def main():
    parser = argparse.ArgumentParser(description="Batch process all running data files.")
    parser.add_argument('--raw_dir', type=str, default=str(PROJECT_ROOT / 'cultivation/data/raw'), help='Path to raw data directory')
    parser.add_argument('--processed_dir', type=str, default=str(PROJECT_ROOT / 'cultivation/data/processed'), help='Path to processed data directory')
    parser.add_argument('--figures_dir', type=str, default=str(PROJECT_ROOT / 'cultivation/outputs/figures'), help='Path to figures output directory')
    args = parser.parse_args()

    # Step 0: Auto-rename files before processing (integrated, no need for separate call)
    print("Checking for generic file names and auto-renaming if needed...")
    subprocess.run([
        PYTHON_EXEC, str(SCRIPTS_DIR / 'auto_rename_raw_files.py'),
        '--raw_dir', args.raw_dir
    ], cwd=str(PROJECT_ROOT), check=True)

    # Step 0.5: Ensure wellness data is up to date BEFORE processing runs
    ensure_wellness_uptodate()

    run_files = get_run_files(args.raw_dir)
    if not run_files:
        print("No .fit or .gpx files found in raw directory. Exiting.")
        return
    for run_file in run_files:
        base = run_file.stem  # e.g., 20250425_afternoon_run
        print(f"Processing {run_file}...")
        figures_dir = Path(args.figures_dir)
        try:
            date_part = base.split('_')[0]
            week = pd.to_datetime(date_part).isocalendar().week
            txt_dir = figures_dir / f"week{week}" / base / "txt"
            marker_path = txt_dir / "weather_failed.marker"
            # Decide input file (HR override or not)
            gpx_path = run_file
            override_gpx = gpx_path.with_name(gpx_path.stem + "_hr_override.gpx")
            if override_gpx.exists():
                print(f"  [DEBUG] Using override GPX: {override_gpx}")
                input_file = override_gpx
            else:
                input_file = gpx_path
            # Build csv_out once, before any checks
            if input_file.suffix == '.fit':
                csv_out = Path(args.processed_dir) / f"{base}_fit_summary.csv"
            else:
                csv_out = Path(args.processed_dir) / f"{base}_gpx_summary.csv"
            # ---- marker & skip logic use the correct path ----
            if marker_path.exists() and not csv_out.exists():
                print("  [INFO] Weather marker found but CSV missing. Processing anyway.")
            elif marker_path.exists() and csv_out.exists():
                print("  [INFO] Weather marker found but CSV exists. Explicitly reprocessing.")
        except Exception as e:
            print(f"  [WARN] Could not determine marker file for {base}: {e}")

        # === Now check if already processed ===
        if csv_out.exists():
            continue
        # Run parse_run_files.py
        # build PID like "2025w18-Tue"
        def build_planning_id(run_file):
            # Placeholder: extract from filename, e.g., "20250429_191120_baseox_wk1_tue_z2_strides_25min"
            import re
            from datetime import datetime
            stem = run_file.stem
            date_match = re.match(r"(\d{8})_", stem)
            day_map = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
            if date_match:
                date_str = date_match.group(1)
                dt = datetime.strptime(date_str, "%Y%m%d")
                week = dt.isocalendar().week
                year = dt.isocalendar().year
                dow = day_map[dt.weekday()]
                return f"{year}w{week:02d}-{dow}"
            return "unknown"
        pid = build_planning_id(run_file)
        subprocess.run([
            PYTHON_EXEC, str(SCRIPTS_DIR / 'parse_run_files.py'),
            '--input', str(input_file),
            '--output', str(csv_out),
            '--planning_id', pid,
            '--figures_dir', str(args.figures_dir),
            '--prefix', base
        ], cwd=str(PROJECT_ROOT), check=True)

        # If GPX, also extract metrics using new module (now integrated into summary CSV)
        # No need to write separate JSON; metrics are included in summary CSV by parse_run_files.py
        subprocess.run([
            PYTHON_EXEC, str(SCRIPTS_DIR / 'analyze_hr_pace_distribution.py'),
            '--input', str(csv_out),
            '--figures_dir', args.figures_dir,
            '--prefix', base
        ], cwd=str(PROJECT_ROOT), check=True)
        # 3. Run advanced analysis
        subprocess.run([
            PYTHON_EXEC, str(SCRIPTS_DIR / 'run_performance_analysis.py'),
            '--input', str(csv_out),
            '--figures_dir', args.figures_dir,
            '--prefix', base
        ], cwd=str(PROJECT_ROOT), check=True, env={**os.environ, 'PYTHONPATH': str(PROJECT_ROOT / 'cultivation' / 'scripts')})

    # === NEW: Weekly Comparison Step ===
    # Find all processed CSVs, group by ISO week, and if any week has 2+ runs, compare the two most recent
    processed_files = sorted(Path(args.processed_dir).glob('*_summary.csv'))
    week_to_files = {}
    for csv_path in processed_files:
        base = os.path.basename(csv_path)
        date_part = base.split('_')[0]
        week = pd.to_datetime(date_part).isocalendar().week
        if week not in week_to_files:
            week_to_files[week] = []
        week_to_files[week].append((csv_path, pd.to_datetime(date_part)))
    for week, files in week_to_files.items():
        if len(files) >= 2:
            # Sort by date, pick two most recent
            files_sorted = sorted(files, key=lambda x: x[1], reverse=True)
            run1, run2 = files_sorted[0][0], files_sorted[1][0]
            print(f"Comparing runs from week {week}: {os.path.basename(run1)} vs {os.path.basename(run2)}")
            subprocess.run([
                PYTHON_EXEC, str(SCRIPTS_DIR / 'compare_weekly_runs.py'),
                '--run1', str(run1),
                '--run2', str(run2),
                '--figures_dir', args.figures_dir
            ], cwd=str(PROJECT_ROOT), check=True)

if __name__ == '__main__':
    from datetime import date
    main()
