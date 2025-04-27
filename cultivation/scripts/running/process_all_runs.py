import os
import argparse
from pathlib import Path
import subprocess
import pandas as pd

# Use hardcoded absolute path for venv python
VENV_PYTHON = "/Users/tomriddle1/Holistic-Performance-Enhancement/.venv/bin/python"
# Set PROJECT_ROOT to the top-level project directory (not cultivation)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / 'cultivation' / 'scripts' / 'running'

def get_run_files(raw_dir):
    """Return list of all .fit and .gpx files in the raw data directory."""
    return sorted([f for f in Path(raw_dir).glob('*') if f.suffix in ['.fit', '.gpx']])

def main():
    parser = argparse.ArgumentParser(description="Batch process all running data files.")
    parser.add_argument('--raw_dir', type=str, default=str(PROJECT_ROOT / 'cultivation/data/raw'), help='Path to raw data directory')
    parser.add_argument('--processed_dir', type=str, default=str(PROJECT_ROOT / 'cultivation/data/processed'), help='Path to processed data directory')
    parser.add_argument('--figures_dir', type=str, default=str(PROJECT_ROOT / 'cultivation/outputs/figures'), help='Path to figures output directory')
    args = parser.parse_args()

    # Step 0: Auto-rename files before processing (integrated, no need for separate call)
    print("Checking for generic file names and auto-renaming if needed...")
    subprocess.run([
        VENV_PYTHON, str(SCRIPTS_DIR / 'auto_rename_raw_files.py'),
        '--raw_dir', args.raw_dir
    ], cwd=str(PROJECT_ROOT), check=True)

    run_files = get_run_files(args.raw_dir)
    if not run_files:
        print("No .fit or .gpx files found in raw directory. Exiting.")
        return
    for run_file in run_files:
        base = run_file.stem  # e.g., 20250425_afternoon_run
        print(f"Processing {run_file}...")
        # 1. Parse raw file to CSV summary
        if run_file.suffix == '.fit':
            csv_out = Path(args.processed_dir) / f"{base}_fit_summary.csv"
        else:
            csv_out = Path(args.processed_dir) / f"{base}_gpx_summary.csv"
        # Run parse_run_files.py
        subprocess.run([
            VENV_PYTHON, str(SCRIPTS_DIR / 'parse_run_files.py'),
            '--input', str(run_file),
            '--output', str(csv_out)
        ], cwd=str(PROJECT_ROOT), check=True)
        # 2. Run HR/pace analysis
        subprocess.run([
            VENV_PYTHON, str(SCRIPTS_DIR / 'analyze_hr_pace_distribution.py'),
            '--input', str(csv_out),
            '--figures_dir', args.figures_dir,
            '--prefix', base
        ], cwd=str(PROJECT_ROOT), check=True)
        # 3. Run advanced analysis
        subprocess.run([
            VENV_PYTHON, str(SCRIPTS_DIR / 'run_performance_analysis.py'),
            '--input', str(csv_out),
            '--figures_dir', args.figures_dir,
            '--prefix', base
        ], cwd=str(PROJECT_ROOT), check=True)

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
                VENV_PYTHON, str(SCRIPTS_DIR / 'compare_weekly_runs.py'),
                '--run1', str(run1),
                '--run2', str(run2),
                '--figures_dir', args.figures_dir
            ], cwd=str(PROJECT_ROOT), check=True)

if __name__ == '__main__':
    main()
