import os
import argparse
from pathlib import Path
import subprocess
import pandas as pd
from metrics import parse_gpx, run_metrics

# Use hardcoded absolute path for venv python
VENV_PYTHON = "/Users/tomriddle1/Holistic-Performance-Enhancement/.venv/bin/python"
# Set PROJECT_ROOT to the top-level project directory (not cultivation)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / 'cultivation' / 'scripts' / 'running'

def get_run_files(raw_dir):
    """Return list of all .gpx files in the raw data directory (skip .fit files, which are only used for HR override)."""
    return sorted([f for f in Path(raw_dir).glob('*') if f.suffix == '.gpx'])

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
        if run_file.suffix == '.fit':
            csv_out = Path(args.processed_dir) / f"{base}_fit_summary.csv"
            input_file = run_file
        else:
            csv_out = Path(args.processed_dir) / f"{base}_gpx_summary.csv"
            gpx_path = run_file
            override_gpx = gpx_path.with_name(gpx_path.stem + "_hr_override.gpx")
            # Always prefer override file if present
            if override_gpx.exists():
                print(f"  [DEBUG] Using override GPX: {override_gpx}")
                input_file = override_gpx
            else:
                # --- Improved FIT/GPX matching logic ---
                def extract_base_name(stem):
                    # Remove leading timestamp and trailing _whoop_accurate_heart_rate if present
                    parts = stem.split('_')
                    # Heuristic: timestamp is always first 2 parts (date+time), extra suffix is last 4 parts for FIT
                    base = '_'.join(parts[2:-4]) if stem.endswith('whoop_accurate_heart_rate') else '_'.join(parts[2:])
                    return base
                gpx_base = extract_base_name(gpx_path.stem)
                fit_candidates = list(Path(args.raw_dir).glob(f"*whoop_accurate_heart_rate.fit"))
                fit_match = None
                for fit_file in fit_candidates:
                    fit_base = extract_base_name(fit_file.stem)
                    print(f"[DEBUG] gpx_base: {gpx_base}, fit_base: {fit_base}")
                    if gpx_base == fit_base:
                        fit_match = fit_file
                        break
                print(f"[DEBUG] fit_match: {fit_match}")
                if fit_match:
                    print(f"  [DEBUG] About to run override script: {gpx_path} -> {override_gpx} using {fit_match}")
                    try:
                        result = subprocess.run([
                            VENV_PYTHON, str(SCRIPTS_DIR / 'override_gpx_hr_with_fit.py'),
                            '--gpx', str(gpx_path),
                            '--fit', str(fit_match),
                            '--output', str(override_gpx)
                        ], cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=True)
                        print(f"  [DEBUG] HR override script output:\n{result.stdout}")
                        if result.stderr:
                            print(f"  [DEBUG] HR override script errors:\n{result.stderr}")
                    except subprocess.CalledProcessError as e:
                        print(f"  [ERROR] HR override script failed: {e}")
                        print(f"  [ERROR] Script stdout:\n{e.stdout}")
                        print(f"  [ERROR] Script stderr:\n{e.stderr}")
                    if override_gpx.exists():
                        input_file = override_gpx
                    else:
                        input_file = gpx_path
                else:
                    input_file = gpx_path
        # === Now check if already processed ===
        if csv_out.exists():
            print(f"  Skipping {run_file}: summary already exists at {csv_out}")
            continue
        # Run parse_run_files.py
        subprocess.run([
            VENV_PYTHON, str(SCRIPTS_DIR / 'parse_run_files.py'),
            '--input', str(input_file),
            '--output', str(csv_out)
        ], cwd=str(PROJECT_ROOT), check=True)

        # If GPX, also extract metrics using new module (now integrated into summary CSV)
        # No need to write separate JSON; metrics are included in summary CSV by parse_run_files.py

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
