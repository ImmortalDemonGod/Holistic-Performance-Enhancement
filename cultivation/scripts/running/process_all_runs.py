import os
import argparse
from pathlib import Path
import subprocess

# Use hardcoded absolute path for venv python
VENV_PYTHON = "/Users/tomriddle1/Holistic-Performance-Enhancement/.venv/bin/python"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / 'cultivation' / 'scripts' / 'running'

def get_run_files(raw_dir):
    """Return list of all .fit and .gpx files in the raw data directory."""
    return sorted([f for f in Path(raw_dir).glob('*') if f.suffix in ['.fit', '.gpx']])

def main():
    parser = argparse.ArgumentParser(description="Batch process all running data files.")
    parser.add_argument('--raw_dir', type=str, default=str(PROJECT_ROOT / 'data/raw'), help='Path to raw data directory')
    parser.add_argument('--processed_dir', type=str, default=str(PROJECT_ROOT / 'data/processed'), help='Path to processed data directory')
    parser.add_argument('--figures_dir', type=str, default=str(PROJECT_ROOT / 'outputs/figures'), help='Path to figures output directory')
    args = parser.parse_args()

    # Step 0: Auto-rename generic files before processing
    print("Checking for generic file names and auto-renaming if needed...")
    subprocess.run([
        VENV_PYTHON, str(SCRIPTS_DIR / 'auto_rename_raw_files.py'),
        '--raw_dir', args.raw_dir
    ], cwd=str(PROJECT_ROOT))

    run_files = get_run_files(args.raw_dir)
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
        ], cwd=str(PROJECT_ROOT))
        # 2. Run HR/pace analysis
        subprocess.run([
            VENV_PYTHON, str(SCRIPTS_DIR / 'analyze_hr_pace_distribution.py'),
            '--input', str(csv_out),
            '--figures_dir', args.figures_dir,
            '--prefix', base
        ], cwd=str(PROJECT_ROOT))
        # 3. Run advanced analysis
        subprocess.run([
            VENV_PYTHON, str(SCRIPTS_DIR / 'run_performance_analysis.py'),
            '--input', str(csv_out),
            '--figures_dir', args.figures_dir,
            '--prefix', base
        ], cwd=str(PROJECT_ROOT))

if __name__ == '__main__':
    main()
