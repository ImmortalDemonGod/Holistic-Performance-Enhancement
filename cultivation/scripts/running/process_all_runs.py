import os
import argparse
from pathlib import Path
import subprocess

def get_run_files(raw_dir):
    """Return list of all .fit and .gpx files in the raw data directory."""
    return sorted([f for f in Path(raw_dir).glob('*') if f.suffix in ['.fit', '.gpx']])

def main():
    parser = argparse.ArgumentParser(description="Batch process all running data files.")
    parser.add_argument('--raw_dir', type=str, default='../../data/raw', help='Path to raw data directory')
    parser.add_argument('--processed_dir', type=str, default='../../data/processed', help='Path to processed data directory')
    parser.add_argument('--figures_dir', type=str, default='../../outputs/figures', help='Path to figures output directory')
    args = parser.parse_args()

    # Step 0: Auto-rename generic files before processing
    print("Checking for generic file names and auto-renaming if needed...")
    subprocess.run([
        'python3', 'auto_rename_raw_files.py',
        '--raw_dir', args.raw_dir
    ], cwd=os.path.dirname(__file__))

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
            'python3', 'parse_run_files.py',
            '--input', str(run_file),
            '--output', str(csv_out)
        ], cwd=os.path.dirname(__file__))
        # 2. Run HR/pace analysis
        subprocess.run([
            'python3', 'analyze_hr_pace_distribution.py',
            '--input', str(csv_out),
            '--figures_dir', args.figures_dir,
            '--prefix', base
        ], cwd=os.path.dirname(__file__))
        # 3. Run advanced analysis
        subprocess.run([
            'python3', 'run_performance_analysis.py',
            '--input', str(csv_out),
            '--figures_dir', args.figures_dir,
            '--prefix', base
        ], cwd=os.path.dirname(__file__))

if __name__ == '__main__':
    main()
