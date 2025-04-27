import os
from pathlib import Path
import fitdecode
import gpxpy
import re

def extract_fit_start_time(fit_path):
    try:
        with fitdecode.FitReader(fit_path) as fit:
            for frame in fit:
                if isinstance(frame, fitdecode.FitDataMessage) and frame.name == 'record':
                    if frame.has_field('timestamp'):
                        ts = frame.get_value('timestamp')
                        return ts.strftime('%Y%m%d')
    except Exception:
        return None
    return None

def extract_gpx_start_time(gpx_path):
    try:
        with open(gpx_path, 'r', encoding='utf-8') as f:
            gpx = gpxpy.parse(f)
            for track in gpx.tracks:
                for segment in track.segments:
                    if segment.points:
                        ts = segment.points[0].time
                        return ts.strftime('%Y%m%d')
    except Exception:
        return None
    return None

def is_generic_name(name):
    # Heuristic: common export names or short/generic names
    return bool(re.match(r'^(activity|export|run|track|file|record|fitfile|gpxfile)[0-9_\-]*$', name, re.I)) or len(name) < 8

def auto_rename_raw_files(raw_dir):
    for file in Path(raw_dir).glob('*'):
        if file.suffix not in ['.fit', '.gpx']:
            continue
        stem = file.stem
        if not is_generic_name(stem):
            continue
        # Try to extract date
        if file.suffix == '.fit':
            date = extract_fit_start_time(file)
        else:
            date = extract_gpx_start_time(file)
        if not date:
            print(f"Could not extract date for {file.name}, skipping rename.")
            continue
        new_name = f"{date}_auto{file.suffix}"
        new_path = file.parent / new_name
        # Avoid overwriting
        if new_path.exists():
            print(f"Target name {new_name} already exists, skipping.")
            continue
        print(f"Renaming {file.name} -> {new_name}")
        file.rename(new_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Auto-rename generic raw run files to descriptive names.')
    parser.add_argument('--raw_dir', type=str, default='../../data/raw', help='Path to raw data directory')
    args = parser.parse_args()
    auto_rename_raw_files(args.raw_dir)
