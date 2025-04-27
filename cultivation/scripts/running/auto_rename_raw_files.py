import os
from pathlib import Path
import fitdecode
import gpxpy
import re

def extract_datetime_str_from_fit(fit_path):
    try:
        with fitdecode.FitReader(fit_path) as fit:
            for frame in fit:
                if isinstance(frame, fitdecode.FitDataMessage) and frame.name == 'record':
                    if frame.has_field('timestamp'):
                        ts = frame.get_value('timestamp')
                        return ts.strftime('%Y%m%d_%H%M%S')
    except Exception:
        return None
    return None

def extract_datetime_str_from_gpx(gpx_path):
    try:
        with open(gpx_path, 'r', encoding='utf-8') as f:
            gpx = gpxpy.parse(f)
            for track in gpx.tracks:
                for segment in track.segments:
                    if segment.points:
                        ts = segment.points[0].time
                        return ts.strftime('%Y%m%d_%H%M%S')
    except Exception:
        return None
    return None

def has_date_prefix(name):
    import re
    # Accepts YYYYMMDD or YYYYMMDD_HHMMSS
    return bool(re.match(r'^\d{8}(?:_\d{6})?[_-]?.*', name))

def auto_rename_raw_files(raw_dir):
    for file in Path(raw_dir).glob('*'):
        if file.suffix not in ['.fit', '.gpx']:
            continue
        stem = file.stem
        if has_date_prefix(stem):
            continue
        # Try to extract datetime
        if file.suffix == '.fit':
            dt_str = extract_datetime_str_from_fit(file)
        else:
            dt_str = extract_datetime_str_from_gpx(file)
        if not dt_str:
            print(f"Could not extract datetime for {file.name}, skipping rename.")
            continue
        # Lowercase the rest of the name, replace spaces with _
        label = stem.lower().replace(' ', '_')
        new_name = f"{dt_str}_{label}{file.suffix}"
        new_path = file.parent / new_name
        if new_path.exists():
            print(f"Target name {new_name} already exists, skipping.")
            continue
        print(f"Renaming {file.name} -> {new_name}")
        file.rename(new_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Auto-rename raw run files to have datetime prefix.')
    parser.add_argument('--raw_dir', type=str, default='../../data/raw', help='Path to raw data directory')
    args = parser.parse_args()
    auto_rename_raw_files(args.raw_dir)
