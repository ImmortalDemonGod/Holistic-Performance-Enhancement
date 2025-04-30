import sys
import os
from pathlib import Path
import argparse
import gpxpy
import gpxpy.gpx
from fitparse import FitFile
from datetime import datetime, timedelta, timezone
import bisect

# --- Helper functions ---
def parse_fit_hr(fit_path):
    """Extract timestamped heart rate from FIT file as a list of (timestamp, hr) tuples."""
    fitfile = FitFile(fit_path)
    hr_data = []
    for record in fitfile.get_messages('record'):
        ts = None
        hr = None
        for d in record:
            if d.name == 'timestamp':
                ts = d.value
            elif d.name == 'heart_rate':
                hr = d.value
        if ts and hr is not None:
            hr_data.append((ts, hr))
    hr_data.sort()
    return hr_data

def parse_gpx_trackpoints(gpx_path):
    """Return list of (trackpoint, timestamp) for all trackpoints in GPX file."""
    with open(gpx_path, 'r') as f:
        gpx = gpxpy.parse(f)
    points = []
    for trk in gpx.tracks:
        for seg in trk.segments:
            for pt in seg.points:
                points.append((pt, pt.time))
    return gpx, points

def ensure_utc(dt):
    if dt is None:
        return None
    try:
        # Remove noisy debug print
        if dt.tzinfo is None:
            # print(f"[DEBUG] Making naive datetime offset-aware: {dt} (type: {type(dt)})")
            if 'timezone' in globals() and hasattr(globals()['timezone'], 'utc'):
                return dt.replace(tzinfo=timezone.utc)
            from datetime import timezone as dt_timezone, timedelta
            return dt.replace(tzinfo=dt_timezone(timedelta(0)))
        return dt.astimezone(timezone.utc)
    except AttributeError as e:
        print(f"[ERROR] Could not set tzinfo to UTC for {dt} (type: {type(dt)}): {e}")
        print("[ERROR] This may indicate a broken datetime import or a shadowed datetime module.")
        raise

def assign_hr_to_gpx(gpx, points, hr_data):
    """Assign closest HR value from hr_data to each GPX trackpoint."""
    hr_times = [ensure_utc(t) for t, _ in hr_data]
    hr_values = [hr for _, hr in hr_data]
    for pt, ts in points:
        ts = ensure_utc(ts)
        if ts is None:
            continue
        # Find closest HR sample
        idx = bisect.bisect_left(hr_times, ts)
        if idx == 0:
            hr = hr_values[0]
        elif idx == len(hr_times):
            hr = hr_values[-1]
        else:
            before = hr_times[idx-1]
            after = hr_times[idx]
            # Pick closer
            if abs((ts - before).total_seconds()) <= abs((after - ts).total_seconds()):
                hr = hr_values[idx-1]
            else:
                hr = hr_values[idx]
        pt.extensions = [e for e in pt.extensions if 'hr' not in str(e)]  # Remove old HR
        # Add new HR extension
        import lxml.etree as ET
        ns = 'http://www.garmin.com/xmlschemas/TrackPointExtension/v1'
        ext = ET.Element('{%s}TrackPointExtension' % ns)
        hr_elem = ET.SubElement(ext, '{%s}hr' % ns)
        hr_elem.text = str(hr)
        pt.extensions.append(ext)
    return gpx

def write_gpx(gpx, out_path):
    with open(out_path, 'w') as f:
        f.write(gpx.to_xml())

# --- Main script ---
def main():
    parser = argparse.ArgumentParser(description="Override GPX HR with FIT HR data.")
    parser.add_argument('--gpx', required=True, help='Input GPX file')
    parser.add_argument('--fit', required=True, help='Input FIT file with accurate HR')
    parser.add_argument('--output', required=True, help='Output GPX file')
    args = parser.parse_args()

    print(f"Reading HR from FIT: {args.fit}")
    hr_data = parse_fit_hr(args.fit)
    if not hr_data:
        print("No HR data found in FIT file. Exiting.")
        sys.exit(1)

    print(f"Reading GPX: {args.gpx}")
    gpx, points = parse_gpx_trackpoints(args.gpx)
    print(f"Assigning HR to GPX trackpoints...")
    gpx = assign_hr_to_gpx(gpx, points, hr_data)
    print(f"Writing modified GPX to {args.output}")
    write_gpx(gpx, args.output)
    print("Done.")

if __name__ == '__main__':
    main()
