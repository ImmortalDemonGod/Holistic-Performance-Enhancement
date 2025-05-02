import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))
import fitdecode
# import gpxpy  # Removed unused import
import pandas as pd
import numpy as np
from haversine import haversine, Unit
import argparse
from cultivation.scripts.running.metrics import parse_gpx
from cultivation.scripts.running.walk_utils import (
    filter_gps_jitter,
    drop_short_segments,
    compute_time_weighted_pace,
    summarize_walk_segments,
    walk_block_segments
)
import os
import math

VERBOSE = False

# ---------- NEW: walk detection thresholds & function ----------
DEFAULT_WALK_PACE_THR = 9.5      # min·km⁻¹
DEFAULT_WALK_CAD_THR  = 140       # full-stride spm

def detect_walk(df,
                pace_col="pace_min_per_km",
                cad_col="cadence",  # Use full-stride cadence!
                pace_thr=DEFAULT_WALK_PACE_THR,
                cad_thr=DEFAULT_WALK_CAD_THR):
    """
    Returns True for rows classified as walking based on pace OR full-stride cadence.
    """
    pace_flag = df[pace_col] > pace_thr
    cad_flag  = df[cad_col]  < cad_thr
    return pace_flag | cad_flag

def parse_fit_with_fitdecode(file_path):
    """Parses a FIT file using fitdecode and extracts record data as a DataFrame."""
    data_records = []
    try:
        with fitdecode.FitReader(file_path) as fit:
            for frame in fit:
                if isinstance(frame, fitdecode.FitDataMessage):
                    if frame.name == 'record':
                        record_data = {'timestamp': None}
                        if frame.has_field('timestamp'):
                            record_data['timestamp'] = frame.get_value('timestamp')
                        for field_name in ['position_lat', 'position_long', 'altitude',
                                           'heart_rate', 'cadence', 'speed', 'distance',
                                           'temperature', 'power']:
                            if frame.has_field(field_name):
                                record_data[field_name] = frame.get_value(field_name)
                        # Convert coordinates
                        if 'position_lat' in record_data and record_data['position_lat'] is not None:
                            record_data['latitude'] = record_data.pop('position_lat') * (180.0 / 2**31)
                        if 'position_long' in record_data and record_data['position_long'] is not None:
                            record_data['longitude'] = record_data.pop('position_long') * (180.0 / 2**31)
                        if record_data['timestamp']:
                            data_records.append(record_data)
    except fitdecode.FitDecodeError as e:
        if VERBOSE: print(f"Error decoding FIT file '{file_path}': {e}")
        return None
    except FileNotFoundError:
        if VERBOSE: print(f"Error: FIT file not found at '{file_path}'")
        return None
    if not data_records:
        if VERBOSE: print(f"No record messages found in '{file_path}'.")
        return None
    df = pd.DataFrame(data_records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').set_index('timestamp')
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return 0
    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)

def add_distance_and_speed(df):
    # Alias lat/lon to latitude/longitude if needed
    if 'latitude' not in df.columns and 'lat' in df.columns:
        df = df.rename(columns={'lat': 'latitude'})
    if 'longitude' not in df.columns and 'lon' in df.columns:
        df = df.rename(columns={'lon': 'longitude'})
    # Ensure time is the index and is datetime-like
    if 'time' in df.columns:
        df = df.set_index('time')
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            if VERBOSE: print(f"[ERROR] Could not convert index to datetime: {e}")
            if VERBOSE: print(f"Index sample: {df.index[:5]}")
            raise
    required = ["latitude", "longitude"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        if VERBOSE: print(f"[ERROR] DataFrame is missing columns: {missing}")
        if VERBOSE: print(f"Columns present: {list(df.columns)}")
        if VERBOSE: print(f"DataFrame shape: {df.shape}")
        if VERBOSE: print(df.head())
        raise KeyError(f"Missing required columns: {missing} in add_distance_and_speed")
    df = df.copy()
    df['lat_prev'] = df['latitude'].shift(1)
    df['lon_prev'] = df['longitude'].shift(1)
    df['distance_segment_m'] = df.apply(
        lambda row: haversine_distance(row['latitude'], row['longitude'], row['lat_prev'], row['lon_prev']), axis=1)
    df['dist'] = df['distance_segment_m']
    # Use shift by row, not by freq (works for irregular time series)
    df['time_prev'] = df.index.to_series().shift(1)
    df['time_delta_s'] = (df.index - df['time_prev']).dt.total_seconds().fillna(0)
    df['dt'] = df['time_delta_s']
    df['speed_mps'] = df['distance_segment_m'] / df['time_delta_s']
    df['speed_mps'] = df['speed_mps'].bfill()
    # --- Always create distance_cumulative_km and pace_min_per_km ---
    if 'distance_segment_m' in df.columns:
        df['distance_cumulative_km'] = df['distance_segment_m'].cumsum() / 1000.0
    else:
        df['distance_cumulative_km'] = np.nan
    if 'speed_mps' in df.columns:
        df['pace_min_per_km'] = 16.6667 / df['speed_mps'].replace(0, np.nan)
    else:
        df['pace_min_per_km'] = np.nan
    # --- Diagnostics if all NaN ---
    if df['distance_cumulative_km'].isna().all() or df['pace_min_per_km'].isna().all():
        if VERBOSE: print("[WARN] All distance or pace values are NaN after computation!")
        if VERBOSE: print(df[['latitude', 'longitude', 'distance_segment_m', 'speed_mps']].head())
    return df

def summarize_run(df, label):
    summary = {}
    if df is None or df.empty:
        if VERBOSE: print(f"No data for {label}")
        return summary
    summary['start_time'] = df.index[0]
    summary['end_time'] = df.index[-1]
    summary['duration'] = summary['end_time'] - summary['start_time']
    summary['total_distance_km'] = df['distance_cumulative_km'].iloc[-1] if 'distance_cumulative_km' in df else np.nan
    if 'pace_min_per_km' in df:
        summary['avg_pace_min_per_km'] = df['pace_min_per_km'].mean()
    else:
        summary['avg_pace_min_per_km'] = np.nan
    summary['avg_hr'] = df['heart_rate'].mean() if 'heart_rate' in df else np.nan
    summary['max_hr'] = df['heart_rate'].max() if 'heart_rate' in df else np.nan
    summary['avg_cadence'] = df['cadence'].mean() if 'cadence' in df else np.nan
    # --- Robust elevation gain calculation ---
    elev_col = None
    for candidate in ['elevation', 'ele']:
        if candidate in df.columns:
            elev_col = candidate
            break
    if elev_col:
        summary['elevation_gain_m'] = df[elev_col].diff().clip(lower=0).sum()
    else:
        summary['elevation_gain_m'] = 'N/A'
    # --- Integrate metrics if GPX ---
    if 'pace_sec_km' in df.columns:
        try:
            from metrics import run_metrics
            metrics = run_metrics(df, threshold_hr=175, resting_hr=50)
            summary.update(metrics)
        except Exception as e:
            if VERBOSE: print(f"[metrics] Could not compute advanced metrics: {e}")
    if VERBOSE: print(f"\nSummary for {label}:")
    for k, v in summary.items():
        if VERBOSE: print(f"  {k}: {v}")
    return summary

def main():
    global VERBOSE
    parser = argparse.ArgumentParser(description='Parse a FIT or GPX file and output summary CSV.')
    parser.add_argument('--input', type=str, required=True, help='Path to input .fit or .gpx file')
    parser.add_argument('--output', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--planning_id', type=str, default='', help='ID from calendar CSV')
    parser.add_argument('--walk_pace_thr', type=float, default=DEFAULT_WALK_PACE_THR, help='min/km threshold for walk')
    parser.add_argument('--walk_cad_thr',  type=float, default=DEFAULT_WALK_CAD_THR,    help='full-stride spm threshold for walk')
    parser.add_argument('--verbose', action='store_true', help='Enable debug output')
    parser.add_argument('--figures_dir', type=str, default='', help='Directory for figures')
    parser.add_argument('--prefix', type=str, default='', help='Prefix for output files')
    args = parser.parse_args()
    VERBOSE = args.verbose

    input_path = args.input
    output_path = args.output
    if input_path.endswith('.fit'):
        if VERBOSE: print(f'Parsing FIT file: {input_path}')
        fit_df = parse_fit_with_fitdecode(input_path)
        if fit_df is not None:
            fit_df['planning_id'] = args.planning_id
            fit_df.to_csv(output_path)
    elif input_path.endswith('.gpx'):
        if VERBOSE: print(f'Parsing GPX file: {input_path}')
        gpx_df = parse_gpx(input_path)
        if gpx_df is not None:
            gpx_df = add_distance_and_speed(gpx_df)
            if 'hr' in gpx_df.columns and 'heart_rate' not in gpx_df.columns:
                gpx_df['heart_rate'] = gpx_df['hr']
            # --- Auto-scale cadence if needed ---
            if 'cadence' in gpx_df.columns:
                # If 90% of cadence values are below 100, assume single-leg and double
                if gpx_df['cadence'].quantile(0.90) < 100:
                    print("[DIAG] Detected likely single-leg cadence, doubling to get full-stride cadence.")
                    gpx_df['cadence'] = gpx_df['cadence'] * 2
            print("[DIAG] Head of gpx_df before walk detection:\n", gpx_df.head())
            print("[DIAG] Cadence stats:", gpx_df['cadence'].describe() if 'cadence' in gpx_df.columns else 'cadence not in columns')
            print("[DIAG] Pace stats:", gpx_df['pace_min_per_km'].describe() if 'pace_min_per_km' in gpx_df.columns else 'pace_min_per_km not in columns')
            gpx_df['is_walk'] = detect_walk(
                gpx_df,
                pace_thr=args.walk_pace_thr,
                cad_thr=args.walk_cad_thr
            )
            print(f"[DIAG] Rows detected as walking: {gpx_df['is_walk'].sum()} / {len(gpx_df)}")
            # --- PATCH: Always write a full, unfiltered CSV for session-level summary ---
            full_out_path = output_path.replace('_gpx_summary.csv', '_gpx_full.csv')
            gpx_df['planning_id'] = args.planning_id
            gpx_df.to_csv(full_out_path)
            # --- Use full, unfiltered df for session summary ---
            session_summary = summarize_run(gpx_df, label='session')
            # --- Use filtered core_df for steady-state/advanced metrics ---
            core_df = gpx_df.loc[~gpx_df['is_walk']].copy()
            walk_df = gpx_df.loc[gpx_df['is_walk']].copy()
            # write run-only CSV
            core_df['planning_id'] = args.planning_id
            core_df.to_csv(output_path)
            # --- ENFORCE: Always build absolute walk summary path under figures_dir/weekXX/prefix/txt/ ---
            if hasattr(args, 'figures_dir') and hasattr(args, 'prefix') and args.figures_dir and args.prefix:
                week = pd.to_datetime(walk_df.index[0]).isocalendar().week if not walk_df.empty else 'unknownweek'
                run_dir = Path(args.figures_dir) / f"week{week}" / args.prefix
                txt_dir = run_dir / 'txt'
                txt_dir.mkdir(parents=True, exist_ok=True)
                # --- Use concise, generic names for summary files ---
                walk_txt = str(txt_dir / 'walk.txt')
                session_summary_txt = str(txt_dir / 'session_full_summary.txt')
                run_only_summary_txt = str(txt_dir / 'run_only_summary.txt')
            else:
                raise RuntimeError("figures_dir and prefix must be provided to determine walk summary output directory.")
            print(f"[DIAG] output_path: {output_path}")
            print(f"[DIAG] walk_txt: {walk_txt}")
            print(f"[DIAG] session_summary_txt: {session_summary_txt}")
            print(f"[DIAG] run_only_summary_txt: {run_only_summary_txt}")
            print(f"[DIAG] CWD: {os.getcwd()}")
            print(f"[DIAG] walk_df shape before writing: {walk_df.shape}")
            # --- Write session-level summary (Strava-like, all data) ---
            with open(session_summary_txt, 'w') as f:
                f.write("Run Session Summary (Strava-like):\n")
                for k, v in session_summary.items():
                    f.write(f"  {k}: {v}\n")
            # --- Write run-only summary (filtered, steady-state) ---
            run_only_summary = summarize_run(core_df, label='run_only')
            with open(run_only_summary_txt, 'w') as f:
                f.write("Run Summary (Run-Only, Filtered):\n")
                for k, v in run_only_summary.items():
                    f.write(f"  {k}: {v}\n")
            if walk_df.empty:
                print(f"[WARN] walk_df is empty, not writing walk summary file: {walk_txt}")
            else:
                print(f"[DIAG] Writing walk summary to: {walk_txt}")
                cols = ['timestamp','distance_cumulative_km','pace_min_per_km','cadence','heart_rate']
                walk_out = walk_df.copy()
                idx_name = walk_out.index.name
                if 'timestamp' not in walk_out.columns:
                    walk_out = walk_out.reset_index()
                    if 'index' in walk_out.columns and 'timestamp' not in walk_out.columns:
                        walk_out = walk_out.rename(columns={'index': 'timestamp'})
                    elif idx_name and idx_name != 'timestamp' and idx_name in walk_out.columns:
                        walk_out = walk_out.rename(columns={idx_name: 'timestamp'})
                walk_out = walk_out[[c for c in cols if c in walk_out.columns]]
                # --- PATCH 1: Seed first row of distance_cumulative_km with 0.0 if missing or blank ---
                if 'distance_cumulative_km' in walk_out.columns and len(walk_out) > 0:
                    walk_out.iloc[0, walk_out.columns.get_loc('distance_cumulative_km')] = 0.0
                walk_out = filter_gps_jitter(walk_out, 'pace_min_per_km', 'cadence', args.walk_cad_thr)
                walk_out = walk_out[[c for c in cols if c in walk_out.columns]]
                walk_out.to_csv(walk_txt, index=False)
                print(f"[DIAG] Finished writing walk summary: {walk_txt}")
                print(f"[DIAG] File exists after write? {os.path.exists(walk_txt)}")
                # --- STEP 1: Write walk_segments.csv (contiguous walk blocks) ---
                segments = walk_block_segments(gpx_df, 'is_walk', 'pace_min_per_km', 'cadence', args.walk_cad_thr)
                # --- DEBUG: Print all walk segments for diagnosis ---
                print(f"[DEBUG] Number of walk segments: {len(segments)}")
                for i, s in enumerate(segments):
                    print(f"[DEBUG] Walk segment {i}: start={s.get('start_ts')}, end={s.get('end_ts')}, dur_s={s['dur_s']}, dist_km={s.get('dist_km')}, tag={s.get('tag')}, note={s.get('note')}")
                seg_df = pd.DataFrame(segments)
                csv_dir = Path(txt_dir).parent / 'csv'
                csv_dir.mkdir(exist_ok=True)
                seg_csv = str(csv_dir / 'walk_segments.csv')
                seg_df.to_csv(seg_csv, index=False)
                print(f"[DIAG] Wrote walk_segments.csv: {seg_csv}")
                # --- STEP 1.5: Walk/run/session integrity assertions ---
                # Compute total session time and distance
                total_session_s = (gpx_df.index[-1] - gpx_df.index[0]).total_seconds() if len(gpx_df) > 1 else 1
                total_session_km = gpx_df['distance_cumulative_km'].iloc[-1] if 'distance_cumulative_km' in gpx_df else np.nan
                run_time = np.nansum(core_df['dt']) if 'dt' in core_df else 0
                run_dist = np.nansum(core_df['dist'])/1000 if 'dist' in core_df else 0
                total_walk_time = sum([s['dur_s'] for s in segments if s['dur_s'] > 0])
                total_walk_dist = sum([s['dist_km'] for s in segments if isinstance(s['dist_km'], (int, float)) and not math.isnan(s['dist_km'])])
                # Assertion 1: walk_time + run_time ≈ session_time (±3s)
                print(f"[DEBUG] total_session_s: {total_session_s}")
                print(f"[DEBUG] total_walk_time: {total_walk_time}")
                print(f"[DEBUG] run_time: {run_time}")
                print(f"[DEBUG] total_walk_time + run_time: {total_walk_time + run_time}")
                print(f"[DEBUG] Difference: {total_session_s - (total_walk_time + run_time)}")
                # --- Deep dive into time sums and row counts ---
                if 'dt' in gpx_df:
                    print(f"[DEBUG] Sum of gpx_df['dt']: {gpx_df['dt'].sum()}")
                if 'is_walk' in gpx_df and 'dt' in gpx_df:
                    print(f"[DEBUG] Sum of dt where walk: {gpx_df.loc[gpx_df['is_walk'] == True, 'dt'].sum()}")
                    print(f"[DEBUG] Sum of dt where run: {gpx_df.loc[gpx_df['is_walk'] == False, 'dt'].sum()}")
                print(f"[DEBUG] Number of rows in gpx_df: {len(gpx_df)}")
                print(f"[DEBUG] Number of rows in core_df: {len(core_df)}")
                # --- Analyze unclassified rows ---
                if 'is_walk' in gpx_df:
                    walk_mask = gpx_df['is_walk'] == True
                    run_mask = gpx_df['is_walk'] == False
                    neither_mask = ~(walk_mask | run_mask)
                    n_neither = neither_mask.sum()
                    neither_df = gpx_df[neither_mask]
                    print(f"[DEBUG] # rows neither walk nor run: {n_neither}")
                    if n_neither > 0:
                        print("[DEBUG] Sample of unclassified rows:")
                        print(neither_df[['pace_min_per_km', 'cadence', 'dt']].head(10))
                        print("[DEBUG] Full is_walk value counts:")
                        print(gpx_df['is_walk'].value_counts(dropna=False))
                        print("[DEBUG] Unclassified rows stats:")
                        print(neither_df.describe(include='all'))
                # Assertion 2: walk_dist + run_dist ≤ session_dist + 0.05 km
                if (total_walk_dist + run_dist) > (total_session_km + 0.05):
                    with open(str(txt_dir / 'sanity-fail.marker'), 'w') as marker:
                        marker.write('Distance accounting mismatch')
                    print('[SANITY FAIL] Distance accounting mismatch')
                    return
                # Assertion 3: every segment with dur_s ≥ 60 has dist_km > 0
                for s in segments:
                    if s['dur_s'] >= 60 and (not isinstance(s['dist_km'], (int, float)) or s['dist_km'] <= 0):
                        with open(str(txt_dir / 'sanity-fail.marker'), 'w') as marker:
                            marker.write('Segment with dur_s ≥ 60 has dist_km ≤ 0')
                        print('[SANITY FAIL] Segment with dur_s ≥ 60 has dist_km ≤ 0')
                        return
                # --- STEP 2: Write walk_summary.txt (human-readable roll-up) ---
                summary = summarize_walk_segments(segments)
                total_walk_time = summary['total_walk_time']
                total_walk_dist = summary['total_walk_dist']
                avg_pace = summary['avg_pace']
                avg_hr = summary['avg_hr']
                max_hr = summary['max_hr']
                avg_cad = summary['avg_cad']
                mins, secs = divmod(int(total_walk_time), 60)
                session_secs = (gpx_df.index[-1] - gpx_df.index[0]).total_seconds() if len(gpx_df) > 1 else 1
                walk_pct = 100 * total_walk_time / session_secs if session_secs > 0 else 0
                kcal = int(round(0.95 * 70 * total_walk_dist)) if not np.isnan(total_walk_dist) else ''
                summary_lines = [
                    "Walk Summary:",
                    f"  Segments detected: {len(summary['valid_segments'])}",
                    f"  Total walk time: {mins} min {secs:02d} s  ({walk_pct:.1f} % of session)",
                    f"  Total walk distance (km): {total_walk_dist:.2f}",
                    f"  Avg walk pace (min/km): {avg_pace:.1f}",
                    f"  Avg walk HR (bpm): {avg_hr:.1f}",
                    f"  Max walk HR (bpm): {max_hr}",
                    f"  Avg walk cadence (spm): {avg_cad:.0f}",
                    f"  Energy cost est. (kcal): {kcal}",
                    "  Notes:",
                    "    • (Auto-generated. Edit as needed.)"
                ]
                summary_txt = str(txt_dir / 'walk_summary.txt')
                with open(summary_txt, 'w') as fh:
                    fh.write('\n'.join(summary_lines) + '\n')
                print(f"[DIAG] Wrote walk_summary.txt: {summary_txt}")
                # --- STEP 3: Write walk_over_time.txt (only walk-flagged rows, time-series) ---
                if len(walk_df) > 0:
                    over_time_cols = [
                        'timestamp',
                        'distance_cumulative_km',
                        'pace_min_per_km',
                        'cadence',
                        'heart_rate'
                    ]
                    # Defensive: ensure all required columns are present before writing walk_over_time
                    missing_cols = [col for col in over_time_cols if col not in walk_df.reset_index().columns]
                    if missing_cols:
                        print(f"[WARN] Skipping walk_over_time.txt: columns missing: {missing_cols}")
                    else:
                        walk_over_time = walk_df.reset_index()[over_time_cols]
                        walk_over_time_txt = str(txt_dir / 'walk_over_time.txt')
                        walk_over_time.to_csv(walk_over_time_txt, sep=',', float_format='%.2f', index=False)
                        print(f"[DIAG] Wrote walk_over_time.txt: {walk_over_time_txt}")
                # --- STEP 4: Write walk_pace_distribution.txt and walk_hr_distribution.txt ---
                if len(walk_df) > 0:
                    # Pace distribution (min/km bins)
                    pace_bins = np.arange(5, 20.5, 0.5)
                    pace_hist, pace_edges = np.histogram(walk_df['pace_min_per_km'].dropna(), bins=pace_bins)
                    pace_dist = pd.DataFrame({
                        'pace_bin_min_per_km': pace_edges[:-1],
                        'count': pace_hist
                    })
                    pace_dist_txt = str(txt_dir / 'walk_pace_distribution.txt')
                    pace_dist.to_csv(pace_dist_txt, sep='\t', float_format='%.2f', index=False)
                    print(f"[DIAG] Wrote walk_pace_distribution.txt: {pace_dist_txt}")
                    # HR distribution (bpm bins)
                    hr_bins = np.arange(60, 201, 5)
                    hr_hist, hr_edges = np.histogram(walk_df['heart_rate'].dropna(), bins=hr_bins)
                    hr_dist = pd.DataFrame({
                        'hr_bin_bpm': hr_edges[:-1],
                        'count': hr_hist
                    })
                    hr_dist_txt = str(txt_dir / 'walk_hr_distribution.txt')
                    hr_dist.to_csv(hr_dist_txt, sep='\t', float_format='%.0f', index=False)
                    print(f"[DIAG] Wrote walk_hr_distribution.txt: {hr_dist_txt}")
    else:
        if VERBOSE: print('Unsupported file type. Please provide a .fit or .gpx file.')

if __name__ == '__main__':
    main()
