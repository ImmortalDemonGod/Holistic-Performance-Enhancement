import fitdecode
import gpxpy
import pandas as pd
import numpy as np
from haversine import haversine, Unit
import argparse

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
        print(f"Error decoding FIT file '{file_path}': {e}")
        return None
    except FileNotFoundError:
        print(f"Error: FIT file not found at '{file_path}'")
        return None
    if not data_records:
        print(f"No record messages found in '{file_path}'.")
        return None
    df = pd.DataFrame(data_records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').set_index('timestamp')
    return df

def parse_gpx(file_path):
    """Parses a GPX file using gpxpy and extracts track point data as a DataFrame."""
    points_data = []
    gpx = None
    try:
        with open(file_path, 'r', encoding='utf-8') as gpx_file:
            gpx = gpxpy.parse(gpx_file)
    except gpxpy.gpx.GPXXMLSyntaxException as e:
        print(f"Error parsing GPX XML in '{file_path}': {e}")
        return None
    except FileNotFoundError:
        print(f"Error: GPX file not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while parsing '{file_path}': {e}")
        return None
    if not gpx or not gpx.tracks:
        print(f"No tracks found in GPX file '{file_path}'.")
        return None
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                point_data = {
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'elevation': point.elevation,
                    'time': point.time,
                    'heart_rate': None,
                    'cadence': None,
                }
                if point.extensions:
                    for ext_element in point.extensions:
                        ns_gpxtpx = "http://www.garmin.com/xmlschemas/TrackPointExtension/v1"
                        hr_elem = ext_element.find(f'{{{ns_gpxtpx}}}hr')
                        if hr_elem is not None and hr_elem.text:
                            point_data['heart_rate'] = int(hr_elem.text)
                        cad_elem = ext_element.find(f'{{{ns_gpxtpx}}}cad')
                        if cad_elem is not None and cad_elem.text:
                            point_data['cadence'] = int(cad_elem.text)
                points_data.append(point_data)
    if not points_data:
        print(f"No track points extracted from '{file_path}'.")
        return None
    df = pd.DataFrame(points_data)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.sort_values('time').set_index('time')
    # Patch: add distance and speed for GPX parses so downstream metrics work
    if df is not None and ('dt' not in df.columns or 'dist' not in df.columns or 'distance_segment_m' not in df.columns):
        df = add_distance_and_speed(df)
    # Ensure compatibility: add 'dist' as segment distance in meters, matching metrics.py
    if 'distance_segment_m' in df.columns and 'dist' not in df.columns:
        df['dist'] = df['distance_segment_m']
    # Ensure compatibility: add 'dt' as time delta in seconds, matching metrics.py
    if 'time_delta_s' in df.columns and 'dt' not in df.columns:
        df['dt'] = df['time_delta_s']
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return 0
    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)

def add_distance_and_speed(df):
    df = df.copy()
    df['lat_prev'] = df['latitude'].shift(1)
    df['lon_prev'] = df['longitude'].shift(1)
    df['distance_segment_m'] = df.apply(
        lambda row: haversine_distance(
            row['lat_prev'], row['lon_prev'], row['latitude'], row['longitude']
        ) if pd.notna(row['lat_prev']) else 0,
        axis=1
    )
    df['distance_cumulative_km'] = df['distance_segment_m'].cumsum() / 1000.0
    df['time_delta_s'] = df.index.to_series().diff().dt.total_seconds().fillna(0)
    df['speed_mps'] = df['distance_segment_m'] / df['time_delta_s'].replace(0, np.nan)
    df['pace_min_per_km'] = 16.6667 / df['speed_mps'].replace(0, np.nan)  # 1000/60 = 16.6667
    return df

def summarize_run(df, label):
    summary = {}
    if df is None or df.empty:
        print(f"No data for {label}")
        return summary
    summary['start_time'] = df.index[0]
    summary['end_time'] = df.index[-1]
    summary['duration'] = summary['end_time'] - summary['start_time']
    summary['total_distance_km'] = df['distance_cumulative_km'].iloc[-1] if 'distance_cumulative_km' in df else np.nan
    summary['avg_pace_min_per_km'] = df['pace_min_per_km'].mean() if 'pace_min_per_km' in df else np.nan
    summary['avg_hr'] = df['heart_rate'].mean() if 'heart_rate' in df else np.nan
    summary['max_hr'] = df['heart_rate'].max() if 'heart_rate' in df else np.nan
    summary['avg_cadence'] = df['cadence'].mean() if 'cadence' in df else np.nan
    summary['elevation_gain_m'] = df['elevation'].diff().clip(lower=0).sum() if 'elevation' in df else np.nan
    # --- Integrate metrics if GPX ---
    if 'pace_sec_km' in df.columns:
        try:
            from metrics import run_metrics
            metrics = run_metrics(df, threshold_hr=175, resting_hr=50)
            summary.update(metrics)
        except Exception as e:
            print(f"[metrics] Could not compute advanced metrics: {e}")
    print(f"\nSummary for {label}:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary

def main():
    parser = argparse.ArgumentParser(description='Parse a FIT or GPX file and output summary CSV.')
    parser.add_argument('--input', type=str, required=True, help='Path to input .fit or .gpx file')
    parser.add_argument('--output', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--planning_id', type=str, default='', help='ID from calendar CSV')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    if input_path.endswith('.fit'):
        print(f'Parsing FIT file: {input_path}')
        fit_df = parse_fit_with_fitdecode(input_path)
        if fit_df is not None:
            fit_df = add_distance_and_speed(fit_df)
            summarize_run(fit_df, 'FIT')
            # --- attach calendar linkage & extra units ---
            fit_df['planning_id'] = args.planning_id
            fit_df.to_csv(output_path)
    elif input_path.endswith('.gpx'):
        print(f'Parsing GPX file: {input_path}')
        gpx_df = parse_gpx(input_path)
        if gpx_df is not None:
            gpx_df = add_distance_and_speed(gpx_df)
            summarize_run(gpx_df, 'GPX')
            # --- attach calendar linkage & extra units ---
            gpx_df['planning_id'] = args.planning_id
            gpx_df.to_csv(output_path)
    else:
        print('Unsupported file type. Please provide a .fit or .gpx file.')

if __name__ == '__main__':
    main()
