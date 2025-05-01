import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from weather_utils import fetch_weather_open_meteo
from metrics import load_personal_zones, compute_training_zones, run_metrics

def time_in_zone(df, zone_col='zone_hr'):
    df = df.copy()
    df['time_delta_s'] = df.index.to_series().diff().dt.total_seconds().fillna(0)
    zone_times = df.groupby(zone_col)['time_delta_s'].sum()
    total_time = zone_times.sum()
    zone_pct = zone_times / total_time * 100
    return pd.DataFrame({'seconds': zone_times, 'percent': zone_pct})

def generate_recommendations(results):
    recs = []
    # Time in high zones
    if results['time_in_zone_hr'].loc['Z4 (Threshold)', 'percent'] > 30 or results['time_in_zone_hr'].loc['Z5 (VO2max)', 'percent'] > 10:
        recs.append("Consider more aerobic base training to balance high-intensity work.")
    # HR drift
    if results['hr_drift']['hr_drift_pct'] > 5:
        recs.append("Significant HR drift detected—focus on aerobic endurance and pacing.")
    # Pacing
    if results['pacing']['strategy'] == 'positive':
        recs.append("Try to avoid slowing down in the second half—aim for even or negative splits.")
    return recs

def detect_strides(df, pace_threshold=4.5, min_stride_duration=8, max_stride_duration=40, cadence_threshold=80):
    df = df.copy()
    stride_candidate = (df['pace_min_per_km'] < pace_threshold) & (df['cadence'] > cadence_threshold)
    df['stride'] = False
    in_stride = False
    stride_start_idx = None
    for idx in df.index:
        if stride_candidate.loc[idx]:
            if not in_stride:
                in_stride = True
                stride_start_idx = idx
        else:
            if in_stride:
                in_stride = False
                stride_end_idx = idx
                start_pos = df.index.get_loc(stride_start_idx)
                end_pos = df.index.get_loc(stride_end_idx)
                stride_duration = (df.iloc[end_pos-1].name - df.iloc[start_pos].name).total_seconds()
                if min_stride_duration <= stride_duration <= max_stride_duration:
                    df.loc[stride_start_idx:df.iloc[end_pos-1].name, 'stride'] = True
                stride_start_idx = None
    if in_stride and stride_start_idx is not None:
        stride_duration = (df.iloc[-1].name - df.loc[stride_start_idx].name).total_seconds()
        if min_stride_duration <= stride_duration <= max_stride_duration:
            df.loc[stride_start_idx:df.iloc[-1].name, 'stride'] = True
    return df

def compute_hr_drift(df):
    # Divide run into two halves by time
    midpoint = df.index[0] + (df.index[-1] - df.index[0]) / 2
    first_half = df[df.index <= midpoint]
    second_half = df[df.index > midpoint]
    hr1 = first_half['heart_rate'].mean()
    hr2 = second_half['heart_rate'].mean()
    drift = (hr2 - hr1) / hr1 * 100 if hr1 else np.nan
    return {'first_half_hr': hr1, 'second_half_hr': hr2, 'hr_drift_pct': drift}

def pacing_strategy(df):
    # Split by distance
    total_dist = df['distance_cumulative_km'].iloc[-1]
    halfway = total_dist / 2
    first_half = df[df['distance_cumulative_km'] <= halfway]
    second_half = df[df['distance_cumulative_km'] > halfway]
    pace1 = first_half['pace_min_per_km'].mean()
    pace2 = second_half['pace_min_per_km'].mean()
    if pace2 < pace1:
        strategy = 'negative'
    elif pace2 > pace1:
        strategy = 'positive'
    else:
        strategy = 'even'
    return {'first_half_pace': pace1, 'second_half_pace': pace2, 'strategy': strategy}

def main():
    parser = argparse.ArgumentParser(description='Advanced run performance analysis.')
    parser.add_argument('--input', type=str, required=True, help='Path to input summary CSV')
    parser.add_argument('--figures_dir', type=str, required=True, help='Directory to save figures')
    parser.add_argument('--prefix', type=str, required=True, help='Prefix for output files (e.g., date_activity)')
    args = parser.parse_args()

    df = pd.read_csv(args.input, index_col=0, parse_dates=True)
    zones = load_personal_zones()
    zone_hr, zone_pace, zone_effective = compute_training_zones(df['heart_rate'], df['pace_min_per_km'], zones)
    df['zone_hr'] = zone_hr
    df['zone_pace'] = zone_pace
    df['zone_effective'] = zone_effective
    df = detect_strides(df)
    tiz_hr = time_in_zone(df, zone_col='zone_hr')
    tiz_pace = time_in_zone(df, zone_col='zone_pace')
    tiz_effective = time_in_zone(df, zone_col='zone_effective')
    hr_drift = compute_hr_drift(df)
    pacing = pacing_strategy(df)
    results = {'time_in_zone_hr': tiz_hr, 'time_in_zone_pace': tiz_pace, 'time_in_zone_effective': tiz_effective, 'hr_drift': hr_drift, 'pacing': pacing}
    recs = generate_recommendations(results)
    print("\n--- Training Zone Time (%) ---")
    print(tiz_hr)
    print("\n--- Training Zone Time (%) ---")
    print(tiz_pace)
    print("\n--- Training Zone Time (%) ---")
    print(tiz_effective)
    print("\n--- HR Drift ---")
    print(hr_drift)
    print("\n--- Pacing Strategy ---")
    print(pacing)
    print("\n--- Recommendations ---")
    for rec in recs:
        print(f"- {rec}")
    # Determine week/run subdirectory for figures
    date_part = args.prefix.split('_')[0]
    week = pd.to_datetime(date_part).isocalendar().week
    img_dir = f"{args.figures_dir}/week{week}/{args.prefix}/images"
    txt_dir = f"{args.figures_dir}/week{week}/{args.prefix}/txt"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    # Plot time in zones
    tiz_hr['percent'].plot(kind='bar', color='skyblue', title='Time in HR Zone (%)')
    plt.ylabel('% of Run Time')
    plt.tight_layout()
    plt.savefig(f"{img_dir}/time_in_hr_zone.png")
    plt.close()
    tiz_pace['percent'].plot(kind='bar', color='skyblue', title='Time in Pace Zone (%)')
    plt.ylabel('% of Run Time')
    plt.tight_layout()
    plt.savefig(f"{img_dir}/time_in_pace_zone.png")
    plt.close()
    tiz_effective['percent'].plot(kind='bar', color='skyblue', title='Time in Effective Zone (%)')
    plt.ylabel('% of Run Time')
    plt.tight_layout()
    plt.savefig(f"{img_dir}/time_in_effective_zone.png")
    plt.close()
    # Save textual representation of time in zones
    tiz_hr.to_csv(f"{txt_dir}/time_in_hr_zone.txt", sep='\t')
    tiz_pace.to_csv(f"{txt_dir}/time_in_pace_zone.txt", sep='\t')
    tiz_effective.to_csv(f"{txt_dir}/time_in_effective_zone.txt", sep='\t')
    # Plot HR drift
    df['heart_rate'].plot(label='Heart Rate', alpha=0.7)
    plt.axvline(df.index[len(df)//2], color='red', linestyle='--', label='Midpoint')
    plt.title('Heart Rate Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{img_dir}/hr_over_time_drift.png")
    plt.close()
    # Save textual representation of HR drift
    with open(f"{txt_dir}/hr_over_time_drift.txt", "w") as f:
        f.write("Heart Rate Drift Analysis:\n")
        f.write(str(hr_drift))
        f.write("\n")
    # Plot pacing
    df['pace_min_per_km'].plot(label='Pace (min/km)', alpha=0.7)
    plt.axvline(df.index[len(df)//2], color='red', linestyle='--', label='Midpoint')
    plt.title('Pace Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{img_dir}/pace_over_time.png")
    plt.close()
    # Save textual representation of pacing
    with open(f"{txt_dir}/pace_over_time.txt", "w") as f:
        f.write("Pacing Strategy Analysis:\n")
        f.write(str(pacing))
        f.write("\n")
    # Stride summary
    stride_groups = df[df['stride']].groupby((df['stride'] != df['stride'].shift()).cumsum())
    stride_summary_lines = [f"Stride segments detected: {stride_groups.ngroups}"]
    for i, (_, group) in enumerate(stride_groups):
        stride_summary_lines.append(
            f"Stride {i+1}: {group.index[0]} to {group.index[-1]}, duration: {(group.index[-1]-group.index[0]).total_seconds():.1f}s, avg pace: {group['pace_min_per_km'].mean():.2f}, avg HR: {group['heart_rate'].mean():.1f}")
    with open(f"{txt_dir}/stride_summary.txt", "w") as f:
        f.write("\n".join(stride_summary_lines) + "\n")
    # Weather fetch
    lat = df['latitude'].iloc[0] if 'latitude' in df else None
    lon = df['longitude'].iloc[0] if 'longitude' in df else None
    weather, offset = fetch_weather_open_meteo(lat, lon, df.index[0])
    if weather and 'hourly' in weather and weather['hourly']['temperature_2m']:
        idx = 0
        if 'time' in weather['hourly']:
            try:
                times = pd.to_datetime(weather['hourly']['time'])
                idx = (np.abs(times - df.index[0])).argmin()
            except Exception:
                pass
        temp = weather['hourly']['temperature_2m'][idx]
        app_temp = weather['hourly'].get('apparent_temperature', [None]*len(weather['hourly']['temperature_2m']))[idx]
        with open(f"{txt_dir}/weather.txt", "w") as f:
            f.write(f"Temperature: {temp}, Apparent: {app_temp}\n")
    else:
        marker_path = os.path.join(txt_dir, "weather_failed.marker")
        with open(marker_path, "w") as mf:
            mf.write(f"Weather fetch failed for run at {df.index[0]} (lat={lat}, lon={lon}) on {pd.Timestamp.now()}\n")
    # Advanced metrics
    try:
        adv_metrics = run_metrics(df, threshold_hr=175, resting_hr=50)
        with open(f"{txt_dir}/advanced_metrics.txt", "w") as f:
            f.write(str(adv_metrics) + "\n")
    except Exception as e:
        with open(f"{txt_dir}/advanced_metrics.txt", "w") as f:
            f.write(f"Failed to compute advanced metrics: {e}\n")
    # --- Save run-level summary as text ---
    summary_lines = [
        f"Run Summary:",
        f"  Start time: {df.index[0]}",
        f"  End time: {df.index[-1]}",
        f"  Duration: {df.index[-1] - df.index[0]}",
        f"  Total distance (km): {df['distance_cumulative_km'].iloc[-1]:.2f}" if 'distance_cumulative_km' in df else "  Total distance (km): N/A",
        f"  Avg pace (min/km): {df['pace_min_per_km'].mean():.2f}" if 'pace_min_per_km' in df else "  Avg pace (min/km): N/A",
        f"  Avg HR: {df['heart_rate'].mean():.1f}" if 'heart_rate' in df else "  Avg HR: N/A",
        f"  Max HR: {df['heart_rate'].max():.1f}" if 'heart_rate' in df else "  Max HR: N/A",
        f"  Avg cadence: {df['cadence'].mean():.1f}" if 'cadence' in df else "  Avg cadence: N/A",
        f"  Elevation gain (m): {df['elevation'].diff().clip(lower=0).sum():.1f}" if 'elevation' in df else "  Elevation gain (m): N/A"
    ]
    # --- Add advanced metrics from metrics.py if available ---
    try:
        adv_metrics = run_metrics(df, threshold_hr=175, resting_hr=50)
        summary_lines.append("  --- Advanced Metrics (run_metrics) ---")
        for k, v in adv_metrics.items():
            summary_lines.append(f"    {k}: {v}")
    except Exception as e:
        summary_lines.append(f"  [metrics] Could not compute advanced metrics: {e}")
    # --- Add location and weather ---
    lat = df['latitude'].iloc[0] if 'latitude' in df else None
    lon = df['longitude'].iloc[0] if 'longitude' in df else None
    summary_lines.append(f"  Start location: ({lat:.5f}, {lon:.5f})" if lat is not None and lon is not None else "  Start location: N/A")
    weather, offset = fetch_weather_open_meteo(lat, lon, df.index[0])
    if weather and 'hourly' in weather and weather['hourly']['temperature_2m']:
        idx = 0
        if 'time' in weather['hourly']:
            try:
                times = pd.to_datetime(weather['hourly']['time'])
                idx = (np.abs(times - df.index[0])).argmin()
            except Exception:
                pass
        temp = weather['hourly']['temperature_2m'][idx]
        app_temp = weather['hourly'].get('apparent_temperature', [None]*len(weather['hourly']['temperature_2m']))[idx]
        precip = weather['hourly'].get('precipitation', [None]*len(weather['hourly']['temperature_2m']))[idx]
        humidity = weather['hourly'].get('relative_humidity_2m', [None]*len(weather['hourly']['temperature_2m']))[idx]
        wind = weather['hourly'].get('windspeed_10m', [None]*len(weather['hourly']['temperature_2m']))[idx]
        desc = weather['hourly'].get('weathercode', [None]*len(weather['hourly']['temperature_2m']))[idx]
        summary_lines.append("  Weather at start:")
        summary_lines.append(f"    Temperature: {temp} °C (feels like {app_temp} °C)")
        summary_lines.append(f"    Precipitation: {precip} mm")
        summary_lines.append(f"    Humidity: {humidity} %")
        summary_lines.append(f"    Wind speed: {wind} km/h")
        summary_lines.append(f"    Description: {desc}")
    else:
        summary_lines.append("  Weather at start: N/A [weather API unavailable or failed]")
        marker_path = os.path.join(txt_dir, "weather_failed.marker")
        with open(marker_path, "w") as mf:
            mf.write(f"Weather fetch failed for run at {df.index[0]} (lat={lat}, lon={lon}) on {pd.Timestamp.now()}\n")
    # --- Stride Segments ---
    stride_groups = df[df['stride']].groupby((df['stride'] != df['stride'].shift()).cumsum())
    stride_summary_lines = [f"Stride segments detected: {stride_groups.ngroups}"]
    for i, (_, group) in enumerate(stride_groups):
        stride_summary_lines.append(
            f"Stride {i+1}: {group.index[0]} to {group.index[-1]}, duration: {(group.index[-1]-group.index[0]).total_seconds():.1f}s, avg pace: {group['pace_min_per_km'].mean():.2f}, avg HR: {group['heart_rate'].mean():.1f}")
    summary_lines.append("  --- Stride Segments ---")
    summary_lines.extend(["    " + line for line in stride_summary_lines])
    with open(f"{txt_dir}/run_summary.txt", "w") as f:
        f.write("\n".join(summary_lines) + "\n")

if __name__ == '__main__':
    main()
