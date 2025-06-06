import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt
import argparse
import os
import sys
from datetime import timedelta
import json
import datetime

# Add the script directory to the path for direct script execution
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import the required modules
try:
    # First try direct imports (for script execution)
    from weather_utils import fetch_weather_open_meteo, get_weather_description  # type: ignore
    from metrics import load_personal_zones, compute_training_zones, run_metrics, lower_z2_bpm  # type: ignore
except ImportError:
    # Fall back to full module path (for module imports)
    from cultivation.scripts.running.weather_utils import fetch_weather_open_meteo, get_weather_description  # type: ignore
    from cultivation.scripts.running.metrics import load_personal_zones, compute_training_zones, run_metrics, lower_z2_bpm  # type: ignore

def time_in_zone(df, zone_col='zone_hr'):
    """Calculate time spent in each zone.

    Args:
        df: DataFrame with time index
        zone_col: Column containing zone labels

    Returns:
        DataFrame with seconds and percent columns
    """
    df = df.copy()
    df['time_delta_s'] = df.index.to_series().diff().dt.total_seconds().fillna(0)
    zone_times = df.groupby(zone_col)['time_delta_s'].sum()
    total_time = zone_times.sum()
    zone_pct = zone_times / total_time * 100
    return pd.DataFrame({'seconds': zone_times, 'percent': zone_pct})

def calculate_fatigue_kpi_zones(df):
    """Calculate fatigue KPI zones based on heart rate data.

    Fatigue KPI zones are used to monitor training load and fatigue.
    - 'Recovery': HR < lower_z2_bpm
    - 'Aerobic': lower_z2_bpm <= HR <= 160 (Base-Ox Z2 ceiling)
    - 'Threshold': 160 < HR <= 175 (Base-Ox Z3 ceiling)
    - 'High Intensity': HR > 175

    Args:
        df: DataFrame with heart_rate column

    Returns:
        Series with fatigue KPI zone labels
    """
    if 'heart_rate' not in df.columns:
        return pd.Series(index=df.index)

    # Get the lower Z2 boundary
    lz2 = lower_z2_bpm()

    # Define fatigue KPI zones based on heart rate
    conditions = [
        df['heart_rate'] < lz2,
        (df['heart_rate'] >= lz2) & (df['heart_rate'] <= 160),
        (df['heart_rate'] > 160) & (df['heart_rate'] <= 175),
        df['heart_rate'] > 175
    ]
    choices = [
        'Recovery (< Z2)',
        'Aerobic (Z2)',
        'Threshold (Z3)',
        'High Intensity (Z4+)'
    ]

    return pd.Series(np.select(conditions, choices, default=None), index=df.index)

def generate_recommendations(results):
    recs = []
    # Time in high zones: safely retrieve percentages for zones
    tiz_hr = results['time_in_zone_hr']
    z4_pct = tiz_hr['percent'].get('Z4 (Threshold)', 0)
    z5_pct = tiz_hr['percent'].get('Z5 (VO2max)', 0)
    if z4_pct > 30 or z5_pct > 10:
        recs.append("Consider more aerobic base training to balance high-intensity work.")

    # Time in fatigue KPI zones
    tiz_fatigue = results['time_in_zone_fatigue_kpi']
    threshold_pct = tiz_fatigue['percent'].get('Threshold (Z3)', 0)
    high_intensity_pct = tiz_fatigue['percent'].get('High Intensity (Z4+)', 0)

    # Base-Ox phase recommendations
    if threshold_pct + high_intensity_pct > 20:
        recs.append("During Base-Ox phase, limit time above 160 bpm to less than 20% of total run time.")

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
    # ---- HR-drift filter: ignore warm-up below Z2 lower bound ----
    lz2 = lower_z2_bpm()
    core = df[df['heart_rate'] >= lz2]
    if core.empty:
        core = df
    # Divide run into two halves by time (on core segment)
    midpoint = core.index[0] + (core.index[-1] - core.index[0]) / 2
    first_half = core[core.index <= midpoint]
    second_half = core[core.index > midpoint]
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

    # --- Load wellness context if available ---
    wellness_context = None
    wellness_df = None
    VERBOSE = False
    try:
        wellness_path = os.path.join(os.path.dirname(__file__), '../../data/daily_wellness.parquet')
        wellness_path = os.path.abspath(wellness_path)
        if os.path.exists(wellness_path):
            wellness_df = pd.read_parquet(wellness_path)
            # We'll set the wellness context date after loading the main dataframe
            wellness_context = {}
        else:
            wellness_context = {}
    except Exception as e:
        wellness_context = {}
        if VERBOSE:
            print(f"[WARN] Could not load wellness data: {e}")

    df = pd.read_csv(args.input, index_col=0, parse_dates=True)
    # Now set the wellness context date if possible
    if wellness_df is not None and not wellness_df.empty:
        run_date = df.index[0].date() if hasattr(df.index[0], 'date') else pd.to_datetime(df.index[0]).date()
        # Ensure both index and lookup are datetime.date
        idx = [d if isinstance(d, datetime.date) and not isinstance(d, pd.Timestamp) else d.date() for d in wellness_df.index]
        run_date_dt = run_date if isinstance(run_date, datetime.date) and not isinstance(run_date, pd.Timestamp) else run_date.date()
        mask = [d <= run_date_dt for d in idx]
        if any(mask):
            closest = max([d for d in idx if d <= run_date_dt])
            # Use positional index since wellness_df.index may not be unique or sorted
            pos = idx.index(closest)
            wellness_context = wellness_df.iloc[pos].to_dict()
        else:
            wellness_context = {k: 'n/a' for k in wellness_df.columns}
            print(f"[WARN] No wellness data available on or before {run_date}.")

    zones = load_personal_zones()
    zone_hr, zone_pace, zone_effective = compute_training_zones(df['heart_rate'], df['pace_min_per_km'], zones)
    df['zone_hr'] = zone_hr
    df['zone_pace'] = zone_pace
    df['zone_effective'] = zone_effective

    # Calculate fatigue KPI zones
    df['zone_fatigue_kpi'] = calculate_fatigue_kpi_zones(df)

    df = detect_strides(df)
    tiz_hr = time_in_zone(df, zone_col='zone_hr')
    tiz_pace = time_in_zone(df, zone_col='zone_pace')
    tiz_effective = time_in_zone(df, zone_col='zone_effective')
    tiz_fatigue_kpi = time_in_zone(df, zone_col='zone_fatigue_kpi')

    hr_drift = compute_hr_drift(df)
    pacing = pacing_strategy(df)

    results = {
        'time_in_zone_hr': tiz_hr,
        'time_in_zone_pace': tiz_pace,
        'time_in_zone_effective': tiz_effective,
        'time_in_zone_fatigue_kpi': tiz_fatigue_kpi,
        'hr_drift': hr_drift,
        'pacing': pacing
    }
    recs = generate_recommendations(results)
    print("\n--- HR Zone Time (%) ---")
    print(tiz_hr)
    print("\n--- Pace Zone Time (%) ---")
    print(tiz_pace)
    print("\n--- Effective Zone Time (%) ---")
    print(tiz_effective)
    print("\n--- Fatigue KPI Zone Time (%) ---")
    print(tiz_fatigue_kpi)
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

    # Plot time in fatigue KPI zones
    tiz_fatigue_kpi['percent'].plot(kind='bar', color='skyblue', title='Time in Fatigue KPI Zone (%)')
    plt.ylabel('% of Run Time')
    plt.tight_layout()
    plt.savefig(f"{img_dir}/time_in_fatigue_kpi_zone.png")
    plt.close()
    # Save textual representation of time in zones
    tiz_hr.to_csv(f"{txt_dir}/time_in_hr_zone.txt", sep='\t')
    tiz_pace.to_csv(f"{txt_dir}/time_in_pace_zone.txt", sep='\t')
    tiz_effective.to_csv(f"{txt_dir}/time_in_effective_zone.txt", sep='\t')
    tiz_fatigue_kpi.to_csv(f"{txt_dir}/time_in_fatigue_kpi_zone.txt", sep='\t')
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
        f.write("hr_drift:\n")
        f.write(f"first_half_hr: {float(hr_drift['first_half_hr']):.2f}\n")
        f.write(f"second_half_hr: {float(hr_drift['second_half_hr']):.2f}\n")
        f.write(f"hr_drift_pct: {float(hr_drift['hr_drift_pct']):.2f}\n")
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
        # Write pacing strategy as JSON
        first = float(pacing['first_half_pace'])
        raw_second = pacing['second_half_pace']
        if isinstance(raw_second, (float, np.floating)) and np.isnan(raw_second):
            second = None
        else:
            second = float(raw_second)
        data = {
            "first_half_pace": first,
            "second_half_pace": second,
            "strategy": pacing['strategy']
        }
        json.dump(data, f, indent=2)
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
    weather, _ = fetch_weather_open_meteo(lat, lon, df.index[0])
    if (
        weather
        and 'hourly' in weather
        and 'temperature_2m' in weather['hourly']
        and weather['hourly']['temperature_2m'] is not None
        and len(weather['hourly']['temperature_2m']) > 0
    ):
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
        # Format weather metrics with N/A fallback
        temp_val = f"{temp:.1f}" if temp is not None else "N/A"
        app_temp_val = f"{app_temp:.1f}" if app_temp is not None else "N/A"
        precip_val = f"{precip:.1f}" if precip is not None else "N/A"
        humidity_val = f"{humidity}" if humidity is not None else "N/A"
        wind_val = f"{wind}" if wind is not None else "N/A"
        with open(f"{txt_dir}/weather.txt", "w") as f:
            f.write(f"Temperature: {temp} °C, Apparent: {app_temp} °C\n")
            f.write(f"Description: {get_weather_description(desc)}\n")
    else:
        marker_path = os.path.join(txt_dir, "weather_failed.marker")
        with open(marker_path, "w") as mf:
            mf.write(f"Weather fetch failed for run at {df.index[0]} (lat={lat}, lon={lon}) on {pd.Timestamp.now()}\n")
    # Advanced metrics
    adv_metrics = None
    adv_metrics_error = None
    try:
        # Ensure 'hr' column exists for metrics.py compatibility
        if 'heart_rate' in df.columns and 'hr' not in df.columns:
            df['hr'] = df['heart_rate']
        # Ensure 'pace_sec_km' exists for metrics.py compatibility
        if 'pace_sec_km' not in df.columns and 'pace_min_per_km' in df.columns:
            df['pace_sec_km'] = df['pace_min_per_km'] * 60
        adv_metrics = run_metrics(df, threshold_hr=175, resting_hr=50)
    except Exception as e:
        adv_metrics_error = str(e)
    # Write advanced metrics to file
    adv_metrics_lines = []
    if adv_metrics is not None:
        for k, v in adv_metrics.items():
            if k == 'zones_applied':
                adv_metrics_lines.append("    zones_applied:")
                try:
                    zones_dict = json.loads(v) if isinstance(v, str) else v
                    for zone, vals in zones_dict.items():
                        adv_metrics_lines.append(f"      {zone}: {vals}")
                except Exception as e:
                    adv_metrics_lines.append(f"      [Error decoding zones_applied: {e}]")
            else:
                adv_metrics_lines.append(f"    {k}: {v}")
    else:
        adv_metrics_lines.append(f"Failed to compute advanced metrics: {adv_metrics_error}")
    with open(f"{txt_dir}/advanced_metrics.txt", "w") as f:
        f.write("\n".join(adv_metrics_lines) + "\n")
    # --- Save run-level summary as text ---
    summary_lines = [
        "\nRun Summary:",
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
    summary_lines.append("")
    summary_lines.append("  --- Advanced Metrics (run_metrics) ---")
    if adv_metrics is not None:
        for k, v in adv_metrics.items():
            if k == 'zones_applied':
                summary_lines.append("    zones_applied:")
                try:
                    zones_dict = json.loads(v) if isinstance(v, str) else v
                    for zone, vals in zones_dict.items():
                        summary_lines.append(f"      {zone}: {vals}")
                except Exception as e:
                    summary_lines.append(f"      [Error decoding zones_applied: {e}]")
            else:
                summary_lines.append(f"    {k}: {v}")
    else:
        summary_lines.append(f"  [metrics] Could not compute advanced metrics: {adv_metrics_error}")
    summary_lines.append("")
    # Convert skin_temp_whoop and its comparison values from C to F before summary
    def _to_float(val):
        try:
            return float(val)
        except Exception:
            return None
    if 'skin_temperature_whoop' in wellness_context and wellness_context['skin_temperature_whoop'] is not None:
        val = _to_float(wellness_context['skin_temperature_whoop'])
        wellness_context['skin_temperature_whoop'] = val * 9/5 + 32 if val is not None else wellness_context['skin_temperature_whoop']
    if 'skin_temperature_whoop_1d' in wellness_context and wellness_context['skin_temperature_whoop_1d'] is not None:
        val = _to_float(wellness_context['skin_temperature_whoop_1d'])
        wellness_context['skin_temperature_whoop_1d'] = val * 9/5 + 32 if val is not None else wellness_context['skin_temperature_whoop_1d']
    if 'skin_temperature_whoop_7d' in wellness_context and wellness_context['skin_temperature_whoop_7d'] is not None:
        val = _to_float(wellness_context['skin_temperature_whoop_7d'])
        wellness_context['skin_temperature_whoop_7d'] = val * 9/5 + 32 if val is not None else wellness_context['skin_temperature_whoop_7d']

    # Format block
    def _fmt(val, unit, *, convert_s_to_h=False, convert_s_to_min=False):
        try:
            if val is None or val == '':
                return "n/a"
            if isinstance(val, str):
                val = float(val)
            if convert_s_to_h:
                val = val / 3600
            if convert_s_to_min:
                val = val / 60
            if unit == 'h' and val < 0.1:
                return f"{val*60:.1f} min"
            return f"{val:.1f} {unit}"
        except Exception:
            return str(val) if val is not None else "n/a"

    def _delta_str(today_val, prev_val, unit_conv=None, cap_pct=300, min_baseline=1e-2):
        try:
            if today_val is None or prev_val is None or prev_val == '' or today_val == '':
                return "n/a"
            t, p = today_val, prev_val
            if isinstance(t, str):
                t = float(t)
            if isinstance(p, str):
                p = float(p)
            if unit_conv:
                t = unit_conv(t)
                p = unit_conv(p)
            if abs(p) < min_baseline:
                return "--"
            delta = 100 * (t - p) / abs(p)
            if abs(delta) > cap_pct:
                return f"{cap_pct:+.0f}%+" if delta > 0 else f"-{cap_pct:.0f}%+"
            return f"{delta:+.1f}%"
        except Exception:
            return "n/a"

    # For each metric, add daily and weekly delta if possible
    metric_specs = [
        ("heart_rate_variability_whoop", "HRV (Whoop)", "ms", None),
        ("resting_heart_rate_whoop", "RHR (Whoop)", "bpm", None),
        ("resting_heart_rate_garmin", "RHR (Garmin)", "bpm", None),
        ("recovery_score_whoop", "Recovery Score (Whoop)", "%", None),
        ("sleep_score_whoop", "Sleep Score (Whoop)", "%", None),
        ("body_battery_garmin", "Body Battery (Garmin)", "%", None),
    if wellness_context:
        run_date = df.index[0].date() if hasattr(df.index[0], 'date') else pd.to_datetime(df.index[0]).date()
        summary_lines.append("\n--- Pre-Run Wellness Context (Data for {}) ---".format(run_date))
        for key, label, unit, unit_conv in metric_specs:
            today_val = wellness_context.get(key)
            # Daily delta
            prev_val = None
            week_val = None
            if key.endswith('_prev_day'):
                # Only daily delta makes sense
                prev_val = None
                week_val = None
            else:
                try:
                    prev_val = wellness_df.loc[run_date - timedelta(days=1)].get(key, None)
                except Exception:
                    prev_val = None
                try:
                    week_val = wellness_df.loc[run_date - timedelta(days=7)].get(key, None)
                except Exception:
                    week_val = None
            delta_1d = _delta_str(today_val, prev_val, unit_conv)
            delta_7d = _delta_str(today_val, week_val, unit_conv)
            # Format value
            disp_val = _fmt(today_val, unit, convert_s_to_h=(key=="sleep_total_whoop"), convert_s_to_min=(key=="total_activity_garmin"))
            if delta_1d != "n/a" or delta_7d != "n/a":
                summary_lines.append(f"  {label}: {disp_val} (Δ1d: {delta_1d}, Δ7d: {delta_7d})")
            else:
                summary_lines.append(f"  {label}: {disp_val}")
            # Insert separator ONLY after vo2max_garmin
            if key == "vo₂_max_garmin":
                summary_lines.append("  ---")
    else:
        summary_lines.append("\n--- Pre-Run Wellness Context: n/a ---")

    # --- Add location and weather ---
    lat = df['latitude'].iloc[0] if 'latitude' in df else None
    lon = df['longitude'].iloc[0] if 'longitude' in df else None
    summary_lines.append(f"  Start location: ({lat:.5f}, {lon:.5f})" if lat is not None and lon is not None else "  Start location: N/A")
    weather, _ = fetch_weather_open_meteo(lat, lon, df.index[0])
    if (
        weather
        and 'hourly' in weather
        and 'temperature_2m' in weather['hourly']
        and weather['hourly']['temperature_2m'] is not None
        and len(weather['hourly']['temperature_2m']) > 0
    ):
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
        # Format weather values with human-readable placeholders
        temp_val = f"{temp:.1f} °C" if temp is not None else "N/A"
        app_temp_val = f"{app_temp:.1f}" if app_temp is not None else "N/A"
        precip_val = f"{precip:.1f}" if precip is not None else "N/A"
        humidity_val = f"{humidity}" if humidity is not None else "N/A"
        wind_val = f"{wind}" if wind is not None else "N/A"
        summary_lines.append("  Weather at start:")
        summary_lines.append(f"    Temperature: {temp_val} (feels like {app_temp_val})")
        summary_lines.append(f"    Precipitation: {precip_val} mm")
        summary_lines.append(f"    Humidity: {humidity_val} %")
        summary_lines.append(f"    Wind speed: {wind_val} km/h")
        summary_lines.append(f"    Description: {get_weather_description(desc)}")
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
    # summary_lines.append("  --- Stride Segments ---")
    # summary_lines.extend(["    " + line for line in stride_summary_lines])
    with open(f"{txt_dir}/run_summary.txt", "w") as f:
        f.write("\n".join(summary_lines) + "\n")

if __name__ == '__main__':
    main()
