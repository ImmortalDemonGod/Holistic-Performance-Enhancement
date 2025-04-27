import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def compute_training_zones(df, hrmax, zones=None):
    """
    Annotate DataFrame with heart rate zones.
    zones: list of (zone_name, lower_pct, upper_pct)
    """
    if zones is None:
        zones = [
            ("Z1 (Recovery)", 0.5, 0.6),
            ("Z2 (Endurance)", 0.6, 0.7),
            ("Z3 (Tempo)", 0.7, 0.8),
            ("Z4 (Threshold)", 0.8, 0.9),
            ("Z5 (VO2max)", 0.9, 1.0)
        ]
    def hr_zone(hr):
        if np.isnan(hr):
            return None
        pct = hr / hrmax
        for name, low, high in zones:
            if low <= pct < high:
                return name
        if pct >= zones[-1][2]:
            return zones[-1][0]
        return None
    df = df.copy()
    df['hr_zone'] = df['heart_rate'].apply(hr_zone)
    return df, zones

def time_in_zone(df, zone_col='hr_zone'):
    # Compute time delta for each row
    df = df.copy()
    df['time_delta_s'] = df.index.to_series().diff().dt.total_seconds().fillna(0)
    zone_times = df.groupby(zone_col)['time_delta_s'].sum()
    total_time = zone_times.sum()
    zone_pct = zone_times / total_time * 100
    return pd.DataFrame({'seconds': zone_times, 'percent': zone_pct})

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
        strategy = 'Negative split (faster second half)'
    elif pace2 > pace1:
        strategy = 'Positive split (slower second half)'
    else:
        strategy = 'Even pacing'
    return {'first_half_pace': pace1, 'second_half_pace': pace2, 'strategy': strategy}

def generate_recommendations(results):
    recs = []
    # Time in high zones
    if results['time_in_zone'].loc['Z4 (Threshold)', 'percent'] > 30 or results['time_in_zone'].loc['Z5 (VO2max)', 'percent'] > 10:
        recs.append("Consider more aerobic base training to balance high-intensity work.")
    # HR drift
    if results['hr_drift']['hr_drift_pct'] > 5:
        recs.append("Significant HR drift detected—focus on aerobic endurance and pacing.")
    # Pacing
    if results['pacing']['strategy'].startswith('Positive'):
        recs.append("Try to avoid slowing down in the second half—work on even pacing or negative splits.")
    if not recs:
        recs.append("Great job! Your run shows good balance and control.")
    return recs

def main():
    parser = argparse.ArgumentParser(description='Advanced run performance analysis.')
    parser.add_argument('--input', type=str, required=True, help='Path to input summary CSV')
    parser.add_argument('--figures_dir', type=str, required=True, help='Directory to save figures')
    parser.add_argument('--prefix', type=str, required=True, help='Prefix for output files (e.g., date_activity)')
    args = parser.parse_args()

    df = pd.read_csv(args.input, index_col=0, parse_dates=True)
    hrmax = df['heart_rate'].max() if 'heart_rate' in df else 199
    df, zones = compute_training_zones(df, hrmax)
    tiz = time_in_zone(df)
    hr_drift = compute_hr_drift(df)
    pacing = pacing_strategy(df)
    results = {'time_in_zone': tiz, 'hr_drift': hr_drift, 'pacing': pacing}
    recs = generate_recommendations(results)
    print("\n--- Training Zone Time (%) ---")
    print(tiz)
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
    run_dir = os.path.join(args.figures_dir, f"week{week}", args.prefix)
    os.makedirs(run_dir, exist_ok=True)
    # Create subfolders for images and txt
    img_dir = os.path.join(run_dir, "images")
    txt_dir = os.path.join(run_dir, "txt")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    # Plot time in zone
    tiz['percent'].plot(kind='bar', color='skyblue', title='Time in HR Zone (%)')
    plt.ylabel('% of Run Time')
    plt.tight_layout()
    plt.savefig(f"{img_dir}/time_in_hr_zone.png")
    plt.close()
    # Save textual representation of time in HR zone
    tiz.to_csv(f"{txt_dir}/time_in_hr_zone.txt", sep='\t')
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
    with open(f"{txt_dir}/run_summary.txt", "w") as f:
        f.write("\n".join(summary_lines) + "\n")

if __name__ == '__main__':
    main()
