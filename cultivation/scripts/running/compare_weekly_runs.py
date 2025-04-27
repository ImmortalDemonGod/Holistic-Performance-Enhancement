import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


def load_run_summary(csv_path):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return df


def extract_week_and_label(filename):
    # Assumes filename like '20250421_000013_evening_run_gpx_summary.csv'
    base = os.path.basename(filename)
    date_part = base.split('_')[0]  # '20250421'
    label = base.replace('_gpx_summary.csv', '').replace('.csv', '')
    week = pd.to_datetime(date_part).isocalendar().week
    return week, label


def compare_two_runs(run1_path, run2_path, figures_dir):
    df1 = load_run_summary(run1_path)
    df2 = load_run_summary(run2_path)
    week1, label1 = extract_week_and_label(run1_path)
    week2, label2 = extract_week_and_label(run2_path)
    # Only compare if in the same week
    if week1 != week2:
        print(f"Runs are from different weeks: {week1} vs {week2}. Aborting.")
        return
    # Create week directory
    week_dir = os.path.join(figures_dir, f"week{week1}")
    os.makedirs(week_dir, exist_ok=True)
    # Create subfolders for images and txt
    img_dir = os.path.join(week_dir, "images")
    txt_dir = os.path.join(week_dir, "txt")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    # Align by time or distance if needed
    # Plot heart rate over time
    plt.figure(figsize=(12, 6))
    plt.plot(df1.index, df1['heart_rate'], label=f'{label1} HR', alpha=0.7)
    plt.plot(df2.index, df2['heart_rate'], label=f'{label2} HR', alpha=0.7)
    plt.title(f'Heart Rate Comparison (Week {week1})')
    plt.xlabel('Time')
    plt.ylabel('Heart Rate (bpm)')
    plt.legend()
    plt.tight_layout()
    hr_fig_path = os.path.join(img_dir, 'compare_hr.png')
    plt.savefig(hr_fig_path)
    plt.close()
    # --- TEXTUAL REPRESENTATIONS OF COMPARISON PLOTS ---
    # Heart Rate Comparison
    hr1_mean = df1['heart_rate'].mean()
    hr2_mean = df2['heart_rate'].mean()
    hr1_max = df1['heart_rate'].max()
    hr2_max = df2['heart_rate'].max()
    with open(os.path.join(txt_dir, 'compare_hr.txt'), 'w') as f:
        f.write(f"Heart Rate Comparison (Week {week1}):\n")
        f.write(f"{label1}: mean={hr1_mean:.1f}, max={hr1_max}\n")
        f.write(f"{label2}: mean={hr2_mean:.1f}, max={hr2_max}\n")
    # Plot pace over time
    plt.figure(figsize=(12, 6))
    plt.plot(df1.index, df1['pace_min_per_km'], label=f'{label1} Pace', alpha=0.7)
    plt.plot(df2.index, df2['pace_min_per_km'], label=f'{label2} Pace', alpha=0.7)
    plt.title(f'Pace Comparison (Week {week1})')
    plt.xlabel('Time')
    plt.ylabel('Pace (min/km)')
    plt.legend()
    plt.tight_layout()
    pace_fig_path = os.path.join(img_dir, 'compare_pace.png')
    plt.savefig(pace_fig_path)
    plt.close()
    # --- TEXTUAL REPRESENTATIONS OF COMPARISON PLOTS ---
    # Pace Comparison
    pace1_mean = df1['pace_min_per_km'].mean()
    pace2_mean = df2['pace_min_per_km'].mean()
    pace1_min = df1['pace_min_per_km'].min()
    pace2_min = df2['pace_min_per_km'].min()
    with open(os.path.join(txt_dir, 'compare_pace.txt'), 'w') as f:
        f.write(f"Pace Comparison (Week {week1}):\n")
        f.write(f"{label1}: mean={pace1_mean:.2f}, min={pace1_min:.2f}\n")
        f.write(f"{label2}: mean={pace2_mean:.2f}, min={pace2_min:.2f}\n")
    print(f"Saved comparison plots: {hr_fig_path}, {pace_fig_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare two runs from the same week.')
    parser.add_argument('--run1', type=str, required=True, help='Path to first run summary CSV')
    parser.add_argument('--run2', type=str, required=True, help='Path to second run summary CSV')
    parser.add_argument('--figures_dir', type=str, required=True, help='Directory to save comparison figures')
    args = parser.parse_args()
    compare_two_runs(args.run1, args.run2, args.figures_dir)


if __name__ == '__main__':
    main()
