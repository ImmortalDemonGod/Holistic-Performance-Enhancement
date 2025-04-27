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
    hr_fig_path = os.path.join(week_dir, 'compare_hr.png')
    plt.savefig(hr_fig_path)
    plt.close()
    # Plot pace over time
    plt.figure(figsize=(12, 6))
    plt.plot(df1.index, df1['pace_min_per_km'], label=f'{label1} Pace', alpha=0.7)
    plt.plot(df2.index, df2['pace_min_per_km'], label=f'{label2} Pace', alpha=0.7)
    plt.title(f'Pace Comparison (Week {week1})')
    plt.xlabel('Time')
    plt.ylabel('Pace (min/km)')
    plt.legend()
    plt.tight_layout()
    pace_fig_path = os.path.join(week_dir, 'compare_pace.png')
    plt.savefig(pace_fig_path)
    plt.close()
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
