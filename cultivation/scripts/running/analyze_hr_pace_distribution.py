import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze HR and pace distribution for a run summary CSV.')
    parser.add_argument('--input', type=str, required=True, help='Path to input summary CSV')
    parser.add_argument('--figures_dir', type=str, required=True, help='Directory to save figures')
    parser.add_argument('--prefix', type=str, required=True, help='Prefix for output files (e.g., date_activity)')
    args = parser.parse_args()

    # Load the detailed GPX summary (preferred for elevation and completeness)
    df = pd.read_csv(args.input, index_col=0, parse_dates=True)

    # Drop rows with missing HR or pace
    hr_df = df.dropna(subset=['heart_rate'])
    pace_df = df.dropna(subset=['pace_min_per_km'])

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

    # Plot heart rate distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(hr_df['heart_rate'], bins=20, kde=True, color='red')
    plt.title('Heart Rate Distribution')
    plt.xlabel('Heart Rate (bpm)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{img_dir}/hr_distribution.png")
    plt.close()

    # --- TEXTUAL REPRESENTATIONS OF PLOTS ---
    # Heart Rate Distribution
    hr_desc = hr_df['heart_rate'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    with open(f"{txt_dir}/hr_distribution.txt", "w") as f:
        f.write("Heart Rate Distribution (bpm):\n")
        f.write(hr_desc.to_string())
        f.write("\n")

    # Plot pace distribution (less spread, more readable bins)
    plt.figure(figsize=(10, 5))
    pace_values = pace_df['pace_min_per_km']
    lower, upper = pace_values.quantile(0.01), pace_values.quantile(0.99)
    trimmed_pace = pace_values[(pace_values >= lower) & (pace_values <= upper)]
    bin_width = 0.1
    bins = int((upper - lower) / bin_width)
    sns.histplot(trimmed_pace, bins=bins, kde=True, color='blue')
    plt.title('Pace Distribution (min/km)')
    plt.xlabel('Pace (min/km)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{img_dir}/pace_distribution.png")
    plt.close()

    # Pace Distribution
    pace_desc = pace_df['pace_min_per_km'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    with open(f"{txt_dir}/pace_distribution.txt", "w") as f:
        f.write("Pace Distribution (min/km):\n")
        f.write(pace_desc.to_string())
        f.write("\n")

    # 2D joint distribution: Heart Rate vs. Pace
    plt.figure(figsize=(8, 6))
    sns.jointplot(x='pace_min_per_km', y='heart_rate', data=df, kind='hex', cmap='viridis')
    plt.suptitle('Heart Rate vs. Pace (min/km)', y=1.02)
    plt.savefig(f"{img_dir}/hr_vs_pace_hexbin.png")
    plt.close()

    # HR vs Pace: correlation
    corr = df[['heart_rate', 'pace_min_per_km']].corr().iloc[0,1]
    with open(f"{txt_dir}/hr_vs_pace_hexbin.txt", "w") as f:
        f.write("Correlation between Heart Rate and Pace (min/km):\n")
        f.write(f"Correlation coefficient: {corr:.3f}\n")

    # Optional: Print some quantiles
    print('Heart Rate Quantiles:')
    print(hr_df['heart_rate'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))
    print('\nPace Quantiles (min/km):')
    print(pace_df['pace_min_per_km'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))

    print(f'\nImages saved in {img_dir}, textual summaries in {txt_dir}')
