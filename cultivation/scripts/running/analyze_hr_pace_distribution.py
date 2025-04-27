import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

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

    # Plot heart rate distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(hr_df['heart_rate'], bins=20, kde=True, color='red')
    plt.title('Heart Rate Distribution')
    plt.xlabel('Heart Rate (bpm)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{args.figures_dir}/{args.prefix}_hr_distribution.png")
    plt.close()

    # Plot pace distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(pace_df['pace_min_per_km'], bins=20, kde=True, color='blue')
    plt.title('Pace Distribution')
    plt.xlabel('Pace (min/km)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{args.figures_dir}/{args.prefix}_pace_distribution.png")
    plt.close()

    # 2D joint distribution: Heart Rate vs. Pace
    plt.figure(figsize=(8, 6))
    sns.jointplot(x='pace_min_per_km', y='heart_rate', data=df, kind='hex', cmap='viridis')
    plt.suptitle('Heart Rate vs. Pace (min/km)', y=1.02)
    plt.savefig(f"{args.figures_dir}/{args.prefix}_hr_vs_pace_hexbin.png")
    plt.close()

    # Optional: Print some quantiles
    print('Heart Rate Quantiles:')
    print(hr_df['heart_rate'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))
    print('\nPace Quantiles (min/km):')
    print(pace_df['pace_min_per_km'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))

    print(f'\nPlots saved as {args.prefix}_hr_distribution.png, {args.prefix}_pace_distribution.png, and {args.prefix}_hr_vs_pace_hexbin.png in {args.figures_dir}')
