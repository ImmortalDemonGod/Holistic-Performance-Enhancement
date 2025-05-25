# NOTE: Seaborn is permitted in this script, but NOT in any python_user_visible tool or UI-facing code per project policy.

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
    pace_values = pace_df['pace_min_per_km'] # pace_df has NaNs dropped from this column

    if pace_values.empty:
        print(f"No pace data available in {args.input} after NaN removal. Plotting placeholder.")
        plt.text(0.5, 0.5, "No pace data available", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.title('Pace Distribution (min/km)')
        plt.xlabel('Pace (min/km)')
        plt.ylabel('Frequency')
    else:
        lower, upper = pace_values.quantile(0.01), pace_values.quantile(0.99)
        
        # Check if quantiles are valid for bin calculation
        if pd.notna(lower) and pd.notna(upper) and upper > lower:
            trimmed_pace = pace_values[(pace_values >= lower) & (pace_values <= upper)]
            
            # If trimming results in an empty series or too few points for a meaningful distribution
            if trimmed_pace.empty or len(trimmed_pace) < 2:
                 print(f"Trimmed pace data is empty or too small (lower={lower}, upper={upper}, len={len(trimmed_pace)}). Using all pace values with default bins.")
                 bins = 20 # Default bins
                 sns.histplot(pace_values, bins=bins, kde=True, color='blue')
            else:
                bin_width = 0.1
                num_bins_float = (upper - lower) / bin_width
                bins = max(1, int(num_bins_float)) # Ensure bins is at least 1
                sns.histplot(trimmed_pace, bins=bins, kde=True, color='blue')
        else:
            # Fallback if quantiles are problematic (e.g., all values are the same, lower=upper, or NaN)
            print(f"Could not determine a valid range from quantiles for pace (lower={lower}, upper={upper}). Using all pace values with default bins.")
            bins = 20 # Default bins
            sns.histplot(pace_values, bins=bins, kde=True, color='blue')
            
        plt.title('Pace Distribution (min/km)')
        plt.xlabel('Pace (min/km)')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f"{img_dir}/pace_distribution.png")
    plt.close()

    # Pace Distribution text summary
    if not pace_df.empty:
        pace_desc = pace_df['pace_min_per_km'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        with open(f"{txt_dir}/pace_distribution.txt", "w") as f:
            f.write("Pace Distribution (min/km):\n")
            f.write(pace_desc.to_string())
            f.write("\n")
    else:
        with open(f"{txt_dir}/pace_distribution.txt", "w") as f:
            f.write("Pace Distribution (min/km):\n")
            f.write("No pace data available for summary.\n")

    # 2D joint distribution: Heart Rate vs. Pace
    subset = df[['pace_min_per_km', 'heart_rate']].dropna()
    if not subset.empty:
        try:
            g = sns.jointplot(x='pace_min_per_km', y='heart_rate', data=subset, kind='hex', cmap='viridis')
            g.fig.suptitle('Heart Rate vs. Pace (min/km)', y=1.02)
            g.fig.tight_layout()
            g.fig.savefig(f"{img_dir}/hr_vs_pace_hexbin.png")
            plt.close('all')
        except Exception as e:
            print(f"Skipping HR vs Pace hexbin: {e}")
    else:
        print("Not enough data for HR vs Pace hexbin plot.")
    # HR vs Pace: correlation
    if not subset.empty:
        corr = subset.corr().iloc[0, 1]
        with open(f"{txt_dir}/hr_vs_pace_hexbin.txt", "w") as f:
            f.write("Correlation between Heart Rate and Pace (min/km):\n")
            f.write(f"Correlation coefficient: {corr:.3f}\n")

    # --- POWER ANALYSIS ---
    if 'power' in df.columns:
        power_df = df.dropna(subset=['power'])
        # Plot power distribution
        plt.figure(figsize=(10, 5))
        sns.histplot(power_df['power'], bins=20, kde=True, color='orange')
        plt.title('Power Distribution (W)')
        plt.xlabel('Power (W)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f"{img_dir}/power_distribution.png")
        plt.close()
        # Text summary
        power_desc = power_df['power'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        with open(f"{txt_dir}/power_distribution.txt", "w") as f:
            f.write("Power Distribution (W):\n")
            f.write(power_desc.to_string())
            f.write("\n")

    # --- CADENCE ANALYSIS ---
    if 'cadence' in df.columns:
        cadence_df = df.dropna(subset=['cadence'])
        # Plot cadence distribution
        plt.figure(figsize=(10, 5))
        sns.histplot(cadence_df['cadence'], bins=20, kde=True, color='green')
        plt.title('Cadence Distribution (spm)')
        plt.xlabel('Cadence (spm)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f"{img_dir}/cadence_distribution.png")
        plt.close()
        # Text summary
        cadence_desc = cadence_df['cadence'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        with open(f"{txt_dir}/cadence_distribution.txt", "w") as f:
            f.write("Cadence Distribution (spm):\n")
            f.write(cadence_desc.to_string())
            f.write("\n")

    # Optional: Print some quantiles
    print('Heart Rate Quantiles:')
    print(hr_df['heart_rate'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))
    print('\nPace Quantiles (min/km):')
    print(pace_df['pace_min_per_km'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))

    print(f'\nImages saved in {img_dir}, textual summaries in {txt_dir}')
