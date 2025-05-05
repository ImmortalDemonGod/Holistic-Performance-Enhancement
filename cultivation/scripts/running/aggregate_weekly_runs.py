import pandas as pd
import glob
import os
import numpy as np
from pathlib import Path

# Directory containing processed CSVs
data_dir = Path(__file__).parents[2] / 'data' / 'processed'
output_parquet = Path(__file__).parents[2] / 'data' / 'weekly_metrics.parquet'

all_files = glob.glob(str(data_dir / '*_summary.csv'))
if not all_files:
    print("No summary CSVs found.")
    exit(1)

# Skip files with '_hr_override_' to avoid double-counting
all_files = [f for f in all_files if '_hr_override_' not in str(f)]

frames = []
for f in all_files:
    df = pd.read_csv(f)
    # Patch: if 'file' column missing, add it from filename
    if 'file' not in df.columns:
        df['file'] = os.path.basename(f)
    frames.append(df)
df = pd.concat(frames, ignore_index=True)

if 'date' not in df.columns:
    if 'start_time' in df.columns:
        df['date'] = pd.to_datetime(df['start_time'])
    else:
        # Now safe to extract date from 'file'
        df['date'] = pd.to_datetime(df['file'].str.extract(r'(\d{8})')[0], errors='coerce')
        # Now 'date' is datetime64; .dt access is valid
df['week'] = df['date'].dt.isocalendar().week
df['year'] = df['date'].dt.isocalendar().year

# Ensure all required columns exist for aggregation
REQUIRED_COLS = [
    'decoupling_%', 'distance_km', 'efficiency_factor',
    'duration_min', 'avg_hr', 'hrTSS', 'rpe'
]
for col in REQUIRED_COLS:
    if col not in df.columns:
        if col == 'distance_km':
            # Use distance_km if present, else total_distance_km, else NaN
            if 'distance_km' not in df.columns:
                if 'total_distance_km' in df.columns:
                    df['distance_km'] = df['total_distance_km']
                else:
                    df['distance_km'] = float('nan')
        else:
            df[col] = float('nan')

groups = df.groupby(['year', 'week'])
agg = groups.agg({
    'efficiency_factor': 'mean',
    'decoupling_%': 'mean',
    'distance_km': 'sum',
    'rpe': 'mean',
})
agg = agg.rename(columns={
    'efficiency_factor': 'ef_mean',
    'decoupling_%': 'decoupling_mean',
    'distance_km': 'km',
    'rpe': 'rpe_avg',
})
agg = agg.reset_index()
agg.to_parquet(output_parquet, index=False)
print(f"Wrote weekly metrics to {output_parquet}")