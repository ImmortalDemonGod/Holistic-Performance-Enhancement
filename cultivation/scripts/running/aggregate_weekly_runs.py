import pandas as pd
import glob
import os
from pathlib import Path

# Directory containing processed CSVs
data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
output_parquet = Path(__file__).parent.parent.parent / 'data' / 'weekly_metrics.parquet'

all_files = glob.glob(str(data_dir / '*_summary.csv'))
if not all_files:
    raise FileNotFoundError(f"No summary CSVs found in {data_dir}")

# Skip files with '_hr_override_' to avoid double-counting
all_files = [f for f in all_files if '_hr_override_' not in str(f)]

frames = [pd.read_csv(f) for f in all_files]
df = pd.concat(frames)

if 'date' not in df.columns:
    if 'start_time' in df.columns:
        df['date'] = pd.to_datetime(df['start_time']).dt.date
    else:
        # Try to infer from filename if needed
        df['date'] = pd.to_datetime(df['file'].str.extract(r'(\d{8})')[0], errors='coerce').dt.date

df['week'] = df['date'].dt.isocalendar().week
df['year'] = df['date'].dt.isocalendar().year

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