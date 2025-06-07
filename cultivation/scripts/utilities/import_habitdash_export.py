"""
import_habitdash_export.py

Bulk import all data from the canonical Habit Dash export CSV into the daily wellness Parquet cache.

- Ensures all export columns are present in the cache.
- Merges or overwrites existing data on date (and user, if multi-user).
- Backs up the existing cache before making changes.

Usage:
    .venv/bin/python cultivation/scripts/import_habitdash_export.py

Edit the EXPORT_CSV and CACHE_PARQUET paths below if needed.
"""
import pandas as pd
import shutil
import os
from datetime import datetime

EXPORT_CSV = "cultivation/data/raw/habitdash_export_2025-06-06/2025-06-06 Habit Dash - Integrations (flat file).csv"
CACHE_PARQUET = "cultivation/data/daily_wellness.parquet"
BACKUP_DIR = "cultivation/data/cache_backups"

# 1. Load export (long format, canonical columns)
export_df = pd.read_csv(EXPORT_CSV)

# Map export metric names to canonical names where necessary
metric_name_map = {
    'strain score': 'strain',
    'sleep respiratory rate': 'respiratory_rate',
    # Add more mappings as needed
}

def canonical_metric_name(row):
    raw_name = row['name'].strip().lower()
    canonical = metric_name_map.get(raw_name, raw_name.replace(' ', '_').replace('/', '_per_'))
    return f"{canonical}_{row['source'].strip().lower()}"

export_df['metric_col'] = export_df.apply(canonical_metric_name, axis=1)

# Pivot to wide format: index=date, columns=metric_col, values=value
wide_export = export_df.pivot_table(index='date', columns='metric_col', values='value', aggfunc='first').reset_index()
# Robust date parsing
wide_export['date'] = pd.to_datetime(wide_export['date'], errors='raise')
if wide_export['date'].isna().any():
    raise ValueError("Some dates in the export could not be parsed. Check export file for malformed dates.")
# Diagnostic: print unique dates after pivot
print('Unique dates after pivot:', wide_export['date'].unique())
print('Is 2025-05-27 present?', pd.Timestamp('2025-05-27') in wide_export['date'].unique())
# Remove rows with 1970-01-01 or NaT (parsing failures)
wide_export = wide_export[~wide_export['date'].isin([pd.Timestamp('1970-01-01'), pd.NaT])]
# Set date as index for downstream reliability
wide_export = wide_export.set_index('date')


# 2. Load or initialize cache
def load_cache(path):
    if os.path.exists(path):
        return pd.read_parquet(path)
    else:
        return pd.DataFrame()

cache_df = load_cache(CACHE_PARQUET)

# Ensure 'date' is a column in cache_df (handle legacy index case)
if not cache_df.empty and 'date' not in cache_df.columns:
    # If the index is date-like, reset and rename
    idx_name = cache_df.index.name
    if idx_name and 'date' in str(idx_name).lower():
        cache_df = cache_df.reset_index().rename(columns={idx_name: 'date'})
    elif isinstance(cache_df.index, pd.DatetimeIndex):
        cache_df = cache_df.reset_index().rename(columns={'index': 'date'})
    else:
        # Try to infer if the index is date-like
        try:
            pd.to_datetime(cache_df.index)
            cache_df = cache_df.reset_index().rename(columns={'index': 'date'})
        except Exception:
            raise ValueError("Cache file index is not date-like and no 'date' column found.")

# 3. Backup cache
os.makedirs(BACKUP_DIR, exist_ok=True)
if os.path.exists(CACHE_PARQUET):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(CACHE_PARQUET, os.path.join(BACKUP_DIR, f"daily_wellness_{ts}.parquet"))
    print(f"Backup of cache created at {BACKUP_DIR}/daily_wellness_{ts}.parquet")

# 4. Merge export into cache (on 'date')
# After pivot and set_index above, 'date' is the index, not a column.
if wide_export.index.name != 'date':
    raise ValueError("Export DataFrame must have 'date' as index after pivot.")

# No need to parse or check 'date' as a column anymore. All merges will be on index.

# Ensure all columns from both wide_export and cache exist in both DataFrames
all_cols = set(wide_export.columns) | set(cache_df.columns)
for df in [wide_export, cache_df]:
    for col in all_cols:
        if col not in df.columns:
            df[col] = pd.NA

# Merge: wide_export takes precedence for overlapping dates
if cache_df.empty:
    merged = wide_export.copy()
else:
    # Ensure cache uses date as index for merge
    if 'date' in cache_df.columns:
        cache_df['date'] = pd.to_datetime(cache_df['date'], errors='coerce')
        cache_df = cache_df.set_index('date')
    merged = pd.concat([cache_df, wide_export])
    merged = merged[~merged.index.duplicated(keep='last')]
    merged = merged.sort_index()

# 5. Ensure all export columns are present
for col in wide_export.columns:
    if col not in merged.columns:
        merged[col] = pd.NA

# 6. Write back to Parquet
# Remove any remaining bad index values
merged = merged[~merged.index.isin([pd.Timestamp('1970-01-01'), pd.NaT])]
merged.to_parquet(CACHE_PARQUET)
print(f"Cache updated: all export data imported into {CACHE_PARQUET}")

