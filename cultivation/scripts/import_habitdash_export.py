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

# 1. Load export (long format)
export_df = pd.read_csv(EXPORT_CSV)

# Pivot export to wide format: index=date, columns=canonical metric name, values=value
# Determine the canonical column name (prefer 'cache_col' if present, else compose from name+source)
if 'cache_col' in export_df.columns:
    export_df['metric_col'] = export_df['cache_col']
else:
    # Fallback: create a canonical column name from name+source
    export_df['metric_col'] = export_df['name'].str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_per_') + '_' + export_df['source'].str.strip().str.lower()

wide_export = export_df.pivot_table(index='date', columns='metric_col', values='value', aggfunc='first').reset_index()
# Try to preserve date as datetime
wide_export['date'] = pd.to_datetime(wide_export['date'])

# 2. Load or initialize cache
def load_cache(path):
    if os.path.exists(path):
        return pd.read_parquet(path)
    else:
        return pd.DataFrame()

cache_df = load_cache(CACHE_PARQUET)
# If the date is the index, reset it to a column
if not cache_df.empty and 'date' not in cache_df.columns:
    if cache_df.index.name and 'date' in str(cache_df.index.name).lower():
        cache_df = cache_df.reset_index().rename(columns={cache_df.index.name: 'date'})
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

# 4. Merge export into cache (on 'date', add user if relevant)
if 'date' not in export_df.columns:
    raise ValueError("Export file must contain a 'date' column.")

export_df['date'] = pd.to_datetime(export_df['date'])
if not cache_df.empty:
    cache_df['date'] = pd.to_datetime(cache_df['date'])

# Use 'date' as the unique key (add 'user' if multi-user)
merge_keys = ['date']

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
    merged = pd.merge(cache_df, wide_export, on=merge_keys, how='outer', suffixes=('_old', ''))
    # For each export column, prefer the export value if present
    for col in wide_export.columns:
        if col not in merge_keys:
            col_old = f"{col}_old"
            if col_old in merged.columns:
                merged[col] = merged[col].combine_first(merged[col_old])
                merged.drop(columns=[col_old], inplace=True)
            # If col_old doesn't exist, merged[col] is already correct

# 5. Ensure all export columns are present
for col in wide_export.columns:
    if col not in merged.columns:
        merged[col] = pd.NA

# 6. Write back to Parquet
merged = merged.sort_values('date').reset_index(drop=True)
merged.to_parquet(CACHE_PARQUET, index=False)
print(f"Cache updated: all export data imported into {CACHE_PARQUET}")
