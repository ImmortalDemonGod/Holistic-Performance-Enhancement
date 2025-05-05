"""
detect_strides_temp.py
---------------------
Detects stride intervals in running data based on pace and cadence thresholds.

This script can be used standalone or as an importable module for stride detection in the run processing pipeline.

Usage (CLI):
    python detect_strides_temp.py --input <input_csv> --output <output_csv>

Inputs:
    - CSV file with columns: 'time', 'pace_min_per_km', 'cadence', 'heart_rate', ...
Outputs:
    - CSV file with an added 'stride' boolean column indicating stride intervals.
    - Prints summary of detected stride segments.

Typical pipeline role: Called by parse_run_files.py or metrics.py to annotate strides in processed run data.
"""
import pandas as pd
import argparse
import os

def detect_strides(df, pace_threshold=4.5, min_stride_duration=8, max_stride_duration=40, cadence_threshold=80):
    """
    Annotate stride intervals in a DataFrame based on pace and cadence.
    Args:
        df: DataFrame indexed by datetime with 'pace_min_per_km', 'cadence' columns.
        pace_threshold: Max pace (min/km) for stride.
        min_stride_duration: Min stride duration (s).
        max_stride_duration: Max stride duration (s).
        cadence_threshold: Min cadence for stride.
    Returns:
        DataFrame with added 'stride' boolean column.
    """
    df = df.copy()
    stride_candidate = (df['pace_min_per_km'] < pace_threshold) & (df['cadence'] > cadence_threshold)
    df['stride'] = False
    in_stride = False
    stride_start_idx = None
    for idx in df.index:
        if stride_candidate.loc[idx]:
            if not in_stride:
                in_stride = True
                stride_start_idx = idx
        else:
            if in_stride:
                in_stride = False
                stride_end_idx = idx
                start_pos = df.index.get_loc(stride_start_idx)
                end_pos = df.index.get_loc(stride_end_idx)
                stride_duration = (df.iloc[end_pos-1].name - df.iloc[start_pos].name).total_seconds()
                if min_stride_duration <= stride_duration <= max_stride_duration:
                    df.loc[stride_start_idx:df.iloc[end_pos-1].name, 'stride'] = True
                stride_start_idx = None
    if in_stride and stride_start_idx is not None:
        stride_duration = (df.iloc[-1].name - df.loc[stride_start_idx].name).total_seconds()
        if min_stride_duration <= stride_duration <= max_stride_duration:
            df.loc[stride_start_idx:df.iloc[-1].name, 'stride'] = True
    return df

def summarize_strides(df):
    """
    Print a summary of detected stride segments.
    """
    stride_blocks = df[df['stride']].copy()
    if stride_blocks.empty:
        print("No stride segments detected.")
        return
    stride_blocks['block'] = (stride_blocks.index.to_series().diff().dt.total_seconds() > 5).cumsum()
    stride_groups = stride_blocks.groupby('block')
    print(f"Stride segments detected: {stride_groups.ngroups}")
    for i, group in stride_groups:
        print(f"Stride {i+1}: {group.index[0]} to {group.index[-1]}, duration: {(group.index[-1]-group.index[0]).total_seconds():.1f}s, avg pace: {group['pace_min_per_km'].mean():.2f}, avg HR: {group['heart_rate'].mean():.1f}")

def main():
    parser = argparse.ArgumentParser(description="Detect stride intervals in running CSV data.")
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--output', '-o', help='Output CSV file (optional, will overwrite input if omitted)')
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=['time'])
    df = df.set_index('time')
    df = detect_strides(df)
    summarize_strides(df)
    output_path = args.output if args.output else args.input
    df.to_csv(output_path)
    print(f"Stride-annotated data written to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
