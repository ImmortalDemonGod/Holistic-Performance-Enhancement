import pandas as pd

def detect_strides(df, pace_threshold=4.5, min_stride_duration=8, max_stride_duration=40, cadence_threshold=80):
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

def main():
    csv_path = "cultivation/data/processed/20250429_191120_baseox_wk1_tue_z2_strides_25min_147_155bpm_6x20shill_gpx_summary.csv"
    df = pd.read_csv(csv_path, parse_dates=['time'])
    df = df.set_index('time')
    df = detect_strides(df)
    stride_blocks = df[df['stride']].copy()
    stride_blocks['block'] = (stride_blocks.index.to_series().diff().dt.total_seconds() > 5).cumsum()
    stride_groups = stride_blocks.groupby('block')
    print(f"Stride segments detected: {stride_groups.ngroups}")
    for i, group in stride_groups:
        print(f"Stride {i+1}: {group.index[0]} to {group.index[-1]}, duration: {(group.index[-1]-group.index[0]).total_seconds():.1f}s, avg pace: {group['pace_min_per_km'].mean():.2f}, avg HR: {group['heart_rate'].mean():.1f}")

if __name__ == "__main__":
    main()
