import numpy as np
import pandas as pd
import math

def filter_gps_jitter(df, pace_col, cad_col, cad_thr):
    """Keep rows where pace > 8.7 min/km (14 min/mile) OR cadence < 140 spm (i.e., walking if either is true)."""
    walk_pace = (df[pace_col] > 8.7)
    walk_cad = (df[cad_col] < 140)
    return df[walk_pace | walk_cad]


def drop_short_segments(segments, min_duration=5):
    """Drop segments shorter than min_duration seconds."""
    return [s for s in segments if s['dur_s'] >= min_duration]


def compute_time_weighted_pace(dur_s, dist_km):
    """Compute pace as time_minutes / distance_km."""
    return (dur_s / 60) / dist_km if dist_km > 0 else float('nan')


def summarize_walk_segments(segments):
    """
    Given a list of segment dicts, return summary statistics using only valid segments:
    - total_walk_time
    - total_walk_dist
    - avg_pace
    - avg_hr
    - max_hr
    - avg_cad
    """
    valid_segments = [
        s for s in segments
        if s['dur_s'] > 0
        and isinstance(s['dist_km'], (int, float))
        and s['dist_km'] > 0
    ]
    total_walk_time = sum(s['dur_s'] for s in valid_segments)
    total_walk_dist = sum(s['dist_km'] for s in valid_segments if s.get('dist_km') is not None and not math.isnan(s['dist_km']))
    avg_pace = np.mean([s['avg_pace_min_km'] for s in valid_segments if s['avg_pace_min_km'] != '' and not pd.isnull(s['avg_pace_min_km'])])
    avg_hr = np.mean([s['avg_hr'] for s in valid_segments if s['avg_hr'] != '' and not pd.isnull(s['avg_hr'])])
    max_hr = max([s['avg_hr'] for s in valid_segments if s['avg_hr'] != '' and not pd.isnull(s['avg_hr'])]) if valid_segments else ''
    avg_cad = np.mean([s['avg_cad'] for s in valid_segments if s['avg_cad'] != '' and not pd.isnull(s['avg_cad'])])
    return {
        'total_walk_time': total_walk_time,
        'total_walk_dist': total_walk_dist,
        'avg_pace': avg_pace,
        'avg_hr': avg_hr,
        'max_hr': max_hr,
        'avg_cad': avg_cad,
        'valid_segments': valid_segments
    }


def walk_block_segments(gpx_df, is_walk_col, pace_col, cad_col, cad_thr=128, max_gap_s=2, min_dur_s=2):
    """
    Group contiguous walk blocks after filtering GPS jitter, allowing up to `max_gap_s` seconds of non-walk between walk intervals.
    Returns a list of segment dicts.
    """
    # Filter walk points to remove GPS jitter
    walk_df = filter_gps_jitter(
        gpx_df[gpx_df[is_walk_col]].copy(), pace_col, cad_col, cad_thr
    )

    # Identify start/end of blocks allowing for gaps
    blocks = []
    block_start = None
    last_walk_idx = None
    for _, (idx, row) in enumerate(gpx_df.iterrows()):
        # Check if this is a valid walk point (in original data AND passed the jitter filter)
        if row[is_walk_col] and idx in walk_df.index:
            if block_start is None:
                block_start = idx
            last_walk_idx = idx
            # last_walk_i variable removed (unused)
        else:
            # Non-walk row
            if block_start is not None:
                # Check if gap exceeds max_gap_s
                time_gap = (idx - last_walk_idx).total_seconds() if last_walk_idx is not None else None
                if time_gap is not None and time_gap > max_gap_s:
                    blocks.append((block_start, last_walk_idx))
                    block_start = None
                    last_walk_idx = None
    # Close last block
    if block_start is not None and last_walk_idx is not None:
        blocks.append((block_start, last_walk_idx))
    # Now build segments for each block
    session_start = gpx_df.index[0]
    session_end = gpx_df.index[-1]
    session_duration = (session_end - session_start).total_seconds()
    segments = []
    seg_id = 1
    for start_ts, end_ts in blocks:
        grp_df = gpx_df.loc[start_ts:end_ts]
        dur_s = (end_ts - start_ts).total_seconds()
        if dur_s < min_dur_s:
            continue
        dist_km = float(grp_df['distance_cumulative_km'].iloc[-1] - grp_df['distance_cumulative_km'].iloc[0]) if 'distance_cumulative_km' in grp_df.columns else np.nan
        # --- DEBUG for sanity fail ---
        if dur_s >= 60 and (pd.isnull(dist_km) or dist_km <= 0):
            print(f"[SANITY DEBUG] BAD SEGMENT: seg_id={seg_id}, start={start_ts}, end={end_ts}, dur_s={dur_s}, dist_km={dist_km}")
            print(grp_df[['distance_cumulative_km', 'dt']])
        avg_pace = compute_time_weighted_pace(dur_s, dist_km)
        if pd.isnull(avg_pace) and not pd.isnull(dist_km) and dist_km > 0:
            avg_pace_val = ''
            dist_km_val = round(dist_km, 3)
        else:
            avg_pace_val = round(avg_pace, 1) if not pd.isnull(avg_pace) else ''
            dist_km_val = round(dist_km, 3) if not pd.isnull(dist_km) else ''
        avg_hr = grp_df['heart_rate'].mean() if 'heart_rate' in grp_df.columns else np.nan
        avg_cad = grp_df['cadence'].replace(0, np.nan).mean() if 'cadence' in grp_df.columns else np.nan
        start_offset = (start_ts - session_start).total_seconds()
        end_offset = (end_ts - session_start).total_seconds()
        if start_offset < 60:
            tag = 'warm-up'
        elif session_duration - end_offset < 120:
            tag = 'cool-down'
        else:
            tag = 'mid-session'
        note = ''
        if tag == 'mid-session' and dur_s < 30 and dist_km < 0.05:
            note = 'pause?'
        # --- Optionally skip bad segment ---
        if dur_s >= 60 and (pd.isnull(dist_km) or dist_km <= 0):
            note += '[SKIPPED: BAD SEGMENT]'
            print(f"[SANITY DEBUG] SKIPPING SEGMENT seg_id={seg_id}")
            continue
        segments.append({
            'segment_id': seg_id,
            'start_ts': str(start_ts),
            'end_ts': str(end_ts),
            'dur_s': int(dur_s),
            'dist_km': dist_km_val,
            'avg_pace_min_km': avg_pace_val,
            'avg_hr': round(avg_hr, 1) if not pd.isnull(avg_hr) else '',
            'avg_cad': round(avg_cad, 1) if not pd.isnull(avg_cad) else '',
            'tag': tag,
            'start_offset_s': int(start_offset),
            'end_offset_s': int(end_offset),
            'note': note
        })
        seg_id += 1
    return segments
