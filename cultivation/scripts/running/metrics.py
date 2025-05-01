"""
hpe.running.metrics
===================

Core utilities for turning raw GPX tracks into physiology-aware
run summaries (distance, pace, Efficiency Factor, aerobic decoupling,
and HR-based Training Stress Score).

Usage
-----
>>> from metrics import parse_gpx, run_metrics
>>> df = parse_gpx("20250425_201748_afternoon_run.gpx")
>>> summary = run_metrics(df, threshold_hr=175, resting_hr=50)
>>> print(summary)

Copyright 2025 3q & contributors
License: MIT
"""
from __future__ import annotations

import math
import json
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ helpers
EARTH_RADIUS_M = 6_371_000  # mean Earth radius in metres

def _haversine(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Return great-circle distance in **metres** for two WGS-84 points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ------------------------------------------------------------------ parsing
def parse_gpx(path: str | Path) -> pd.DataFrame:
    """
    Parse a GPX file into a tidy **chronologically sorted** DataFrame.

    Returns columns
    ---------------
    time  : pandas.Timestamp (UTC)
    lat   : float  (deg)
    lon   : float  (deg)
    ele   : float  (m) – NaN if absent
    hr    : int    (bpm) – NaN if absent
    cadence : int  (rpm) – NaN if absent
    power  : int   (W)   – NaN if absent
    dt    : float  (s)  – seconds since previous point
    dist  : float  (m)  – segment distance
    speed_mps        : raw speed (m/s)
    speed_mps_smooth : 5-point rolling avg (m/s)
    pace_sec_km      : time to cover 1 km, smoothed (s/km)
    """
    ns = {
        "g": "http://www.topografix.com/GPX/1/1",
        "gpxtpx": "http://www.garmin.com/xmlschemas/TrackPointExtension/v1",
    }
    rows = []
    root = ET.parse(str(path)).getroot()
    for pt in root.findall(".//g:trkpt", ns):
        lat, lon = float(pt.attrib["lat"]), float(pt.attrib["lon"])
        ele_node = pt.find("g:ele", ns)
        ele = float(ele_node.text) if ele_node is not None else np.nan
        ts = pd.to_datetime(pt.find("g:time", ns).text, utc=True)
        hr_node = pt.find(".//gpxtpx:hr", ns)
        hr = int(hr_node.text) if hr_node is not None else np.nan
        cad_node = pt.find(".//gpxtpx:cad", ns)
        cad = int(cad_node.text) if cad_node is not None else np.nan
        power_node = pt.find(".//power", ns)
        power = int(power_node.text) if power_node is not None else np.nan
        rows.append((ts, lat, lon, ele, hr, cad, power))

    if not rows:
        raise ValueError(f"No <trkpt> found in {path}")

    df = (
        pd.DataFrame(rows, columns=["time", "lat", "lon", "ele", "hr", "cadence", "power"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    # elapsed seconds since previous point
    df["dt"] = df["time"].diff().dt.total_seconds().fillna(0)

    # great-circle distance of each segment
    seg_dist = [0.0] + [
        _haversine(
            df.at[i - 1, "lat"],
            df.at[i - 1, "lon"],
            df.at[i, "lat"],
            df.at[i, "lon"],
        )
        for i in range(1, len(df))
    ]
    df["dist"] = seg_dist

    # instantaneous speed and 5-sample rolling mean (≈5 s on 1 Hz GPX)
    df["speed_mps"] = df["dist"] / df["dt"].replace(0, np.nan)
    df["speed_mps"] = df["speed_mps"].fillna(method="bfill")
    df["speed_mps_smooth"] = (
        df["speed_mps"].rolling(window=5, center=True, min_periods=1).mean()
    )
    df["pace_sec_km"] = 1000 / df["speed_mps_smooth"].replace(0, np.nan)
    return df

# ------------------------------------------------------------------ metrics
_ZONES_FILE = Path(__file__).parents[3] / "data" / "zones_personal.yml"

def load_personal_zones():
    if not _ZONES_FILE.exists():
        raise FileNotFoundError(
            f"Personal zone file {_ZONES_FILE} missing; "
            "run `scripts/running/generate_zone_file.py` first"
        )
    with _ZONES_FILE.open() as f:
        zones = yaml.safe_load(f)
    # Validate schema
    for z in zones.values():
        assert 'bpm' in z and 'pace_min_per_km' in z, "Each zone must have bpm and pace_min_per_km"
        assert len(z['bpm']) == 2 and len(z['pace_min_per_km']) == 2, "Each range must have two values"
    return zones

def compute_training_zones(hr_array, pace_array, zones=None):
    if zones is None:
        zones = load_personal_zones()
    def zone_hr(hr):
        if np.isnan(hr):
            return None
        for name, z in zones.items():
            lo, hi = z['bpm']
            if lo <= hr <= hi:
                return name
        return None
    # Convert pace to min/km if needed
    # Use median-based check to avoid GPS spikes breaking scaling logic
    if pace_array.median() > 20:  # ≈ sec/km
        pace_min = pace_array / 60
    else:
        pace_min = pace_array
    def zone_pace(pace):
        if np.isnan(pace):
            return None
        for name, z in zones.items():
            lo, hi = z['pace_min_per_km']
            if lo <= pace <= hi:
                return name
        return None
    zone_hr_col = hr_array.apply(zone_hr)
    zone_pace_col = pace_min.apply(zone_pace)
    def zone_effective(hr, pace):
        if hr and pace:
            return hr if hr == pace else "mixed"
        return hr or pace
    zone_effective_col = [zone_effective(h, p) for h, p in zip(zone_hr_col, zone_pace_col)]
    return zone_hr_col, zone_pace_col, zone_effective_col

def run_metrics(
    df: pd.DataFrame, *, threshold_hr: int = 186, resting_hr: int = 50
) -> Dict[str, float]:
    """
    Compute headline metrics for a single steady-state run.

    Parameters
    ----------
    df : DataFrame from `parse_gpx`
    threshold_hr : Personal lactate-threshold HR (bpm)
    resting_hr   : Morning resting HR (bpm)

    Returns
    -------
    dict with keys:
        distance_km
        duration_min
        avg_pace_min_per_km
        avg_hr
        efficiency_factor
        decoupling_%     – aerobic decoupling (Pa:HR drift)
        hrTSS            – Training-Stress Score (HR-proxy)

    Notes
    -----
    • Efficiency Factor (EF) = avg speed (m/s) ÷ avg HR (bpm).
    • Aerobic decoupling is |EF² – EF¹| / EF¹ expressed %.
      < 5 % : strong aerobic base; 5-10 % : borderline; >10 % : drift.
    • hrTSS scales like TrainingPeaks TSS (100 ≈ 1 h at threshold).
    """
    total_dist_m = df["dist"].sum()
    secs = df["dt"].sum()
    avg_hr = df["hr"].mean()
    avg_speed = total_dist_m / secs  # m s⁻¹
    avg_pace = 1000 / avg_speed / 60  # min km⁻¹
    ef = avg_speed / avg_hr
    # Time-based midpoint for decoupling
    midpoint = df.index[0] + (df.index[-1] - df.index[0]) / 2
    first_half = df[df.index <= midpoint]
    second_half = df[df.index > midpoint]
    ef_1 = (
        first_half["dist"].sum()
        / first_half["dt"].sum()
        / first_half["hr"].mean()
    )
    ef_2 = (
        second_half["dist"].sum()
        / second_half["dt"].sum()
        / second_half["hr"].mean()
    )
    decouple_pct = abs(ef_2 - ef_1) / ef_1 * 100
    intensity_factor = (avg_hr - resting_hr) / (threshold_hr - resting_hr)
    hr_tss = secs * intensity_factor**2 / 36
    zones = load_personal_zones()
    zone_hr, zone_pace, zone_effective = compute_training_zones(df['hr'], df['pace_sec_km'], zones)
    df['zone_hr'] = zone_hr
    df['zone_pace'] = zone_pace
    df['zone_effective'] = zone_effective
    return {
        "distance_km": round(total_dist_m / 1_000, 2),
        "duration_min": round(secs / 60, 1),
        "avg_pace_min_per_km": round(avg_pace, 2),
        "avg_hr": round(avg_hr, 1),
        "efficiency_factor": round(ef, 5),
        "decoupling_%": round(decouple_pct, 2),
        "hrTSS": round(hr_tss, 1),
        "zones_applied": json.dumps(zones),
    }
