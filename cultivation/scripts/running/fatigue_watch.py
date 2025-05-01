#!/usr/bin/env python
"""
Scan subjective.csv → raise GitHub Issue+label when any red-flag rule breaches.

Run nightly via cron Action OR manual GH-CLI.
"""
import os, json, datetime as dt
import pandas as pd
from subprocess import run, CalledProcessError
from pathlib import Path

SUBJ = Path(__file__).parents[3] / "data" / "subjective.csv"
WINDOW = 7  # days

REST_HR_UP   = 8     # bpm
HRV_DROP_PCT = 20    # %
SRPE_ALERT   = 6     # RPE units

def main():
    df = pd.read_csv(SUBJ, parse_dates=["date"]).tail(WINDOW)
    if len(df) < WINDOW:
        return  # not enough data yet

    rest_delta = df["rest_hr"].iloc[-1] - df["rest_hr"].mean()
    hrv_drop   = (df["hrv"].mean() - df["hrv"].iloc[-1]) / df["hrv"].mean() * 100
    srpe_hi    = df["rpe"].max() >= SRPE_ALERT

    if (rest_delta > REST_HR_UP) or (hrv_drop > HRV_DROP_PCT) or srpe_hi:
        title = "[Auto] Fatigue flag – check training load"
        body  = f"""Rest-HR ↑ {rest_delta:+.1f} bpm, HRV ↓ {hrv_drop:.0f} %, max sRPE {df['rpe'].max()}."""
        try:
            run(["gh", "issue", "create", "-t", title, "-b", body, "-l", "fatigue-alert"], check=True)
        except CalledProcessError as e:
            print("⚠️  Unable to create issue:", e)

if __name__ == "__main__":
    main()
