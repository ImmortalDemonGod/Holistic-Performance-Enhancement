#!/usr/bin/env python
"""
Scan subjective.csv → raise GitHub Issue+label when any red-flag rule breaches.

Run nightly via cron Action OR manual GH-CLI.
"""
import os, json, datetime as dt
import pandas as pd
from subprocess import run, CalledProcessError
from pathlib import Path

SUBJ = Path(__file__).parents[2] / "data" / "subjective.csv"
WELLNESS_PARQUET = Path(__file__).parents[2] / "data" / "daily_wellness.parquet"
WINDOW = 7  # days

REST_HR_UP   = 8     # bpm
HRV_DROP_PCT = 20    # %
SRPE_ALERT   = 6     # RPE units
ANKLE_PAIN_ALERT = 5  # threshold for ankle pain alert (scale 0–10)
SAT_HR_CAP = 151  # bpm threshold for Saturday runs

def main():
    # Load subjective data
    df = pd.read_csv(SUBJ, parse_dates=["date"]).tail(WINDOW)
    if len(df) < WINDOW:
        return  # not enough data yet

    # Load objective wellness data
    try:
        wellness_df = pd.read_parquet(WELLNESS_PARQUET)
        if not pd.api.types.is_datetime64_any_dtype(wellness_df.index):
            wellness_df.index = pd.to_datetime(wellness_df.index).date
    except Exception:
        wellness_df = None

    # Map dates for last WINDOW days
    date_list = df["date"].dt.date.tolist()
    rest_hr_obj = []
    hrv_obj = []
    recovery_obj = []
    for d in date_list:
        if wellness_df is not None and d in wellness_df.index:
            rest_hr_obj.append(wellness_df.loc[d].get("rhr_whoop", None))
            hrv_obj.append(wellness_df.loc[d].get("hrv_whoop", None))
            recovery_obj.append(wellness_df.loc[d].get("recovery_score_whoop", None))
        else:
            rest_hr_obj.append(None)
            hrv_obj.append(None)
            recovery_obj.append(None)

    # Use objective RHR/HRV if available, else fallback to subjective
    rest_hr_series = pd.Series(rest_hr_obj, index=df.index)
    hrv_series = pd.Series(hrv_obj, index=df.index)
    recovery_series = pd.Series(recovery_obj, index=df.index)
    df["rest_hr_obj"] = rest_hr_series
    df["hrv_obj"] = hrv_series
    df["recovery_score_obj"] = recovery_series

    # Use objective for calculations if present, else fallback
    rest_delta = (rest_hr_series.iloc[-1] if pd.notnull(rest_hr_series.iloc[-1]) else df["rest_hr"].iloc[-1]) - \
                 (rest_hr_series.mean() if rest_hr_series.notnull().any() else df["rest_hr"].mean())
    hrv_mean = hrv_series.mean() if hrv_series.notnull().any() else df["hrv"].mean()
    hrv_last = hrv_series.iloc[-1] if pd.notnull(hrv_series.iloc[-1]) else df["hrv"].iloc[-1]
    hrv_drop = (hrv_mean - hrv_last) / hrv_mean * 100 if hrv_mean else 0

    srpe_hi    = df["rpe"].max() >= SRPE_ALERT
    ankle_hi   = ("ankle_pain" in df.columns) and (df["ankle_pain"].max() >= ANKLE_PAIN_ALERT)
    is_saturday_run = (df["date"].iloc[-1].weekday() == 5)
    hr_cap_breach = is_saturday_run and (df.get('avg_hr', pd.Series()).iloc[-1] > SAT_HR_CAP)

    if (rest_delta > REST_HR_UP) or (hrv_drop > HRV_DROP_PCT) or srpe_hi or ankle_hi or hr_cap_breach:
        title = "[Auto] Fatigue flag – check training load"
        body  = f"""Rest-HR ↑ {rest_delta:+.1f} bpm, HRV ↓ {hrv_drop:.0f} %, max sRPE {df['rpe'].max()}"""
        if ankle_hi:
            max_ankle = df["ankle_pain"].max()
            body += f"\nAnkle pain alert: max={max_ankle} (≥ {ANKLE_PAIN_ALERT})"
        if hr_cap_breach:
            cap_hr = df.get('avg_hr', pd.Series()).iloc[-1]
            body += f"\nSaturday HR cap alert: avg {cap_hr:.0f} bpm > {SAT_HR_CAP}"
        if recovery_series.notnull().any():
            body += f"\nRecovery Score (Whoop, most recent): {recovery_series.iloc[-1] if pd.notnull(recovery_series.iloc[-1]) else 'n/a'}"
        body += "."  # End sentence
        try:
            run(["gh", "issue", "create", "-t", title, "-b", body, "-l", "fatigue-alert"], check=True)
        except CalledProcessError as e:
            print("⚠️  Unable to create issue:", e)

if __name__ == "__main__":
    main()
