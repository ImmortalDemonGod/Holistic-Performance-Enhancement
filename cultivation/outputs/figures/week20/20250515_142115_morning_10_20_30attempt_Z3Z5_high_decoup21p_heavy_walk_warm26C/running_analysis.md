🧙🏾‍♂️: We are aligning on a systematic audit of a Cultivation running session.

INPUT METADATA
--------------
RUN_LABEL: "W20_Thu_10-20-30_attempt"
BLOCK_SPEC_PATH: `cultivation/outputs/training_plans/baseox_daily_plans/week20/Week20_Thu_2025-05-15_10-20-30_WORKOUT.md` (primary), `cultivation/outputs/training_plans/baseox_daily_plans/week20/GOAL.md` (secondary for context)
ZONE_YAML_PATH:  `cultivation/data/zones_personal.yml`
ZONE_DOC_PATH:   `cultivation/outputs/training_plans/pace-zones.md`
RUN_DIR:         `cultivation/outputs/figures/week20/20250515_142115_morning_10_20_30attempt_Z3Z5_high_decoup21p_heavy_walk_warm26C/`
RAW_FILE_MAP:
```
├── advanced_metrics.txt
├── cadence_distribution.txt
├── csv
│   └── walk_segments.csv
├── hr_distribution.txt
├── hr_over_time_drift.txt
├── hr_vs_pace_hexbin.txt
├── pace_distribution.txt
├── pace_over_time.txt
├── power_distribution.txt
├── run_only_summary.txt
├── run_summary.txt
├── session_full_summary.txt
├── stride_summary.txt
├── time_in_effective_zone.txt
├── time_in_fatigue_kpi_zone.txt
├── time_in_hr_zone.txt
├── time_in_pace_zone.txt
├── walk_hr_distribution.txt
├── walk_over_time.txt
├── walk_pace_distribution.txt
├── walk_summary.txt
└── weather.txt
```

TASK
----
1. **Analyze the Pre-Run Wellness Context section** (as shown below) in detail. For each metric, comment on its value, trend (Δ1d, Δ7d), and any notable deviations, improvements, or risks. Consider how these wellness factors might impact the athlete's readiness, performance, and recovery for this session. If multiple data sources are present (e.g., Whoop vs Garmin RHR), compare and interpret both. Highlight any discrepancies or patterns that could inform training decisions.

--- Pre-Run Wellness Context (Data for 2025-05-15 from `run_summary.txt`) ---
  HRV (Whoop): 90.5 ms (Δ1d: -14.5%, Δ7d: +3.3%)
  RHR (Whoop): 59.0 bpm (Δ1d: +1.7%, Δ7d: +5.4%)
  RHR (Garmin): 55.0 bpm (Δ1d: +3.8%, Δ7d: +5.8%)
  Recovery Score (Whoop): 62.0 % (Δ1d: -24.4%, Δ7d: +19.2%)
  Sleep Score (Whoop): 72.0 % (Δ1d: -12.2%, Δ7d: -10.0%)
  Body Battery (Garmin): 37.2 % (Δ1d: +7.9%, Δ7d: -16.6%)
  Avg Stress (Garmin, Prev Day): n/a
  Sleep Duration (Whoop): 7.2 h (Δ1d: -17.0%, Δ7d: -8.3%)
  Sleep Consistency (Whoop): 48.0 % (Δ1d: -29.4%, Δ7d: -12.7%)
  Sleep Disturbances/hr (Whoop): 0.7  (Δ1d: -44.5%, Δ7d: -40.3%)
  Strain Score (Whoop): 12.8  (Δ1d: +2.3%, Δ7d: -7.1%)
  Skin Temp (Whoop): 93.7 °F (Δ1d: +187.4%, Δ7d: +180.9%)
  Resp Rate (Whoop): 16.5 rpm (Δ1d: -2.1%, Δ7d: -0.8%)
  Steps (Garmin): 4756.0  (Δ1d: -30.0%, Δ7d: -64.4%)
  Total Activity (Garmin): 62.3 min (Δ1d: -42.6%, Δ7d: -59.0%)
  Resp Rate (Garmin): 14.9 rpm (Δ1d: +0.1%, Δ7d: +1.6%)
  VO2max (Garmin): n/a

2. Continue with the standard audit: compare planned vs. actual, surface key metrics, inconsistencies, and actionable steps as usual.

3. Extract **planned workout targets** from `cultivation/outputs/training_plans/baseox_daily_plans/week20/Week20_Thu_2025-05-15_10-20-30_WORKOUT.md`.

4. Load **zone definitions**:
   • Primary: Parse `cultivation/data/zones_personal.yml` for `model`, HR & pace breakpoints. (Assuming AeT-anchored model with Z2: 145-160bpm as per `pace-zones.md` and values in `advanced_metrics.txt`)
   • Cross-verify with `cultivation/outputs/training_plans/pace-zones.md`; flag discrepancies as critical.

5. Parse **actual performance** from `cultivation/outputs/figures/week20/20250515_142115_morning_10_20_30attempt_Z3Z5_high_decoup21p_heavy_walk_warm26C/txt/`.

6. Generate the analysis report using house layout.

STYLE & CONSTRAINTS
-------------------
* Begin with “🧙🏾‍♂️:” (alignment), then “🏃‍♂️:” (main body)
* End with a clarifying question or suggested next step
* No invented values—quote only from actual files; use “n/a” if missing
* Default HRmax = 201 unless overridden in YAML (Using zones from `advanced_metrics.txt`)
* Treat “pause?” tags as walk unless stated otherwise
* Flag ZONE_YAML `model` ≠ ZONE_DOC_PATH model as critical error

OUTPUT FORMAT
-------------
As described above in step 6—strict layout, human-readable, CI-ready.

🏃‍♂️:
**Analysis of Pre-Run Wellness Context (2025-05-15):**

*   **HRV (Whoop): 90.5 ms (Δ1d: -14.5%, Δ7d: +3.3%)**
    *   Value: 90.5 ms is a decent absolute value, often indicating good readiness.
    *   Trend: Significant drop of -14.5% from the previous day, which is a notable concern and often suggests increased physiological stress or incomplete recovery. The 7-day trend is slightly positive (+3.3%), suggesting overall stability over the week but the acute drop is more pertinent for today's session.
    *   Risk: The sharp 1-day drop flags a potential risk for performing high-intensity work. The body might not be optimally prepared to handle and adapt to intense stress.

*   **RHR (Whoop): 59.0 bpm (Δ1d: +1.7%, Δ7d: +5.4%)**
    *   Value: 59 bpm is relatively low, generally good.
    *   Trend: Slight increase from yesterday (+1.7%) and a more notable increase over 7 days (+5.4%). An increasing RHR, especially over 7 days, can be an early indicator of accumulating fatigue or an impending illness.
    *   Risk: The upward trend, though small for 1-day, combined with the HRV drop, reinforces caution.

*   **RHR (Garmin): 55.0 bpm (Δ1d: +3.8%, Δ7d: +5.8%)**
    *   Value: 55 bpm is also a good low value. Garmin RHR is consistently lower than Whoop RHR in this dataset.
    *   Trend: Larger 1-day increase (+3.8%) than Whoop, and a similar 7-day increase (+5.8%).
    *   Comparison: Both sources show an upward trend in RHR, with Garmin showing a more pronounced daily increase. This consistency across devices strengthens the signal of potentially increased physiological load.

*   **Recovery Score (Whoop): 62.0 % (Δ1d: -24.4%, Δ7d: +19.2%)**
    *   Value: 62% is in the "yellow" or moderate recovery zone for Whoop, suggesting some level of preparedness but not optimal.
    *   Trend: Significant drop of -24.4% from yesterday, aligning with the HRV drop. The 7-day trend shows improvement but is overshadowed by the acute decrease.
    *   Risk: Indicates that the body is not fully recovered, making high-intensity efforts potentially more taxing and less productive.

*   **Sleep Score (Whoop): 72.0 % (Δ1d: -12.2%, Δ7d: -10.0%)**
    *   Value: 72% suggests sleep was somewhat adequate but not ideal.
    *   Trend: Decreases both daily (-12.2%) and weekly (-10.0%).
    *   Risk: Suboptimal sleep can impair recovery, cognitive function, and physical performance. This downward trend is a concern for readiness.

*   **Body Battery (Garmin): 37.2 % (Δ1d: +7.9%, Δ7d: -16.6%)**
    *   Value: 37.2% is quite low, indicating limited energy reserves according to Garmin's algorithm.
    *   Trend: A slight increase from yesterday (+7.9%) might suggest a very slight rebound from an even lower point, but the 7-day trend shows a significant drain (-16.6%).
    *   Risk: Low Body Battery suggests reduced capacity to handle stress, including a hard workout.

*   **Sleep Duration (Whoop): 7.2 h (Δ1d: -17.0%, Δ7d: -8.3%)**
    *   Value: 7.2 hours is below the generally recommended 7-9 hours for adults, especially athletes.
    *   Trend: Significant decrease from yesterday (-17.0%) and a decrease over the week (-8.3%).
    *   Risk: Shorter sleep duration directly impacts recovery.

*   **Sleep Consistency (Whoop): 48.0 % (Δ1d: -29.4%, Δ7d: -12.7%)**
    *   Value: 48% is very low, indicating irregular sleep patterns.
    *   Trend: Large drop from yesterday and a negative weekly trend.
    *   Risk: Poor sleep consistency disrupts circadian rhythms and impairs recovery processes. This is a significant flag.

*   **Sleep Disturbances/hr (Whoop): 0.7 (Δ1d: -44.5%, Δ7d: -40.3%)**
    *   Value: 0.7 disturbances per hour is relatively low, which is positive.
    *   Trend: Significant improvement (decrease) both daily and weekly.
    *   Improvement: This is a positive sign amidst other negative trends, suggesting sleep quality, when asleep, might have been better in terms of interruptions.

*   **Strain Score (Whoop): 12.8 (Δ1d: +2.3%, Δ7d: -7.1%)**
    *   Value: 12.8 represents a moderate level of daily strain experienced on the previous day.
    *   Trend: Slightly higher than the day before, but lower over the week.
    *   Context: This strain likely contributed to the current recovery state.

*   **Skin Temp (Whoop): 93.7 °F (Δ1d: +187.4%, Δ7d: +180.9%)**
    *   Value: This value seems unusually low for skin temperature if the previous value was in a normal human range (e.g., around 33-35°C or 91-95°F). The percentage changes are exceptionally high.
    *   Trend: Massive increase.
    *   Risk: Such a drastic change is highly suspect and likely indicates a data error, sensor issue, or a misinterpretation of the previous day's data (e.g. if the previous day was ~32°F due to a bug). If accurate and previous was normal, this would be a significant fever indicator. **This metric needs careful validation.** Assuming it's an error, otherwise it would be a major red flag. For this analysis, it will be treated as a likely data anomaly.

*   **Resp Rate (Whoop): 16.5 rpm (Δ1d: -2.1%, Δ7d: -0.8%)**
    *   Value: 16.5 rpm is within a normal range.
    *   Trend: Slight decreases, generally stable.
    *   No major risk indicated.

*   **Steps (Garmin): 4756.0 (Δ1d: -30.0%, Δ7d: -64.4%)**
    *   Value: Low step count for the previous day.
    *   Trend: Significantly lower activity levels.
    *   Context: Might indicate a rest day or very low activity, which *should* aid recovery, but other metrics don't align with this.

*   **Total Activity (Garmin): 62.3 min (Δ1d: -42.6%, Δ7d: -59.0%)**
    *   Value: Low total activity.
    *   Trend: Similar to steps, significantly lower.
    *   Context: Reinforces low physical activity on the preceding day.

*   **Resp Rate (Garmin): 14.9 rpm (Δ1d: +0.1%, Δ7d: +1.6%)**
    *   Value: 14.9 rpm is normal.
    *   Trend: Stable. Garmin's respiratory rate is slightly lower than Whoop's but both are in a plausible range.

**Overall Wellness Impression:**
The pre-run wellness data on 2025-05-15 indicates a **compromised state of readiness**. Key negative indicators include:
*   Significant 1-day drop in HRV and Whoop Recovery Score.
*   Increasing RHR trend (both Whoop and Garmin).
*   Poor sleep metrics: score, duration, and especially consistency are all down.
*   Very low Garmin Body Battery.
*   The anomalous skin temperature reading needs investigation but is likely an error.

Despite low physical activity on the previous day (steps, total activity), the physiological markers point towards poor recovery. This suggests other stressors (mental, poor sleep timing, etc.) might be at play.
Proceeding with a high-intensity workout like the 10-20-30s carries a **significant risk** of poor performance, excessive physiological stress, and potentially delayed recovery or even injury. An "Amber Light" or even "Red Light" as per the training plan's wellness-based adjustments would be strongly indicated.

---
**0 · TL;DR**
The athlete attempted a 10-20-30 workout despite significant pre-run wellness flags indicating poor recovery (low HRV, Recovery Score, sleep metrics; high RHR trend). The "run" portion showed very high HR, low EF, and extremely high decoupling (21%), reflecting the poor readiness. The session deviated significantly from plan due to fatigue and likely the hot weather (30°C), resulting in extensive walking and an abandoned workout structure.

**1 · KPI Dashboard**

| Metric                      | Plan (10-20-30 Workout)                                      | Actual (Run Segments)        | Actual (Full Session)      | Status      |
| :-------------------------- | :----------------------------------------------------------- | :--------------------------- | :------------------------- | :---------- |
| **Duration (Total)**        | 30-35 min                                                    | 14.1 min (run only)          | 36.3 min                   | ⚠️ / ❌     |
| **Workout Structure**       | 2 sets of [5x(30s jog, 20s mod, 10s sprint)], 2min recovery b/w sets | 5 detected "strides"         | Heavily modified           | ❌          |
| **Avg HR (Run portions)**   | Z3-Z4 (mod), Z5 (sprint)                                     | 172.7 bpm (mostly Z3/Z4)     | 158.7 bpm                  | ❌          |
| **% Time in HR Zones (Run)**| Target Z3-Z5 for efforts                                     | Z1: 5%, Z2: 40%, Z3: 22%, Z4: 21%, Z5: 11% | (see full session)         | ⚠️          |
| **HR Drift (Run portion)**  | Low expected for interval structure                          | +3.89% (HR)                  | n/a                        | ❌          |
| **Decoupling (Run portion)**| <10% ideally                                                 | **20.99%**                   | 6.18% (full, incl. walks)  | ❌          |
| **Efficiency Factor (Run)** | >0.018 target (dynamic for intensity)                        | **0.01679**                  | 0.01278 (full)             | ❌          |
| **Avg Cadence (Run portion)** | 165-170 spm (general Wk20 goal)                              | 165.4 spm (±8.6)             | 130.4 spm (full)           | ✅ (Run)    |
| **Walk Ratio**              | Minimal (warmup/cooldown, recovery jogs)                     | 0% (by definition)           | **55.9%** (20m18s)         | ❌          |
| **hrTSS (Run portion)**     | Moderate (for 10 min of work)                                | 22.7                         | 45.7 (full)                | ⚠️          |
| **Fatigue Flags**           | Should be Green Light to attempt                             | Multiple pre-run red flags   | Multiple pre-run red flags | ❌          |

**2 · Root-Cause / Consistency Notes**
*   **Pre-Run Wellness:** Major red flags (HRV -14.5% Δ1d, Recovery Score -24.4% Δ1d, poor sleep) indicated athlete was not recovered for intensity. The plan explicitly states "SKIP THIS WORKOUT" for a Red Light day.
*   **Environmental Conditions:** `weather.txt` reports 30.0°C, "Clear sky." High heat would significantly increase physiological strain and HR for any given RPE/pace.
*   **Zone Definitions:** `advanced_metrics.txt` shows zones used: Z1 [0, 145], Z2 [145, 160], Z3 [161, 175], Z4 [176, 186], Z5 [187, 201]. This aligns with the "AeT-anchored" model in `pace-zones.md` (HRmax 201). No critical discrepancy.
*   **File Consistency:**
    *   `run_summary.txt` (adv. metrics section) and `advanced_metrics.txt` are identical for run portion.
    *   `run_only_summary.txt` is also consistent for the run portion.
    *   `session_full_summary.txt` provides metrics for the entire 36.3 min activity, including extensive walks. Decoupling here (6.18%) is misleading due to high walk percentage.
    *   `power_distribution.txt` is empty: "count 0.0, mean NaN..." indicating no power data was recorded or processed.
*   **Workout Execution:** The `stride_summary.txt` detected 5 strides, which were likely the 10s sprint attempts. Durations (15-17s) are slightly longer than planned 10s. Avg HR during these is very high (186-191bpm), aligning with Z4/Z5 effort. However, the overall structure was not maintained.

**3 · Target vs Actual**

*   **Session Intent (from `Week20_Thu_2025-05-15_10-20-30_WORKOUT.md`):**
    *   **Type:** 10-20-30 Intensity Workout.
    *   **Purpose:** High-intensity stimulus for VO2max, speed, and running economy with low volume.
    *   **Planned Duration:** 30-35 min total.
    *   **Warm-up:** 10-12 min very easy jog + strides.
    *   **Main Workout:** 2 SETS of [5 repetitions of (30s very easy jog RPE 2-3/10 - 20s moderate pace RPE 5-6/10 Z3/Z4 - 10s near-maximal sprint RPE 9/10 Z5)]. 2 min very slow jog/walk recovery between sets.
    *   **Cool-down:** 10-12 min very easy jog/walk.
    *   **Wellness Check:** MANDATORY, with specific adjustments: ❤️ Red Light = SKIP THIS WORKOUT.

*   **Actual Execution & Deviations:**
    *   **Wellness:** Athlete was clearly in a ❤️ Red Light state based on pre-run metrics. The workout should have been skipped.
    *   **Warm-up/Cooldown:** Not clearly discernible from data; `walk_over_time.txt` shows extensive walking from the start. The total "run" portion was only 14.1 minutes.
    *   **Main Workout:**
        *   The structure of 2 sets of 5 reps was not completed. Only 5 "strides" (likely the 10s efforts) were detected by `stride_summary.txt`.
        *   Average HR for run segments was 172.7 bpm, with significant time in Z3 (22%) and Z4 (21%), and Z5 (11%) as per `time_in_hr_zone.txt`. This is very high for what should have included easy jog portions.
        *   The "30s very easy jog" and "20s moderate pace" components are not clearly distinguishable and likely morphed into general high-effort running or walking due to fatigue and heat.
        *   `walk_summary.txt` shows 17 walk segments, totaling 20 min 18 s (55.9% of session). Avg walk HR was very high at 159.7 bpm. This indicates the "very easy jog" portions of the 30-20-10 were mostly walking at a high HR.
    *   **Pacing:** `pace_over_time.txt` shows a "negative" split for the *run portions*, but this is likely due to walking more in the first half of *running time*.
    *   **Cadence (Run):** 165.4 spm (from `cadence_distribution.txt`) is good and meets the Wk20 target of 165-170 spm during run segments.
    *   **Decoupling (Run):** 20.99% (from `advanced_metrics.txt`) is extremely high, indicating significant cardiac drift and inability to maintain intensity. This is a clear sign of overreaching for the session given the state of readiness and conditions.
    *   **Efficiency Factor (Run):** 0.01679 is low, especially for higher intensity segments, reflecting the high HR for paces achieved.
    *   **HR vs Pace Correlation:** -0.449 (from `hr_vs_pace_hexbin.txt`) shows a moderate negative correlation, as expected (faster pace, higher HR).

*   **Highlights:**
    *   Cadence during actual running segments was on target.
    *   Athlete did attempt some high-intensity efforts (strides detected).

*   **Limiters:**
    *   **Poor Pre-Run Readiness:** Overwhelmingly the primary limiter. All wellness indicators pointed to needing rest or very light activity.
    *   **High Environmental Temperature:** 30.0°C is very warm for intense running and would have exacerbated the physiological strain from poor readiness.
    *   **Workout Structure Abandoned:** The planned 10-20-30 structure was not maintained, likely due to the above factors. High proportion of walking at elevated HR.
    *   **Extremely High Decoupling:** Shows a significant mismatch between intended effort and bodily response.

*   **Analysis of Other Files:**
    *   `hr_distribution.txt`: Mean HR 172.7, median 177 bpm for run portions.
    *   `pace_distribution.txt`: Mean pace 6.19 min/km for run portions (skewed by efforts and slow recovery jogs).
    *   `time_in_pace_zone.txt` (run portions): Z1: 25%, Z2: 3%, Z3: 34%, Z4: 23%, Z5: 15%. Pace zones show more time in faster zones than HR zones, which is consistent with high decoupling (HR too high for pace).
    *   `time_in_effective_zone.txt`: Only 5% in Z1, 0.7% in Z2. 76% "mixed". Highlights difficulty in matching HR and pace zones.
    *   `time_in_fatigue_kpi_zone.txt`: 40% Aerobic (Z2), 33% High Intensity (Z4+), 22% Threshold (Z3). Suggests high overall internal load during run periods.
    *   `walk_hr_distribution.txt` & `walk_pace_distribution.txt` & `walk_over_time.txt`: Confirm high HR during walks (many data points >150-160bpm) and slow walking paces, indicating significant fatigue even during walk recoveries.

**4 · Action Plan**
1.  **Prioritize Recovery:** Athlete needs significant rest. Subsequent 1-2 days should be full rest or extremely light active recovery (e.g., short Z1 walk IF wellness shows marked improvement).
2.  **Strict Adherence to Wellness-Based Adjustments:** Re-emphasize the "Red Light = SKIP/REST" rule from the training plan, especially for intensity.
3.  **Log Session Subjectives:** Add subjective notes for this run: RPE for efforts, how the body felt, reasons for modification/abandonment. This context is crucial.
4.  **Review Heat Acclimatization Strategy:** If high temperatures persist, adjust planned intensities and durations significantly downwards or shift workout times to cooler parts of the day.
5.  **Investigate Skin Temperature Anomaly:** Check Whoop data/sensor for the skin temperature reading. If it's a persistent issue, contact Whoop support.
6.  **Future 10-20-30 Attempts:** Only attempt when wellness metrics are green across the board and environmental conditions are more favorable. Consider starting with 1 set instead of 2 if new to the workout or conditions are marginal.

**5 · Integration Hooks**

| Action                                      | Impacted Files/Workflows                                                                                                |
| :------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------- |
| 1. Prioritize Recovery                      | `cultivation/outputs/training_plans/baseox_daily_plans/week20/Week20_Fri_2025-05-16_ACTIVE_RECOVERY.md` (adjust to full rest if needed), Future daily plans |
| 2. Strict Adherence to Wellness Adjustments | `cultivation/docs/4_analysis/operational_playbook.md` (reinforce protocol), Daily wellness check process                 |
| 3. Log Session Subjectives                  | `cultivation/data/subjective.csv` (add entry for this run), `RUN_DIR/session_notes_...md` (create this file)            |
| 4. Review Heat Acclimatization              | Future session planning, `cultivation/docs/3_design/*_block.md` (consider adding heat guidelines)                     |
| 5. Investigate Skin Temp Anomaly            | Whoop app/dashboard, potentially `scripts/utilities/habitdash_api.py` if data pulled from there                         |
| 6. Future 10-20-30 Attempts               | `cultivation/outputs/training_plans/baseox_daily_plans/week20/Week20_Thu_2025-05-15_10-20-30_WORKOUT.md` (re-evaluation) |

**6 · Appendices**

<details>
<summary>Zone Definitions (from advanced_metrics.txt, assumed AeT-anchored HRmax 201)</summary>

```
Z1 (Recovery): {'bpm': [0, 145], 'pace_min_per_km': [7.0, 999]}
Z2 (Aerobic): {'bpm': [145, 160], 'pace_min_per_km': [6.85, 8.7]}
Z3 (Tempo): {'bpm': [161, 175], 'pace_min_per_km': [5.2, 6.85]}
Z4 (Threshold): {'bpm': [176, 186], 'pace_min_per_km': [4.8, 5.2]}
Z5 (VO2max): {'bpm': [187, 201], 'pace_min_per_km': [0.0, 4.8]}
```
</details>

<details>
<summary>Stride Summary (10s Sprint Attempts)</summary>

```
Stride 1: 2025-05-15 14:32:10+00:00 to 2025-05-15 14:32:27+00:00, duration: 17.0s, avg pace: 3.17, avg HR: 186.7
Stride 2: 2025-05-15 14:35:16+00:00 to 2025-05-15 14:35:31+00:00, duration: 15.0s, avg pace: 3.75, avg HR: 187.2
Stride 3: 2025-05-15 14:38:21+00:00 to 2025-05-15 14:38:36+00:00, duration: 15.0s, avg pace: 3.72, avg HR: 189.8
Stride 4: 2025-05-15 14:42:15+00:00 to 2025-05-15 14:42:31+00:00, duration: 16.0s, avg pace: 3.29, avg HR: 191.0
Stride 5: 2025-05-15 14:45:18+00:00 to 2025-05-15 14:45:33+00:00, duration: 15.0s, avg pace: 3.63, avg HR: 190.9
```
</details>

This session was a clear instance of pushing through significant warning signs from the body and environment, leading to a largely unproductive and potentially detrimental workout. The key learning is to respect the wellness data and adjust plans accordingly. What is the plan for the next 2-3 days to ensure adequate recovery from this effort?