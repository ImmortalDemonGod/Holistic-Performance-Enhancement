🧙🏾‍♂️: We are aligning on a systematic audit of a Cultivation running session, specifically the long run from Week 21, and comparing it to the long run from Week 19.

INPUT METADATA
--------------
RUN_LABEL: "20250524_164032_W21Sat_LongRun_Hot31C_HR160Drift9p_EF0161_Walk31p"
BLOCK_SPEC_PATH: "cultivation/outputs/training_plans/baseox_daily_plans/week21/week21_assessment_training_plan.md" (Context: Week 21 is an assessment week. While a specific Z2 run was planned for Saturday, this was executed as a general "long run".)
ZONE_YAML_PATH:  "cultivation/data/zones_personal.yml"
ZONE_DOC_PATH:   "cultivation/outputs/training_plans/pace-zones.md"
RUN_DIR:         "cultivation/outputs/figures/week21/20250524_164032_W21Sat_LongRun_Hot31C_HR160Drift9p_EF0161_Walk31p/"
RAW_FILE_MAP:
```
txt/
├── advanced_metrics.txt
├── cadence_distribution.txt
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
1.  Analyze the Pre-Run Wellness Context.
2.  Conduct a standard audit of the Week 21 long run.
3.  Compare the Week 21 long run to the Week 19 long run.

🏃‍♂️:

ANALYSIS SCOPE
--------------
For this session, all files in `cultivation/outputs/figures/week21/20250524_164032_W21Sat_LongRun_Hot31C_HR160Drift9p_EF0161_Walk31p/txt/` will be systematically analyzed.

**0 · TL;DR**
The Week 21 long run was a challenging ~89-minute effort in very hot conditions (31.5°C), showing moderate aerobic efficiency (EF 0.01611) and significant HR drift (8.69%). While cadence was good (158 spm), the overall performance reflects the environmental stress and possibly accumulated fatigue from an assessment week, differing significantly from the planned shorter Z2 controlled run. Compared to the Week 19 long run, this session was hotter, had higher average HR, slightly better EF, much higher decoupling, and slightly better cadence, but covered less distance in a shorter run-only duration.

**1 · Pre-Run Wellness Context Analysis (W21 Long Run: 2025-05-24)**

Data from `cultivation/outputs/figures/week21/20250524_164032_W21Sat_LongRun_Hot31C_HR160Drift9p_EF0161_Walk31p/txt/run_summary.txt`:
--- Pre-Run Wellness Context (Data for 2025-05-24) ---
  HRV (Whoop): 111.7 ms (Δ1d: +26.5%, Δ7d: -0.2%)
    *   **Value & Trend:** Excellent HRV value, showing a significant positive jump from the previous day (+26.5%) and stable over 7 days. This is a strong indicator of good autonomic nervous system recovery.
    *   **Impact:** Suggests good readiness for a demanding session.

  RHR (Whoop): 51.0 bpm (Δ1d: -15.0%, Δ7d: -3.8%)
    *   **Value & Trend:** Low RHR, with a substantial drop from the previous day (-15.0%) and lower than the weekly average. This aligns with the high HRV, indicating good cardiovascular recovery.
    *   **Impact:** Positive sign for readiness.

  RHR (Garmin): 48.0 bpm (Δ1d: -7.7%, Δ7d: -4.0%)
    *   **Value & Trend:** Very low Garmin RHR, also showing a good drop from the previous day and lower than the weekly average.
    *   **Comparison:** Both Whoop and Garmin RHR are low and show similar positive trends, reinforcing the recovery picture. Garmin RHR is typically lower than Whoop's.
    *   **Impact:** Reinforces good readiness.

  Recovery Score (Whoop): 91.0 % (Δ1d: +78.4%, Δ7d: +2.2%)
    *   **Value & Trend:** Excellent Whoop Recovery Score, a very large jump from the previous day and slightly above the weekly average. This is a "Green" recovery state.
    *   **Impact:** Indicates prime physiological readiness for a hard effort.

  Sleep Score (Whoop): 78.0 % (Δ1d: -2.5%, Δ7d: -4.9%)
    *   **Value & Trend:** Good sleep score, though a slight dip from the previous day and weekly average. Still in a decent range.
    *   **Impact:** Sufficient sleep for good performance, though not perfect.

  Body Battery (Garmin): 74.2 % (Δ1d: +54.7%, Δ7d: +55.0%)
    *   **Value & Trend:** Good Body Battery, showing a very significant increase from the previous day and the weekly average.
    *   **Impact:** Suggests high energy reserves and good readiness.

  Avg Stress (Garmin, Prev Day): n/a
    *   **Impact:** Missing data; cannot assess previous day's stress impact.

  Sleep Duration (Whoop): 7.8 h (Δ1d: -20.7%, Δ7d: +14.0%)
    *   **Value & Trend:** Good sleep duration, though a notable decrease from the previous day (-20.7%). However, it's significantly above the 7-day average.
    *   **Impact:** Generally positive, but the shorter sleep compared to the day before might be a minor factor if the previous day's sleep was exceptionally long/restorative.

  Sleep Consistency (Whoop): 37.0 % (Δ1d: -37.3%, Δ7d: -43.9%)
    *   **Value & Trend:** Very poor sleep consistency. This is a significant drop from both the previous day and the weekly average.
    *   **Impact:** This is a key negative flag. Poor sleep consistency can undermine recovery even if total duration is adequate. Could lead to feeling less than optimal despite other good metrics.

  Sleep Disturbances/hr (Whoop): 1.0  (Δ1d: +194.4%, Δ7d: +8.8%)
    *   **Value & Trend:** Low number of disturbances, but a very large percentage increase from the previous day. The absolute value is still good.
    *   **Impact:** Slight concern due to the increase, but 1.0/hr is generally good.

  Strain Score (Whoop): n/a (Likely for the day of the run, pre-activity)
    *   **Impact:** Not available for pre-run assessment.

  Skin Temp (Whoop): 93.0 °F (Δ1d: +178.6%, Δ7d: +181.0%)
    *   **Note:** This seems like an data error or a significant unit/scale issue in the provided summary (Fahrenheit for skin temp usually much lower, or this is a large deviation if Celsius). Assuming it means a deviation: "+1.786% and +1.810%" from baseline if it's typically around 93.0. If it's raw value, it's very high. The percentages are likely relative to a baseline not shown, or there's a data anomaly. *Given the large percentage change, it's treated as a deviation above baseline.*
    *   **Impact:** Elevated skin temperature can be an early sign of physiological stress or impending illness. This is a potential risk factor.

  Resp Rate (Whoop): 16.8 rpm (Δ1d: -3.7%, Δ7d: -0.7%)
    *   **Value & Trend:** Normal respiratory rate, slightly down from the previous day and stable over the week.
    *   **Impact:** No concerns.

  Steps (Garmin): 20407.0  (Δ1d: +300%+, Δ7d: +83.5%)
    *   **Value & Trend:** High step count for the day already, significantly above daily and weekly averages.
    *   **Impact:** Indicates a very active day *before* the long run, which could mean pre-fatigue. This is a notable factor.

  Total Activity (Garmin): 195.3 min (Δ1d: +147.5%, Δ7d: +33.2%)
    *   **Value & Trend:** High total activity minutes for the day, also significantly above averages.
    *   **Impact:** Aligns with high steps, suggesting potential pre-fatigue heading into the long run.

  Resp Rate (Garmin): 14.7 rpm (Δ1d: +1.3%, Δ7d: -8.5%)
    *   **Value & Trend:** Normal Garmin respiratory rate.
    *   **Impact:** No concerns.

  VO2max (Garmin): 54.0 ml/kg/min (Δ1d: n/a, Δ7d: +0.0%)
    *   **Value & Trend:** Stable VO2max.
    *   **Impact:** Neutral.

**Overall Wellness Summary for W21 Long Run:**
While many metrics (HRV, RHR, Recovery Score, Body Battery) pointed to excellent physiological readiness, there were conflicting signals: very poor sleep consistency, a potential flag on skin temperature, and very high activity levels (steps, total activity minutes) *before* the run. This suggests that despite good "Green" recovery numbers, the athlete might not have been optimally rested due to inconsistent sleep patterns and significant prior activity on the day of the run.

**2 · KPI Dashboard (W21 Long Run: `20250524_164032_W21Sat_LongRun_Hot31C_HR160Drift9p_EF0161_Walk31p`)**

| Metric                     | Actual (Run-Only)         | Target (General Long Run) | Status | Notes                                                                 |
| :------------------------- | :------------------------ | :------------------------ | :----- | :-------------------------------------------------------------------- |
| **Duration (Run-Only)**    | 89.0 min                  | ~90-120 min (Typical LR)  | ✅     | Solid duration for a long run.                                        |
| **Avg HR (Run-Only)**      | 159.6 bpm                 | Z2 (145-160bpm)           | ⚠️     | Slightly above top of Z2, avg close to 160bpm cap.                    |
| **% in Z2 HR (Run-Only)**  | Z1:40.2%, **Z2:37.2%**, Z3:13.5%, Z4:5.1%, Z5:4.1% | Predominantly Z2          | ⚠️     | Significant time in Z1, and also Z3+ spillover.                       |
| **HR Drift (PwvHR %)**     | 8.69%                     | < 5%                      | ❌     | High drift, indicating fatigue or heat impact.                        |
| **Efficiency Factor (EF)** | 0.01611                   | > 0.0180                  | ❌     | Below desired aerobic baseline, especially for Z2/low Z3 HR.          |
| **Avg Cadence (Run-Only)** | 158.3 ± 4.9 spm           | ~160-170 spm              | ⚠️     | Lower end of target range. SD is reasonable.                          |
| **Walk Ratio (Session)**   | 30.8% (45.6 min / 148.1 min total) | < 20-25% (typical)    | ⚠️     | High walk ratio, likely due to heat and effort.                       |
| **hrTSS (Run-Only)**       | 114.1                     | Varies                    | N/A    | Reflects a demanding session.                                         |
| **Pacing Strategy**        | Negative Split            | Even/Negative             | ✅     | `pace_over_time.txt`: First half: 7.05 min/km, Second half: 6.25 min/km |
| **Fatigue Flags**          | High HR Drift, Low EF     | Minimal                   | ❌     | Significant flags present.                                            |

**3 · Root-Cause / Consistency Notes (W21 Long Run)**
*   **Planned vs. Actual Discrepancy:** The W21 Assessment plan for Saturday was a "Controlled Zone 2 HR Run" of 25-35 minutes with strict HR (145-155bpm) and cadence (165-170spm) targets. The executed run (`20250524_164032_W21Sat_LongRun_Hot31C_HR160Drift9p_EF0161_Walk31p`) was an 89-minute run-only effort with avg HR 159.6bpm and avg cadence 158.3spm. This was a significantly different session.
*   **Zone Definitions:**
    *   `cultivation/data/zones_personal.yml`: Uses HRmax 201. Z2: 145-160 bpm.
    *   `cultivation/outputs/training_plans/pace-zones.md`: Aligns with YAML for HR zones.
    *   No discrepancies found in zone definitions.
*   **Environmental Impact:** `weather.txt` shows "Temperature: 31.5 °C, Description: Clear sky". This is extremely hot and would significantly impact HR, RPE, and ability to maintain pace/efficiency. The high walk ratio and HR drift are consistent with running in such heat.
*   **Data Gaps:** `power_distribution.txt` shows "count    0.0", indicating no power data was recorded or processed for this run.

**4 · Target vs Actual (W21 Long Run)**

*   **Session Intent (Inferred for a general "Long Run" in an assessment week):**
    *   Build/test aerobic endurance and durability.
    *   Maintain a controlled effort, ideally in Zone 2 HR.
    *   Practice pacing and fueling for longer distances.
    *   Assess current long-run capability after recent training/assessments.

*   **Execution Analysis based on File Data:**
    *   **Overall Effort & Duration (`session_full_summary.txt`, `run_only_summary.txt`):**
        *   Total session duration: 148.1 min (2h 28m). Run-only duration: 89.0 min.
        *   Total distance: 17.72 km. Run-only distance: 13.73 km.
        *   This is a substantial duration, especially in heat.
    *   **Pacing (`pace_distribution.txt`, `pace_over_time.txt`):**
        *   Run-only avg pace: 6.48 min/km (approx 10:26 min/mile).
        *   Pacing strategy was negative (First half: 7.05 min/km, Second half: 6.25 min/km), which is positive.
    *   **Heart Rate (`hr_distribution.txt`, `time_in_hr_zone.txt`, `hr_over_time_drift.txt`):**
        *   Run-only avg HR: 159.6 bpm (top end of Z2/borderline Z3). Max HR: 193 bpm (Z5).
        *   HR Drift: 8.69% (High). First half HR: 155.62 bpm, Second half HR: 169.14 bpm. This shows significant cardiovascular strain increase.
        *   Time in HR Zones (run-only): Z1: 40.2%, Z2: 37.2%, Z3: 13.5%, Z4: 5.1%, Z5: 4.1%. Less than half the running time was in the target Z2. The significant Z1 time might be very slow recovery jogs between efforts or at the start/end. The Z3+ time indicates higher intensity segments.
    *   **Efficiency (`advanced_metrics.txt`):**
        *   Efficiency Factor (EF): 0.01611 (Below target >0.0180). This is low for an average HR around the Z2/Z3 border.
        *   hrTSS: 114.1 (High, indicating significant training stress).
    *   **Cadence (`cadence_distribution.txt`):**
        *   Run-only avg cadence: 158.3 spm (Mean), std: 4.86 spm. Min: 140, Max: 232 (max likely an anomaly/stride). 50% of data between 156-160 spm. This is on the lower side of a 160-170spm target.
    *   **Walks (`walk_summary.txt`, `walk_hr_distribution.txt`, `walk_pace_distribution.txt`, `walk_over_time.txt`):**
        *   Total walk time: 45 min 37 s (30.8% of session). 60 walk segments.
        *   Avg walk HR: 146.3 bpm (High, indicating walks were not providing full recovery, HR often remained in Z2).
        *   Walks were frequent, consistent with managing effort in severe heat. `walk_over_time.txt` shows walk data points.
    *   **Strides (`stride_summary.txt`):**
        *   4 strides detected, varying in pace and HR. Durations: 8s, 13s, 10s, 18s. Avg paces from 2.93 min/km to 4.21 min/km. HRs during strides went up to 189.3 bpm.
    *   **Correlation (`hr_vs_pace_hexbin.txt`):**
        *   Correlation coefficient between HR and Pace: -0.519. A moderate negative correlation, as expected (faster pace, higher HR).
    *   **Weather (`weather.txt`):**
        *   Temperature: 31.5°C. Clear sky. Wind: 21.1 km/h.
        *   **Limiter:** Extreme heat is a major factor explaining high HR, high drift, low EF, and high walk ratio.

*   **Highlights:**
    *   Completed a long duration run (89 min running) despite extreme heat.
    *   Achieved a negative split for pacing.
    *   Pre-run "Green" wellness metrics (HRV, RHR, Recovery Score) were excellent.
    *   Cadence was mostly maintained around 158 spm during running.

*   **Limiters:**
    *   **Extreme Heat (31.5°C):** Overriding factor impacting all performance metrics.
    *   **High HR Drift (8.69%):** Indicates significant fatigue accumulation or inability to maintain physiological efficiency, exacerbated by heat.
    *   **Low Efficiency Factor (0.01611):** Suggests poor aerobic economy under these conditions.
    *   **High Walk Ratio (30.8%):** Necessary for heat management but broke up continuous running.
    *   **Average HR (159.6bpm) at the top of Z2 / low Z3:** Higher than ideal for a typical Z2 long run, driven by heat.
    *   **Pre-run Contextual Flags:** Poor sleep consistency and high prior-day/same-day activity might have contributed to fatigue despite good core recovery metrics.
    *   **Deviation from W21 Saturday Plan:** This was not the planned controlled Z2 assessment.

**5 · Comparison: W21 Long Run vs. W19 Long Run (`20250510_...`)**

| Metric                        | W21 Long Run (2025-05-24) (Run-Only) | W19 Long Run (2025-05-10) (Run-Only) | Comparison Notes                                                                                                |
| :---------------------------- | :----------------------------------- | :----------------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| **Run-Only Duration**         | 89.0 min                             | 97.6 min                             | W19 was ~9 min longer.                                                                                          |
| **Run-Only Distance**         | 13.73 km                             | 14.1 km                              | W19 covered slightly more distance.                                                                             |
| **Avg Pace (Run-Only)**       | 6.48 min/km (10:26/mi)               | 6.93 min/km (11:09/mi)               | W21 was significantly faster on average during run segments.                                                     |
| **Avg HR (Run-Only)**         | 159.6 bpm                            | 153.6 bpm                            | W21 had a higher avg HR (borderline Z3 vs. mid-Z2 for W19).                                                     |
| **Max HR (Run-Only)**         | 193 bpm                              | 181 bpm                              | W21 reached a much higher Max HR.                                                                               |
| **Efficiency Factor (EF)**    | 0.01611                              | 0.01567                              | W21 EF slightly better, but both are low (<0.018). Improvement despite higher HR in W21 is interesting.         |
| **HR Drift / Decoupling (%)** | 8.69%                                | 0.15% (PwvHR from advanced_metrics)  | W21 had very high drift. W19 had exceptionally low decoupling (0.15% noted as likely artifact of high fragmentation from `week21_rpe10_benchmark_analysis.md`). The `hr_over_time_drift.txt` for W19 shows 0.59% drift (155.75 first half, 156.67 second half), which is still much lower than W21. |
| **Avg Cadence (Run-Only)**    | 158.3 ± 4.9 spm                      | 156.2 ± 4.3 spm                      | W21 cadence slightly higher and consistent.                                                                     |
| **hrTSS (Run-Only)**          | 114.1                                | 111.8                                | Similar training stress from running segments.                                                                  |
| **Walk Ratio (Full Session)** | 30.8% (45.6 min / 60 segments)       | 20.8% (29.3 min / 55 segments)       | W21 had a significantly higher walk ratio and more walk time.                                                   |
| **Avg Walk HR**               | 146.3 bpm                            | 148.6 bpm                            | Both high, indicating walks were not fully restful.                                                             |
| **Weather: Temp**             | 31.5 °C                              | 23.7 °C                              | W21 was substantially hotter, a key differentiator.                                                               |
| **Pacing Strategy**           | Negative (7.05 → 6.25 min/km)        | Negative (7.04 → 7.04 min/km, essentially even) | W21 showed a more pronounced negative split.                                                                  |
| **Pre-Run Wellness**          | Mixed (Good core, poor consistency/prior activity) | Poor (Mediocre Whoop Rec 50%, low Body Battery 41.6%) | W19 was on poorer overall wellness. W21 had better core recovery but other confounders.                 |

**Summary of Comparison:**
The W21 long run was performed in much hotter conditions than the W19 long run. Despite this, the W21 run segments were faster on average (6.48 vs 6.93 min/km) with a slightly higher EF (0.01611 vs 0.01567), but this came at the cost of a significantly higher average HR (159.6 vs 153.6 bpm) and much higher HR drift (8.69% vs ~0.6%). Cadence was slightly better in W21. The W21 run had a higher walk ratio, reflecting the harsher conditions. The W19 run, while on poorer overall wellness, was in cooler weather, allowing for a lower average HR and minimal drift during its (highly fragmented) run segments. The W21 run, despite some good pre-run recovery scores, was likely hampered by the heat, prior day activity, and poor sleep consistency, leading to a less efficient run than perhaps hoped for.

**6 · Action Plan**
1.  **Acknowledge Deviation:** Log that the W21 Saturday session was a general "long run" and not the planned "Controlled Zone 2 HR Assessment".
2.  **Heat Impact Analysis:** Future analysis of runs in >28-30°C should heavily weigh heat as a primary performance limiter and adjust expectations for EF, HR, and pace accordingly.
3.  **Wellness Consistency:** Emphasize the importance of sleep consistency, as highlighted by the W21 pre-run data. Even with good core recovery metrics, poor consistency can be a hidden drag.
4.  **Pre-Activity Load:** Consider the impact of high non-training activity (steps, general activity minutes) on the same day as key workouts. This might necessitate better planning of daily activity on hard training days.
5.  **Power Data:** Investigate why power data was missing for this W21 run.
6.  **Review Long Run Strategy in Heat:** Given the high walk ratio and HR drift, future long runs in extreme heat might benefit from a more structured run/walk strategy from the outset, or be significantly shortened / shifted to cooler times.

**7 · Integration Hooks**

| Action                                   | Impacted Files/Workflows                                                                 |
| :--------------------------------------- | :--------------------------------------------------------------------------------------- |
| Log W21 Sat. session deviation         | Session notes for `20250524_164032_W21Sat_LongRun_Hot31C_HR160Drift9p_EF0161_Walk31p`, Weekly Review docs.                        |
| Enhance heat impact considerations       | `running_analysis_template.md`, Future run planning.                                   |
| Track sleep consistency as key metric    | `daily_wellness.parquet` interpretation, `run_report.md` generation.                   |
| Monitor pre-workout activity load      | `daily_wellness.parquet` interpretation, `run_report.md` generation.                   |
| Debug power data for `20250524_164032_W21Sat_LongRun_Hot31C_HR160Drift9p_EF0161_Walk31p` | Garmin Connect export, `parse_run_files.py`.                                             |
| Refine LR strategy for extreme heat      | `long-distance_running.md`, Future training plan generation.                             |

**8 · Appendices**

<details>
<summary>Zone Definitions (cultivation/data/zones_personal.yml)</summary>

```yaml
# HRMax: 201
# Lactate Threshold HR (LTHR): Not explicitly defined, but Z4 is 176-186
# Resting HR (RHR): Varies, e.g. 48-51 for 2025-05-24

heart_rate:
  model: CogganExtended
  # Based on HR Max 201
  Z1 (Recovery): [0, 145] # Active Recovery <72% HRmax
  Z2 (Aerobic): [145, 160] # Endurance 72-80% HRmax (approx)
  Z3 (Tempo): [161, 175] # Tempo 80-87% HRmax (approx)
  Z4 (Threshold): [176, 186] # Threshold 87-93% HRmax (approx)
  Z5 (VO2max): [187, 201] # VO2 Max >93% HRmax

pace:
  # These are example paces and may need adjustment based on current fitness
  # Based on a hypothetical recent 5k time or VDOT
  Z1 (Recovery): ["7:00", "999:00"] # Very easy, conversational
  Z2 (Aerobic): ["6:52", "8:42"] # Easy, sustainable for long durations
  Z3 (Tempo): ["5:12", "6:52"] # Comfortably hard, "marathon pace" to "half-marathon pace"
  Z4 (Threshold): ["4:48", "5:12"] # Hard, "10k pace" to "threshold pace"
  Z5 (VO2max): ["0:00", "4:48"] # Very hard, "5k pace" and faster
```
</details>

<details>
<summary>Key File Data Points - W21 Long Run (`20250524_164032_W21Sat_LongRun_Hot31C_HR160Drift9p_EF0161_Walk31p`)</summary>

*   `advanced_metrics.txt`: `distance_km: 13.73, duration_min: 89.0, avg_pace_min_per_km: 6.48, avg_hr: 159.6, efficiency_factor: 0.01611, decoupling_%: 8.69, hrTSS: 114.1`
*   `cadence_distribution.txt`: `mean: 158.27, std: 4.86, 25%: 156.0, 50%: 158.0, 75%: 160.0`
*   `hr_distribution.txt`: `mean: 159.61, std: 13.86, max: 193.0, 25%: 152.0, 50%: 158.0, 75%: 164.0`
*   `hr_over_time_drift.txt`: `first_half_hr: 155.62, second_half_hr: 169.14, hr_drift_pct: 8.69`
*   `pace_distribution.txt`: `mean: 6.69, std: 1.11, 25%: 5.99, 50%: 6.79, 75%: 7.47` (for run-only points)
*   `pace_over_time.txt`: `first_half_pace: 7.049, second_half_pace: 6.252, strategy: "negative"`
*   `run_only_summary.txt`: Same as `advanced_metrics.txt`.
*   `run_summary.txt`: Contains pre-run wellness, weather. Avg Pace (total session, not run-only): 6.69 min/km.
*   `session_full_summary.txt`: `distance_km: 17.72, duration_min: 148.1, avg_pace_min_per_km: 8.36, avg_hr: 153.2, efficiency_factor: 0.01301, decoupling_%: 1.62, hrTSS: 168.4`
*   `stride_summary.txt`: 4 strides detected. E.g., Stride 4: `duration: 18.0s, avg pace: 3.98, avg HR: 189.3`
*   `time_in_hr_zone.txt` (Run-Only from main analysis): Z1:3531s (40.2%), Z2:3267s (37.2%), Z3:1184s (13.5%), Z4:445s (5.1%), Z5:358s (4.1%)
*   `time_in_pace_zone.txt` (Run-Only, based on default pace zones): Z1:3003s (34.6%), Z2:340s (3.9%), Z3:3914s (45.1%), Z4:1017s (11.7%), Z5:404s (4.7%)
*   `time_in_effective_zone.txt`: `mixed: 6836.0s (77.8%)` - High portion where HR and Pace zones didn't align.
*   `time_in_fatigue_kpi_zone.txt`: Aerobic (Z2 HR): 38.4%, High Intensity (Z4+ HR): 9.1%, Recovery (< Z2 HR): 39.0%, Threshold (Z3 HR): 13.5%.
*   `walk_summary.txt`: `Total walk time: 45 min 37 s (30.8 % of session), Avg walk HR (bpm): 146.3`
*   `weather.txt`: `Temperature: 31.5 °C, Description: Clear sky`
*   `power_distribution.txt`: `count 0.0` - No power data.
</details>

<details>
<summary>Key File Data Points - W19 Long Run (`20250510_203716_...`)</summary>

*   `advanced_metrics.txt`: `distance_km: 14.1, duration_min: 97.6, avg_pace_min_per_km: 6.93, avg_hr: 153.6, efficiency_factor: 0.01567, decoupling_%: 0.15, hrTSS: 111.8`
*   `cadence_distribution.txt`: `mean: 156.17, std: 4.29, 25%: 154.0, 50%: 156.0, 75%: 158.0`
*   `hr_distribution.txt`: `mean: 153.61, std: 8.96, max: 181.0, 25%: 149.0, 50%: 155.0, 75%: 159.0`
*   `hr_over_time_drift.txt`: `first_half_hr: 155.75, second_half_hr: 156.67, hr_drift_pct: 0.59`
*   `pace_distribution.txt`: `mean: 7.04, std: 0.89, 25%: 6.50, 50%: 7.00, 75%: 7.58` (for run-only points)
*   `pace_over_time.txt`: `first_half_pace: 7.0397, second_half_pace: 7.0394, strategy: "negative"` (effectively even)
*   `session_full_summary.txt`: `distance_km: 16.86, duration_min: 140.9, avg_pace_min_per_km: 8.36, avg_hr: 151.8, efficiency_factor: 0.01314, decoupling_%: 3.57, hrTSS: 155.8`
*   `stride_summary.txt`: `Stride segments detected: 1. duration: 13.0s, avg pace: 3.61, avg HR: 156.4`
*   `time_in_hr_zone.txt` (Run-Only): Z1:3088s (36.8%), Z2:4277s (50.9%), Z3:908s (10.8%), Z4:122s (1.5%)
*   `walk_summary.txt`: `Total walk time: 29 min 21 s (20.8 % of session), Avg walk HR (bpm): 148.6`
*   `weather.txt`: `Temperature: 23.7 °C, Description: Mainly clear`
*   `power_distribution.txt`: `count 0.0` - No power data.
</details>

This detailed analysis covers the W21 long run and its comparison to the W19 long run, fulfilling the request.
What specific aspect of the training or data would you like to delve into next? For example, we could further investigate the impact of heat on efficiency across multiple runs.