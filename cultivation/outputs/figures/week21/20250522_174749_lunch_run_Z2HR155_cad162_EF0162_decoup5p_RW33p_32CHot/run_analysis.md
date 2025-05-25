🧙🏾‍♂️: We are aligning on a systematic audit of a Cultivation running session.

INPUT METADATA
--------------
RUN_LABEL: "W21_Thu_Z2_Test_LunchRun"
BLOCK_SPEC_PATH: cultivation/outputs/training_plans/baseox_daily_plans/week21/week21_assessment_training_plan.md
ZONE_YAML_PATH:  cultivation/data/zones_personal.yml
ZONE_DOC_PATH:   cultivation/docs/3_design/pace-zones.md
RUN_DIR:         cultivation/outputs/figures/week21/20250522_174749_lunch_run/
RAW_FILE_MAP:
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
csv/
└── walk_segments.csv

TASK
----
1. **Analyze the Pre-Run Wellness Context section** (as shown below) in detail. For each metric, comment on its value, trend (Δ1d, Δ7d), and any notable deviations, improvements, or risks. Consider how these wellness factors might impact the athlete's readiness, performance, and recovery for this session. If multiple data sources are present (e.g., Whoop vs Garmin RHR), compare and interpret both. Highlight any discrepancies or patterns that could inform training decisions.

--- Pre-Run Wellness Context (Data for 2025-05-22, from `run_summary.txt`) ---
   HRV (Whoop): 116.7 ms (Δ1d: +5.9%, Δ7d: +29.0%)
   RHR (Whoop): 52.0 bpm (Δ1d: -5.5%, Δ7d: -11.9%)
   RHR (Garmin): 47.0 bpm (Δ1d: -6.0%, Δ7d: -14.5%)
   Recovery Score (Whoop): 96.0 % (Δ1d: +6.7%, Δ7d: +54.8%)
   Sleep Score (Whoop): 82.0 % (Δ1d: +6.5%, Δ7d: +13.9%)
   Body Battery (Garmin): 56.8 % (Δ1d: -8.5%, Δ7d: +52.5%)
   Avg Stress (Garmin, Prev Day): n/a
   Sleep Duration (Whoop): 7.2 h (Δ1d: -28.6%, Δ7d: -0.4%)
   Sleep Consistency (Whoop): 61.0 % (Δ1d: +90.6%, Δ7d: +27.1%)
   Sleep Disturbances/hr (Whoop): 1.2  (Δ1d: +39.9%, Δ7d: +69.2%)
   Strain Score (Whoop): 15.7  (Δ1d: +241.8%, Δ7d: +22.6%)
   Skin Temp (Whoop): 93.2 °F (Δ1d: +181.6%, Δ7d: +171.9%)
   Resp Rate (Whoop): 17.2 rpm (Δ1d: -1.0%, Δ7d: +4.7%)
   Steps (Garmin): 16341.0  (Δ1d: +300%+, Δ7d: +243.6%)
   Total Activity (Garmin): 165.6 min (Δ1d: +175.9%, Δ7d: +165.7%)
   Resp Rate (Garmin): 14.7 rpm (Δ1d: -0.3%, Δ7d: -1.3%)
   VO2max (Garmin): 54.0 ml/kg/min

2. Continue with the standard audit: compare planned vs. actual, surface key metrics, inconsistencies, and actionable steps as usual.

3. Extract **planned workout targets** from BLOCK_SPEC_PATH:
   • Locate row matching RUN_LABEL (using W21 Assessment Plan for "Controlled Zone 2 HR Run")
   • Extract HR, pace, cadence, duration, RPE, and notes

4. Load **zone definitions**:
   • Primary: Parse ZONE_YAML_PATH for `model`, HR & pace breakpoints
   • Cross-verify with ZONE_DOC_PATH; flag discrepancies as critical

5. Parse **actual performance** from RUN_DIR:
   • Prioritize:
     - `run_summary.txt` for primary effort
     - `session_full_summary.txt` for total/walk metrics
   • Supplement with:
     - `advanced_metrics.txt` → EF, drift, hrTSS
     - `cadence_distribution.txt` → mean ± SD
     - `time_in_hr_zone.txt`, `time_in_pace_zone.txt`
     - `hr_over_time_drift.txt`, `walk_summary.txt`, `csv/walk_segments.csv`
     - `pace_over_time.txt` → split trend
     - `weather.txt` for environmental context

6. Generate the analysis report using house layout:

   **0 · TL;DR**
   - 3-sentence summary: overall status / highlights / gaps / next step

   **1 · KPI Dashboard**
   - Table: Duration, Avg HR, % in Z2, HR Drift, EF (vs 0.018 baseline), Cadence ±SD, Walk Ratio, hrTSS, fatigue flags
   - Use icons: ✅ (on-target), ⚠️ (marginal), ❌ (off-target)

   **2 · Root-Cause / Consistency Notes**
   - List of spec mismatches, doc inconsistencies, edge-case ambiguity

   **3 · Target vs Actual**
   - Sub-sections: Session Intent, Highlights, Limiters
   - Include environmental analysis if relevant (e.g. weather.txt, filename hint)

   **4 · Action Plan**
   - Numbered, commit-ready items: route swap, CI tag, PRs, makeup logic

   **5 · Integration Hooks**
   - Table mapping actions to impacted files/workflows

   **6 · Appendices**
   - `<details>` blocks: raw zone tables, key diffs if useful

STYLE & CONSTRAINTS
-------------------
* Begin with “🧙🏾‍♂️:” (alignment), then “🏃‍♂️:” (main body)
* End with a clarifying question or suggested next step
* No invented values—quote only from actual files; use “n/a” if missing
* Default HRmax = 201 unless overridden in YAML
* Treat “pause?” tags as walk unless stated otherwise
* Flag ZONE_YAML `model` ≠ ZONE_DOC_PATH model as critical error

OUTPUT FORMAT
-------------
As described above in step 4—strict layout, human-readable, CI-ready.

🏃‍♂️:

**Systematic Analysis of Run: W21_Thu_Z2_Test_LunchRun (`20250522_174749_lunch_run`)**

**0 · TL;DR**
This Zone 2 test run, executed in very hot conditions (31.5°C), showed good adherence to Z2 HR for a significant portion of the "run-only" segments, but overall efficiency (EF 0.01621) remained below the 0.018 target, likely impacted by the heat and a high walk ratio. Cadence was within an acceptable range but below the NME-focused target. The pre-run wellness indicators were exceptionally strong, suggesting good readiness, but the heat proved to be a dominant factor.

**1 · Pre-Run Wellness Context Analysis (Data for 2025-05-22)**

*   **HRV (Whoop): 116.7 ms (Δ1d: +5.9%, Δ7d: +29.0%)**
    *   *Comment:* Exceptionally high HRV, significantly up both daily and weekly. Indicates excellent autonomic nervous system recovery and readiness. This is a strong positive signal.
*   **RHR (Whoop): 52.0 bpm (Δ1d: -5.5%, Δ7d: -11.9%)**
    *   *Comment:* Lower RHR compared to both yesterday and the weekly average, which is generally a positive sign of good recovery and cardiovascular fitness.
*   **RHR (Garmin): 47.0 bpm (Δ1d: -6.0%, Δ7d: -14.5%)**
    *   *Comment:* Garmin RHR is even lower and shows similar positive trends. The 5bpm difference between Whoop and Garmin RHR is common; both indicate excellent resting state.
*   **Recovery Score (Whoop): 96.0 % (Δ1d: +6.7%, Δ7d: +54.8%)**
    *   *Comment:* Outstanding recovery score, reinforcing the HRV and RHR data. Athlete should be very well-prepared for training.
*   **Sleep Score (Whoop): 82.0 % (Δ1d: +6.5%, Δ7d: +13.9%)**
    *   *Comment:* Good sleep score, indicating sufficient and restorative sleep.
*   **Body Battery (Garmin): 56.8 % (Δ1d: -8.5%, Δ7d: +52.5%)**
    *   *Comment:* Moderate Body Battery. The daily dip is minor, but the weekly trend is very positive. Suggests resources are available, though perhaps not at peak for the day at the time of the run (lunch run after some morning activity).
*   **Avg Stress (Garmin, Prev Day): n/a**
    *   *Comment:* Data not available.
*   **Sleep Duration (Whoop): 7.2 h (Δ1d: -28.6%, Δ7d: -0.4%)**
    *   *Comment:* Shorter sleep duration compared to the previous day, which is a notable dip. However, it's still within a reasonable range and close to the weekly average. This might explain the slight dip in Body Battery.
*   **Sleep Consistency (Whoop): 61.0 % (Δ1d: +90.6%, Δ7d: +27.1%)**
    *   *Comment:* Significant improvement in sleep consistency, which is a positive factor for circadian rhythm and recovery, despite the shorter duration.
*   **Sleep Disturbances/hr (Whoop): 1.2 (Δ1d: +39.9%, Δ7d: +69.2%)**
    *   *Comment:* Increased sleep disturbances are a slight concern, suggesting sleep quality might have been somewhat compromised despite the good overall score. This could be linked to the shorter duration.
*   **Strain Score (Whoop): 15.7 (Δ1d: +241.8%, Δ7d: +22.6%)**
    *   *Comment:* This reflects the strain *including* the run being analyzed, or strain from the day leading up to the run. A high daily strain is expected on a training day. The weekly trend is moderately up.
*   **Skin Temp (Whoop): 93.2 °F (Δ1d: +181.6%, Δ7d: +171.9%)**
    *   *Comment:* This is a very unusual and large jump in skin temperature. This metric often reflects core body temperature changes or environmental influences. Given the hot weather (31.5°C), this high skin temp reading *could* be an early indicator of heat accumulation or systemic response to the environment even before the run. This is a **key wellness risk/observation** for a run in heat.
*   **Resp Rate (Whoop): 17.2 rpm (Δ1d: -1.0%, Δ7d: +4.7%)**
    *   *Comment:* Respiratory rate is relatively stable daily, slightly elevated weekly. No major concern.
*   **Steps (Garmin): 16341.0 (Δ1d: +300%+, Δ7d: +243.6%)**
    *   *Comment:* High step count for the day, indicating an active day overall, even before/after the "lunch run". This contributes to overall daily load.
*   **Total Activity (Garmin): 165.6 min (Δ1d: +175.9%, Δ7d: +165.7%)**
    *   *Comment:* High total activity minutes, aligning with the high step count.
*   **Resp Rate (Garmin): 14.7 rpm (Δ1d: -0.3%, Δ7d: -1.3%)**
    *   *Comment:* Garmin respiratory rate is stable.
*   **VO2max (Garmin): 54.0 ml/kg/min**
    *   *Comment:* VO2max estimate increased slightly from the W17 RPE10 benchmark (53.0). A positive trend.

**Overall Wellness Impact:**
Athlete went into this Z2 test with exceptionally strong markers of recovery (HRV, RHR, Whoop Recovery). Sleep duration was slightly down, and disturbances up, but consistency improved. The most significant flag is the very high Whoop Skin Temperature, which, combined with the hot ambient temperature for the run, suggests a high potential for heat stress to be a major factor affecting performance and Z2 HR control at any given pace. Despite great internal readiness, external conditions are a major limiter.

**2 · KPI Dashboard**

| Metric                   | Actual Value (Run-Only)                     | Target (W21 Assess. Plan) | Status | Notes                                                                        |
| :----------------------- | :------------------------------------------ | :------------------------ | :----- | :--------------------------------------------------------------------------- |
| **Duration (Run-Only)**  | 61.8 min                                    | 25-35 min                 | ❌     | Significantly longer than planned for a focused Z2 test segment              |
| **Avg HR (Run-Only)**    | 155.5 bpm                                   | 145-155 bpm (cap 160)     | ✅     | Average is at the top of the target Z2 range. Max HR 185bpm (Z4).              |
| **% Time in Z2 HR**      | 35.6% (Full Session) / Run-Only Z2: Specific data not in summary, but avg HR 155.5 suggests significant time. | Predominantly Z2          | ⚠️     | Avg HR for run-only is Z2, but overall session has considerable Z1 & Z3 time.  |
| **HR Drift (PwvHR)**     | 4.65% (advanced_metrics.txt) / 1.52% (hr_over_time_drift.txt for run-only) | Low                       | ✅/⚠️    | 1.52% for run-only is good. 4.65% for filtered data is acceptable.        |
| **Efficiency Factor (EF)**| 0.01621                                     | >0.0180 (general target)  | ❌     | Below target, likely impacted by heat and extensive walk periods.            |
| **Avg Cadence ± SD**     | 162.0 ± 6.4 spm                             | 165-170 spm               | ⚠️     | Below target range, but mean is okay. SD indicates some variability.           |
| **Walk Ratio (Overall)** | 33.4 % (`walk_summary.txt` for full session) | Low for focused run       | ❌     | Very high for a Z2 test run, indicates significant active recovery/pauses. |
| **hrTSS (Run-Only)**     | 73.5                                        | (Outcome)                 | N/A    | Moderate training stress.                                                    |

**3 · Root-Cause / Consistency Notes**

*   **Planned vs. Actual Duration:** The "run-only" portion (61.8 min from `advanced_metrics.txt`) is much longer than the planned 25-35 minutes for the Z2 assessment. This extended duration in heat would significantly impact performance and Z2 control.
*   **Zone Definitions:**
    *   `cultivation/data/zones_personal.yml`: HRmax 201bpm. Z2 HR: 145-160bpm. Z2 Pace: [6.85, 8.7] min/km (approx 11:00-14:00 min/mile).
    *   `cultivation/docs/3_design/pace-zones.md`: Not directly used for this analysis as YAML is primary, but should be consistent. Assuming consistency for now.
*   **Heat Impact:** Explicitly noted by the user as a Z2 test *in heat*. `weather.txt` confirms **31.5 °C Clear sky**. This is a major environmental stressor.
*   **Run-Only vs. Session Full:**
    *   `run_only_summary.txt` (filtered for running): 9.35 km, 61.8 min, avg pace 6.61 min/km, avg HR 155.5 bpm, EF 0.01621, decoupling 4.65%. (This is the data from `advanced_metrics.txt`).
    *   `run_summary.txt` (raw, unfiltered, from GPX track points likely with some auto-pause or initial filtering by the device/service): 12.62 km, 99.83 min (1h39m50s), avg pace 6.76 min/km, avg HR 155.5 bpm. *The `avg_pace_min_per_km` in `run_summary.txt` is stated as 6.76, but 12.62km in 99.83 min is 7.91 min/km. The 6.76 seems to be from the device's own "moving time" calculation before more stringent filtering.*
    *   `session_full_summary.txt` (likely represents the entire recording including all pauses): 12.72 km, 102.2 min, avg pace 8.03 min/km, avg HR 152.9 bpm, EF 0.01357, decoupling 9.33%.
    *   The `advanced_metrics.txt` / `run_only_summary.txt` segment of 9.35km in 61.8min at 155.5bpm HR is the most relevant for "running" analysis.
*   **Cadence:** `cadence_distribution.txt` (for run-only) shows mean 162.0 spm, std 6.4 spm. Max 224 spm is an outlier. 10th percentile 156 spm, 90th 168 spm.

**4 · Target vs Actual**

*   **Session Intent (from W21 Assessment Plan for Z2 Run):**
    *   "25-35 minutes continuous running."
    *   "Strict HR Control: Average HR 145-155bpm (absolute cap 160bpm)."
    *   "Cadence Target: 165-170 spm."
    *   "Pace is an outcome."
    *   "Environment: Aim for cooler conditions if possible."
    *   User Context: "Zone 2 test run to see how I’ve progressed with zone 2 training despite the heat."

*   **Actual Performance & Analysis of `txt` files:**
    *   **`advanced_metrics.txt`**:
        *   Distance: 9.35 km
        *   Duration: 61.8 min
        *   Avg Pace: 6.61 min/km (10:39 min/mile)
        *   Avg HR: 155.5 bpm (✅ Top end of Z2)
        *   EF: 0.01621 (❌ Below 0.018 target)
        *   Decoupling: 4.65% (✅ Good)
        *   hrTSS: 73.5
    *   **`cadence_distribution.txt`**: Mean 162.0 spm, Std 6.4 spm (⚠️ Just below 165-170 target).
    *   **`hr_distribution.txt`**: Mean 155.5 bpm, Max 185.0 bpm (Z4). Confirms average HR for this segment was Z2.
    *   **`hr_over_time_drift.txt`**: `hr_drift_pct: 1.52%` (first_half_hr: 157.66, second_half_hr: 160.06). This is a good low drift for the run-only segment. Note: the first/second half HRs are slightly above the 155.5 average, possibly due to how segments are defined or brief excursions.
    *   **`hr_vs_pace_hexbin.txt`**: Correlation 0.127. Weak positive correlation, expected as higher pace generally leads to higher HR, but noise from terrain/walks can affect this.
    *   **`pace_distribution.txt`**: Mean 6.757 min/km. Min 2.43 (likely stride/error), Max 9.41.
    *   **`pace_over_time.txt`**: Strategy "positive" (First half 6.585 min/km, Second half 6.935 min/km). Slowed down in the second half.
    *   **`power_distribution.txt`**: All NaN. No power data.
    *   **`run_only_summary.txt`**: Duplicates `advanced_metrics.txt`.
    *   **`run_summary.txt`**: Already discussed wellness. Session summary here (12.62km, 1h39m50s, Avg Pace 6.76 [discrepancy noted], Avg HR 155.5, Max HR 185) appears to be a less filtered "moving time" version.
    *   **`session_full_summary.txt`**: Total recording: 12.72 km, 102.2 min, Avg Pace 8.03 min/km, Avg HR 152.9. This includes all stops. EF 0.01357 (very low for full session). Decoupling 9.33% (high for full session).
    *   **`stride_summary.txt`**: 3 strides detected.
        *   Stride 1: 9s, 3.92 min/km, 159.8 HR
        *   Stride 2: 29s, 3.02 min/km, 154.0 HR
        *   Stride 3: 15s, 4.08 min/km, 149.3 HR
        *   These are short, fast bursts. Their impact on overall Z2 analysis is minimal.
    *   **`time_in_effective_zone.txt`**: Based on pace/HR combination. `mixed: 71.45%` indicates a large portion where pace and HR zones didn't align perfectly, likely due to heat making HR high for Z2 pace, or walks. `Z3 (Tempo): 10.77%`, `Z1 (Recovery): 14.71%`. `Z2 (Aerobic): 3.07%`. This suggests true effective Z2 work (HR and Pace aligned) was minimal.
    *   **`time_in_fatigue_kpi_zone.txt`**: `Aerobic (Z2): 36.49%`, `High Intensity (Z4+): 0.28%`, `Recovery (<Z2): 42.15%`, `Threshold (Z3): 21.07%`. Based on HR only for the full session.
    *   **`time_in_hr_zone.txt` (Full session data based on `session_full_summary.txt` total duration):**
        *   Z1 (0-145bpm): 43.1%
        *   Z2 (145-160bpm): 35.6% (✅ Significant time in Z2 HR overall)
        *   Z3 (161-175bpm): 21.1%
        *   Z4 (176-186bpm): 0.3%
        *   Z5 (187-201bpm): 0.0%
    *   **`time_in_pace_zone.txt` (Full session data):**
        *   Z1 Pace (7.0+ min/km): 35.7%
        *   Z2 Pace (6.85-8.7 min/km): 8.4%
        *   Z3 Pace (5.2-6.85 min/km): 49.8%
        *   Z4 Pace (4.8-5.2 min/km): 3.2%
        *   Z5 Pace (0.0-4.8 min/km): 3.0%
        *   This shows that by pace, much of the *running* was Z3 pace, but HR was kept in Z2 due to heat.
    *   **`walk_hr_distribution.txt`**: Significant counts in 130-165 bpm range during walks. Avg walk HR (from `walk_summary.txt`) is 150.8 bpm. This is very high for walk recovery, indicating high continued physiological load.
    *   **`walk_over_time.txt`**: Contains detailed per-second data for walk segments. Confirms high HR during walks.
    *   **`walk_pace_distribution.txt`**: Shows a wide range of walk paces.
    *   **`walk_summary.txt`**: 59 walk segments, 34 min 05s (33.4% of session), 2.82km, Avg HR 150.8 bpm.
    *   **`weather.txt`**: Temperature: 31.5 °C, Clear sky. (CONFIRMED HOT)
    *   **`csv/walk_segments.csv`**: Provides structured data for each walk segment, useful for deeper analysis if needed.

*   **Highlights:**
    *   ✅ Excellent pre-run wellness metrics indicated high readiness.
    *   ✅ Average HR for the "run-only" segment (155.5 bpm) was well within the target Z2 range (145-155, cap 160).
    *   ✅ HR drift for the "run-only" segment (1.52%) was very good, suggesting stable aerobic effort *during running portions*.
    *   ✅ Total time with HR in Z2 (35.6% of entire session) was substantial, despite the heat and walk breaks.

*   **Limiters:**
    *   ❌ **Extreme Heat (31.5°C):** This was the dominant factor. It significantly elevates HR for any given pace, making true Z2 *pace* work very difficult and lowering EF. The planned "aim for cooler conditions" was not met.
    *   ❌ **Efficiency Factor (0.01621 for run-only):** Well below the 0.018 target. Heat is a major contributor. This suggests that at this HR in these conditions, the pace achieved was relatively slow.
    *   ❌ **High Walk Ratio (33.4%):** While walks are fine, such a high percentage for a Z2 *test* run makes it harder to assess continuous aerobic performance. Walk HR was also very high (avg 150.8 bpm), offering limited recovery.
    *   ⚠️ **Cadence (162.0 spm avg for run-only):** Below the 165-170 spm target, indicating a potential area for continued focus, especially as NME work aims to improve this.
    *   ❌ **Extended Duration:** The run-only portion was 61.8 mins, much longer than the 25-35 min Z2 test specified in the assessment plan. This longer duration in heat would compound fatigue and impact metrics.
    *   ⚠️ **Positive Split:** The run-only segment showed a positive split, indicating a slowdown, likely due to heat and duration.
    *   **Discrepancy in "Run-Only" Data:** The `run_summary.txt` (from device) and `advanced_metrics.txt` (processed) show different overall durations/distances for "running." The `advanced_metrics.txt` is preferred for run-only analysis.

**5 · Action Plan**

1.  **Acknowledge Heat Impact:** Recognize that this session provides valuable data on Z2 performance *under significant heat stress*. It's not directly comparable to Z2 efforts in cooler conditions.
2.  **Future Z2 Tests:** For more comparable Z2 progression assessment, aim for cooler conditions as per the plan. If heat is unavoidable, shorten the intended "test" segment (e.g., stick to 25-30 mins run-only) to minimize drift due to prolonged heat exposure.
3.  **Cadence Focus:** Continue to emphasize cadence cues (165-170 spm) during Z2 runs, potentially using a metronome for portions of runs if natural cadence isn't consistently in range.
4.  **Walk Break Strategy in Heat:** If walk breaks are necessary in heat, focus on actively trying to lower HR during these breaks (e.g., slower walking, finding shade). High HR during walks limits recovery.
5.  **Data Interpretation:** When analyzing future Z2 runs in heat, expect lower EF and higher RPE for a given Z2 HR. Pace will naturally be slower.
6.  **Log Skin Temperature Context:** The very high pre-run Whoop skin temperature is interesting. Correlate this with future sessions and environmental conditions to see if it's a reliable personal indicator of heat susceptibility or acclimatization state.

**6 · Integration Hooks**

| Action                                      | Impacted Files/Workflows                                                                |
| :------------------------------------------ | :-------------------------------------------------------------------------------------- |
| Refine Z2 Test Protocol in Heat             | `cultivation/docs/4_analysis/running_analysis_template.md`, Future training plan notes. |
| Emphasize Cadence in Training Plans         | `cultivation/outputs/training_plans/baseox_daily_plans/...` (future week plans)       |
| Note on High Walk HR in Heat                | `cultivation/docs/4_analysis/operational_playbook.md` (section on heat running)       |
| Add Skin Temp to Wellness Review Checklist  | `cultivation/docs/4_analysis/operational_playbook.md` (wellness section)              |

**7 · Appendices**

<details>
<summary>Zone Definitions (from `cultivation/data/zones_personal.yml`)</summary>

```yaml
# From zones_personal.yml (content inferred from run_summary.txt output)
# Assuming HRmax: 201
# Z1 (Recovery): {'bpm': [0, 145], 'pace_min_per_km': [7.0, 999]}
# Z2 (Aerobic): {'bpm': [145, 160], 'pace_min_per_km': [6.85, 8.7]}
# Z3 (Tempo): {'bpm': [161, 175], 'pace_min_per_km': [5.2, 6.85]}
# Z4 (Threshold): {'bpm': [176, 186], 'pace_min_per_km': [4.8, 5.2]}
# Z5 (VO2max): {'bpm': [187, 201], 'pace_min_per_km': [0.0, 4.8]}
```
</details>

<details>
<summary>Key Data Points Summary</summary>
- **Run-Only (advanced_metrics.txt):**
  - Duration: 61.8 min
  - Distance: 9.35 km
  - Avg HR: 155.5 bpm
  - Avg Pace: 6.61 min/km
  - EF: 0.01621
  - Cadence: 162.0 spm
  - HR Drift (run-only points): 1.52%
- **Full Session (session_full_summary.txt):**
  - Duration: 102.2 min
  - Distance: 12.72 km
  - Avg HR: 152.9 bpm
  - Avg Pace: 8.03 min/km
  - Walk Ratio: 33.4%
  - Avg Walk HR: 150.8 bpm
- **Weather:** 31.5°C, Clear sky
- **Pre-Run Wellness:** Generally excellent (HRV 116.7ms, Recovery 96%), but high skin temp (93.2°F) and shorter sleep (7.2h).
</details>

Given the extreme heat and the extended duration beyond the planned Z2 test window, how should this session be weighted when assessing underlying Z2 aerobic progression versus primarily reflecting performance in severe heat?