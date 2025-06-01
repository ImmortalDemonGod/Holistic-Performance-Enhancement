# Runner Profile & Baseline Model (v1.0 - 2025-05-01)

**Status:** Initial Baseline - Derived from Base-Ox Week 1

**Date:** 2025-05-01

**Author(s):** Gemini AI (derived analysis)

**Data Sources:**
* Calibration Run: `20250421_000013_evening_run_calibration_pre-baseox_RPE_7` (Wk17 Mon)
* Calibration Run: `20250425_201748_afternoon_run_calibration_pre-baseox_RPE_10` (Wk17 Fri)
* Base-Ox Run: `20250429_191120_baseox_wk1_tue_z2_strides_25min_147_155bpm_6x20shill_RPE_5_hr_override` (Wk18 Tue)
* Base-Ox Run: `20250501_164245_baseox_wk1_day4_thu_z2steady_25min_postflood_cameron_loop_hr147_155_cadence82_84_rpe_3` (Wk18 Thu)

**Scope:** This document establishes an initial physiological and biomechanical baseline model for the runner entering the Base-Ox training block. It synthesizes findings from initial calibration runs and the first week of the structured plan to guide training adjustments and system refinements.

---

## 1. Key Findings (Executive Summary)

The runner enters the Base-Ox block with an underdeveloped aerobic base, characterized by low efficiency (EF ~0.018-0.020) and a significant disconnect between target Zone 2 (AeT) heart rates and achievable pace. Attempts to follow planned Z2 *pace* bands result in excessive cardiovascular strain (Z3/Z4 HR). Perceived effort (RPE) is poorly calibrated, particularly at sub-maximal intensities. Running cadence is consistently low (~155-165 spm). While capable of reaching near-max HR and executing strides, sustainability at higher intensities is limited, often requiring walk breaks. Strict adherence to HR targets for Z2 work is paramount, overriding pace initially. The Base-Ox plan's Z2 pace parameters require immediate downward revision based on observed HR responses.

---

## 2. Detailed Profile

### 2.1. Physiological Markers

* **Heart Rate Zones & Max HR:**
    * Model: AeT (as defined in `cultivation/data/zones_personal.yml` and `docs/3_design/pace-zones.md`). Z2 ceiling: 160 bpm.
    * Max HR: Observed at 199-201 bpm (Wk17 Runs). Confirms the upper end of defined zones (Z5: 187-201 bpm).
    * Zone 2 (AeT: 145-160 bpm): Maintaining this HR requires significantly slower pace than planned (See 2.5). Successful HR control demonstrated in Wk18 Thu run (Avg 149.5 bpm) when prioritized over pace, albeit under adverse conditions.
* **Aerobic Efficiency (EF):**
    * Baseline: Appears low, around 0.018 - 0.0205 in calibration runs.
    * Response to Intensity: EF decreased from 0.0205 (RPE 7, Avg HR 179) to 0.0180 (RPE 10, Avg HR 183.6), indicating poor economy at higher efforts.
    * Base-Ox Wk1: 0.0182 (Wk18 Tue, Z3/Z4 effort), 0.0146 (Wk18 Thu, Z2 HR but slow/flood conditions). The low value on Wk18 Thu is likely artifactual due to environmental factors drastically reducing pace.
    * Trend: No positive trend observed in this initial dataset. Requires monitoring during correctly executed Z2 runs.
* **Pace/HR Decoupling:**
    * Significant decoupling observed when targeting Z2 pace. Wk18 Tue required Z3/Z4 HR (Avg 175) to maintain pace near the top of the Z2 *pace* band (5.41 min/km).
    * Wk18 Thu showed minimal decoupling (3.54%) *within* the run segment, but the overall pace (7.71 min/km) was decoupled from the target Z2 pace band (5.00-5.20 min/km) required to stay in Z2 HR.
* **HR Drift:**
    * Generally low to moderate: 5.0% (Wk17 RPE 7), 3.0% (Wk17 RPE 10), -0.4% (Wk18 Tue Z3/Z4), 1.0% (Wk18 Thu Z2).
    * Low values in Wk17 RPE 10 and Wk18 Tue may be influenced by walk breaks/intensity changes. Low drift in Wk18 Thu Z2 run supports good aerobic control *at that specific slow pace*.
* **Sustainability / Endurance:**
    * Limited at higher intensities: Wk17 RPE 10 run required 15.1% walk time. Wk18 Tue (Z3/Z4 effort) required 9.0% walk time.
    * Impacted by environment: Wk18 Thu Z2 run required 34.5% walk time, strongly correlated with "postflood" conditions.
    * Pacing: Predominantly positive splits (Wk17 RPE 7, Wk17 RPE 10, Wk18 Tue), suggesting fatigue or starting too fast.

### 2.2. Biomechanical Markers

* **Cadence:**
    * Level: Consistently low, averaging 155-165 spm across all runs. Significantly below common targets (175-180 spm).
    * Consistency: Generally consistent within runs (SD 3.8-6.0 spm).
    * Target Adherence: Missed specific target of 164-168 spm during Wk18 Thu run (Actual 155.2 spm), potentially due to terrain.

### 2.3. Perceptual Markers

* **RPE Calibration:** Poorly calibrated, especially at sub-maximal levels.
    * RPE 7 (Wk17 Mon) -> Avg HR 179 (Z4).
    * RPE 5 (Wk18 Tue filename) -> Avg HR 175 (Z3/Z4).
    * RPE 10 (Wk17 Fri) -> Avg HR 183.6 (Z5). Better alignment at max effort.
    * RPE 3 (Wk18 Thu filename) -> Avg HR 149.5 (Z2). Good alignment when HR is actively managed, but pace suffered greatly.

---

## 3. Training Plan Interaction (Base-Ox Wk 1)

* **Adherence to Z2 Stimulus:** Failed in Wk18 Tue (HR too high). Succeeded in Wk18 Thu (HR correct), but execution heavily modified by environment. The core aerobic stimulus objective is currently at risk due to intensity control issues.
* **Validity of Pace Parameters:** Z2 Pace Bands in `base_ox_block.md` (derived from 8:03-8:43 min/mile / ~5:00-5:42 min/km range) are **too fast** for the runner's current Z2 *effort* level. Adhering to these paces forces Z3/Z4+ heart rates.
* **Stride Execution:** Wk18 Tue showed reasonable ability to execute planned strides based on detected segments.

---

## 4. System & Data Notes

* **HR Data:** HR override successfully applied in Wk18 Tue, but resulted in sparse data points (935 vs 2245 cadence points). Requires investigation of the source FIT file (`...191330...fit`) and/or the `override_gpx_hr_with_fit.py` script.
* **Walk Detection:** `walk_utils.py` logic may need refinement; minor discrepancies noted between `walk_segments.csv` and `walk_summary.txt` in Wk17. Wk18 runs showed consistency. High walk % in Wk18 Thu correctly reflects reported conditions.
* **Stride Detection:** Appears sensitive to general pace variability in non-stride workouts (Wk17) but functional for detecting planned strides (Wk18 Tue).
* **Environmental Factors:** Wk18 Thu "postflood" conditions significantly impacted pace, cadence, walk ratio, and EF, demonstrating sensitivity to terrain/environment. This context is crucial for interpretation.

---

## 5. Model-Derived Recommendations

1.  **(Training Execution)** **Mandate HR Cap for Z2:** Enforce strict adherence to the upper Z2 HR limit (160 bpm) via watch alerts during all Z2 segments. Pace must be adjusted (slowed or walked) to remain below cap. *Primary action.*
2.  **(Plan Adjustment)** **Revise Z2 Pace Bands:** Update `zones_personal.yml` pace values and `base_ox_block.md` descriptive pace bands based on pace achieved during *correctly executed* Z2 HR runs (e.g., target HR 147-155 resulted in ~7.7 min/km pace under duress; unimpeded pace likely faster but slower than current 5:00-5:42 band). *Requires more data.*
3.  **(Training Focus)** **Improve Cadence:** Implement targeted cadence work (drills, metronome focus) aiming for gradual increase towards 170+ spm. Add specific cadence goals/reminders to plan documents.
4.  **(Training Focus)** **Recalibrate RPE:** Emphasize correlating RPE (target 3-4) specifically with *achieved* Z2 HR during runs. Use `data/subjective.csv` to track RPE against HR Zone adherence.
5.  **(System Check)** **Investigate HR Override Data Sparsity:** Analyze the Wk18 Tue FIT file and `override_gpx_hr_with_fit.py` processing to resolve missing HR data points.
6.  **(System Check)** **Review Walk Detection Parameters:** Minor review of `walk_utils.py` thresholds based on Wk17 discrepancies.
7.  **(Analysis Context)** **Annotate Environmental Runs:** Ensure runs significantly impacted by external factors (e.g., Wk18 Thu postflood) are flagged in logs/reports to avoid misinterpreting metrics like EF.

---

## 6. Future Monitoring

* **% Time in Target HR Zone (Z2):** Primary indicator of adherence to Base-Ox intent. Target >80-90% for Z2 segments.
* **Efficiency Factor (EF) Trend:** Monitor during correctly executed Z2 runs. Expect gradual increase from ~0.018 baseline.
* **Z2 Pace at Target HR:** Track the actual pace achieved while maintaining Z2 HR to refine pace bands.
* **Cadence Average & SD:** Track progress towards higher, consistent cadence.
* **Walk Ratio:** Monitor trends during Z2 runs under normal conditions. Expect decrease as fitness improves.
* **Subjective RPE vs HR Zone:** Track alignment during Z2 runs.