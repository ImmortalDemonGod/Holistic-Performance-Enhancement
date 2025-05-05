⸻

✅ OPTIMIZED CULTIVATION RUN ANALYZER TEMPLATE

⸻

🧠 MISSION

Systematically audit a Cultivation running session by comparing planned workout specifications against actual performance, surfacing key metrics, inconsistencies, and actionable steps, using structured inputs and canonical file sources.

⸻

📇 INPUTS TO PROVIDE (Replace ALL-CAPS placeholders)

RUN_LABEL:           WkX_DayY_<descriptor>       # e.g. "W1_Thu_Z2Steady_postflood"
BLOCK_SPEC_PATH:     cultivation/docs/3_design/base_ox_block.md
ZONE_YAML_PATH:      cultivation/data/zones_personal.yml
ZONE_DOC_PATH:       cultivation/docs/3_design/pace-zones.md
RUN_DIR:             cultivation/outputs/figures/WEEK_DIR/RUN_LABEL/
RAW_FILE_MAP:        (Dump raw file tree from Git under RUN_DIR)

Tip: Use Git-style tree view for RAW_FILE_MAP (indented, file names only). This allows the model to infer metrics location.

⸻

🧞‍♂️ PROMPT TEMPLATE (PASTE AFTER FILLING)

🧙🏾‍♂️: We are aligning on a systematic audit of a Cultivation running session.

INPUT METADATA
--------------
RUN_LABEL: "RUN_LABEL"
BLOCK_SPEC_PATH: BLOCK_SPEC_PATH
ZONE_YAML_PATH:  ZONE_YAML_PATH
ZONE_DOC_PATH:   ZONE_DOC_PATH
RUN_DIR:         RUN_DIR
RAW_FILE_MAP:
RAW_FILE_MAP

TASK
----
1. **Analyze the Pre-Run Wellness Context section** (as shown below) in detail. For each metric, comment on its value, trend (Δ1d, Δ7d), and any notable deviations, improvements, or risks. Consider how these wellness factors might impact the athlete's readiness, performance, and recovery for this session. If multiple data sources are present (e.g., Whoop vs Garmin RHR), compare and interpret both. Highlight any discrepancies or patterns that could inform training decisions.

--- Pre-Run Wellness Context (Data for YYYY-MM-DD) ---
   HRV (Whoop): ...
   RHR (Whoop): ...
   RHR (Garmin): ...
   ... (etc)

2. Continue with the standard audit: compare planned vs. actual, surface key metrics, inconsistencies, and actionable steps as usual.

3. Extract **planned workout targets** from BLOCK_SPEC_PATH:
   • Locate row matching RUN_LABEL  
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
     - `hr_over_time_drift.txt`, `walk_summary.txt`, `walk_segments.csv`  
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

ANALYSIS SCOPE
--------------
For this session, you MUST systematically analyze the contents of EVERY file in the folder:
  cultivation/outputs/figures/weekXX/SESSION_LABEL/txt/
...including but not limited to:
  - run_summary.txt
  - session_full_summary.txt
  - run_only_summary.txt
  - advanced_metrics.txt
  - stride_summary.txt
  - walk.txt
  - walk_summary.txt
  - hr_distribution.txt
  - cadence_distribution.txt
  - pace_distribution.txt
  - hr_over_time_drift.txt
  - pace_over_time.txt
  - hr_vs_pace_hexbin.txt
  - power_distribution.txt
  - time_in_hr_zone.txt
  - time_in_pace_zone.txt
  - time_in_effective_zone.txt
  - walk_hr_distribution.txt
  - walk_pace_distribution.txt
  - weather.txt

Do NOT ignore any file. If a file is empty, missing, or not relevant, explicitly state this in your output. Quote directly from each file as needed. Use “n/a” if data is missing.

Continue with the standard audit as described below:

⸻

⚙️ STYLE & ERGONOMICS
	•	Strong signaling & whitespace: Improves readability when pasted into ChatGPT
	•	File parsing order is deterministic: Sets clear preference hierarchy
	•	Explicit structure labels in TASK and STYLE: Allows better context modeling
	•	Constraints are grouped and formatted for rapid human parsing: No buried assumptions

⸻

🧩 Example Use (Minimal)

RUN_LABEL: "W3_Mon_Z2Progression_wet"
BLOCK_SPEC_PATH: cultivation/docs/3_design/base_ox_block.md
ZONE_YAML_PATH:  cultivation/data/zones_personal.yml
ZONE_DOC_PATH:   cultivation/docs/3_design/pace-zones.md
RUN_DIR:         cultivation/outputs/figures/wk3/W3_Mon_Z2Progression_wet/
RAW_FILE_MAP:
├── run_summary.txt
├── session_full_summary.txt
├── advanced_metrics.txt
├── time_in_hr_zone.txt
├── cadence_distribution.txt
├── walk_segments.csv
├── weather.txt

🧙🏾‍♂️: [Paste optimized template with above values]

⸻