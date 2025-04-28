# Base‑Ox Mesocycle (Weeks 1 – 4)

> **Purpose:** Lay a durable aerobic foundation and oxidative economy that subsequent Tempo‑Dur, Heat‑Acc, and Polar‑VO₂ blocks can safely build on.  All prescriptions below are fully wired to the Cultivation repo’s ETL ➝ metrics ➝ scheduler toolchain.

---

## 0 · Quick‑spec

| Property | Value |
|-----------|-------|
| **Duration** | 4 weeks (3 load + 1 deload) |
| **Primary Stress** | Zone‑2 steady running (72–78 % HRₘₐₓ) |
| **Secondary Stress** | Long‑run ≥ 90 min, 6–8″ hill strides |
| **Progression KPIs** | ① Efficiency‑Factor **+ 5 %** (rolling) ② HR‑drift **< 7 %** on long run |
| **Exit‑gate** | Both KPIs pass; no red‑flag events |

---

## 1 · Physiological Rationale

| Target Adaptation | Mechanistic Driver | Expected Biomarker Response |
|-------------------|--------------------|----------------------------|
| ↑ Mitochondrial density & oxidative enzymes (CS, SDH) | ≥ 45 min continuous at 2–2.5 mmol L⁻¹ lactate (\~75 % HRₘₐₓ) | EF ↑ 3–5 % wk⁻¹; Rest‑HR ↓ 2–4 bpm |
| ↑ Capillary density & plasma volume | 90 min long‑run at 65–75 % HRₘₐₓ | HR‑drift slope ↓ ≥ 0.5 % km⁻¹ |
| Improved neuromuscular recruitment efficiency | 6–8″ strides / hill surges @ \>180 spm | Cadence SD ↓; Flight‑time ↑ |
| Connective‑tissue robustness | Low‑impact Z2 + eccentric hills | sRPE ≤ 4 despite ↑ km |

*Why four weeks?*  Mito biogenesis and plasma‑volume expansion plateaus after \~21 days; week‑4 deload locks gains and mitigates over‑reach.

---

## 2 · Weekly Micro‑cycle Blueprint

| Day | Session | Volume (min) | Intensity cue | Training Focus |
|-----|---------|--------------|---------------|----------------|
| **Mon** | OFF (+ HRV log) | — | — | Super‑compensation |
| **Tue** | Z2 steady + 6×20″ strides | 60 | 73–77 % HRₘₐₓ | Mito flux + neuromuscular snap |
| **Wed** | Recovery jog + mobility | 40 | 65–70 % HRₘₐₓ | Capillary flush |
| **Thu** | Z2 steady (alt route) | 60 | 75 % HRₘₐₓ | Economy consistency |
| **Fri** | OFF / light strength | — | — | Tendon stiffness |
| **Sat** | Long run progressive | 90 → 100 | 72 → 75 % HRₘₐₓ | Durability + fuel pathway |
| **Sun** | Walk / bike spin | 30 | \< 60 % HRₘₐₓ | Glycogen refill |

### 2.1 Load ramp & deload

| Week | Run‑time (min) | Long‑run (min) | Strides (reps) | Km (est @5 min·km⁻¹) |
|------|----------------|----------------|----------------|----------------------|
| 1 | 250 | 90 | 6 | 38 |
| 2 | 270 | 95 | 6 | 42 |
| 3 | 290 | 100 | 8 | 46 |
| 4 *(deload)* | 220 | 80 | 4 | 34 |

*The deload week automatically triggers in `pid_scheduler.py` once the calendar’s `block=BaseOx` & `week=4` condition is met.*

---

## 3 · Plan CSV (integration‑ready)

```csv
week,day,session_code,duration_min,intensity_pct_hrmax,comments
1,Mon,OFF,,,
1,Tue,Z2+STRIDES,60,0.75,"6x20s strides"
1,Wed,RECOVERY,40,0.68,
1,Thu,Z2,60,0.75,
1,Fri,STRENGTH,,,
1,Sat,LONG,90,"0.72→0.75","Prog HR‑drift watch"
1,Sun,CROSS,30,0.55,"Bike / walk"
… repeat pattern for weeks 2‑4 with updated durations …
```
Commit as `training_plans/2025_Q2_BaseOx.csv`.

---

## 4 · KPI Computation & CI Hooks

| KPI | Script call | Pass Threshold | CI Snippet |
|-----|-------------|----------------|-----------|
| **Efficiency‑Factor Δ** | `python metrics.py --ef --lookback 2` | ≥ +5 % vs Week‑1 baseline | `ef_change >= 0.05` |
| **HR‑drift (decoupling)** | `python metrics.py --drift --session long_run` | ≤ 0.07 | `drift_pct <= 0.07` |
| **Subjective RPE** | append to `data/subjective.csv` | ≤ 4 avg | `avg_rpe <= 4` |
| **Cadence variance** | auto in `parse_run_files.py` | SD ≤ 3 spm | `cad_sd <= 3` |

### CI YAML fragment

```yaml
jobs:
  metrics-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: python scripts/running/metrics.py --latest
      - id: gate
        run: |
          echo "ef_change=$(cat ef.txt)" >> $GITHUB_OUTPUT
          echo "drift_pct=$(cat drift.txt)" >> $GITHUB_OUTPUT
      - name: Block progression if KPIs fail
        if: ${{ steps.gate.outputs.ef_change < 0.05 || steps.gate.outputs.drift_pct > 0.07 }}
        run: |
          gh label create base-hold || true
          gh issue edit ${{ github.event.number }} --add-label base-hold
          exit 1
```

---

## 5 · Instrumentation Guidelines

1. **Watch data fields** – Lap pace, Lap HR, Live EF (Garmin CIQ `efficiency-factor` field), HR‑drift alarm after 60′.
2. **Fuel log** – record carbs g·h⁻¹ & fluids mL in `nutrition_log.csv` (schema: date,session_id,carbs,fluid_ml).
3. **Cadence audio** – optional phone metronome 172‑180 spm during strides; raw WAVs stored in `data/audio/` → analysed via `notebooks/running/cadence_audio.ipynb`.

---

## 6 · Validation Milestones

| Week | Mini‑test | Pass Criteria | Script / notebook |
|------|-----------|---------------|-------------------|
| 2 | 30′ sub‑LT constant‑pace | HR‑drift < 4 % | `notebooks/running/subLT_drift.ipynb` |
| 4 | Repeat Week‑1 long‑run | EF ↑ ≥ 5 %; Avg HR ↓ ≥ 6 bpm | `scripts/running/compare_runs.py --id week1_long week4_long` |

Failing either test automatically **extends Base‑Ox one week** (`pid_scheduler.py` checks CI badge). Calendar rows `week=5a` already exist.

---

## 7 · Potential‑Engine Coupling

```python
# inside calculate_synergy.py
p_run_base = zscore(ef_trend)
if p_run_base > 0:
    potential_tensor["Cognitive"].weight += 0.01  # improved focus expected
```
Adds the aerobic gain as a positive modulation for cognitive‑domain potential.

---

## 8 · Risk Flags & Mitigations

| Trigger | Auto‑action |
|---------|------------|
| Rest‑HR ↑ > 8 bpm for 3+ days | Drop Thursday Z2 to 40′ recovery |
| HRV‑VLF power ↓ > 20 % vs baseline | Replace strides with mobility; schedule 30′ extra sleep task |
| sRPE ≥ 6 on any Z2 run | 48‑h full rest + tag `fatigue‑alert` in Task‑Master |

All red flags surface as GitHub issues via the `fatigue_watch.py` daily cron workflow.

---

## 9 · Lean‑Proof Placeholder

- **Lemma:** "If decoupling ≤ ε and EF trend ≥ δ > 0 then aerobic‑economy score monotone‑increases over the Base‑Ox period."  
- Located at `lean/Cultivation/Running/BaseOx.lean` – stub created, proof to be supplied in Phase P2.

---

### 📎 Next steps

1. Commit the CSV & this doc.
2. Implement `planning_id` column in `parse_run_files.py`.
3. Merge CI gate; ensure green.
4. Schedule first mini‑test (`subLT_drift.ipynb`) by Week‑2.

Once those are live the Base‑Ox block becomes **self‑governing**—data in, metrics checked, and progression automated.

