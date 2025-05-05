# Base‑Ox Mesocycle (Weeks 1 – 4)

> **Purpose:** Establish a *durable* aerobic foundation while respecting the current ~10 mi · wk⁻¹ baseline.  All prescriptions are wired to the Cultivation repo’s ETL → metrics → scheduler tool‑chain and ramp total workload by ≈ 8‑10 % per week.

---

## 0 · Quick‑spec

| Property | Value |
|-----------|-------|
| **Duration** | 4 weeks (3 load + 1 deload) |
| **Opening Volume** | **~10 mi · wk⁻¹ ≈ 95 min** total run‑time |
| **Primary Stress** | Zone‑2 steady running (72–78 % HRₘₐₓ) |
| **Secondary Stress** | Long‑run 30→40 min, 6–8″ hill strides |
| **Progression KPIs** | ① Efficiency‑Factor **+ 5 %** (rolling) ② HR‑drift **< 7 %** on long‑run |
| **Exit‑gate** | Both KPIs pass; no red‑flags |

---

## 1 · Physiological Rationale  (why Z2 & why a *small* ramp)

| Target Adaptation | Mechanistic Driver | Expected Biomarker Response |
|-------------------|--------------------|----------------------------|
| ↑ Mitochondrial density & oxidative enzymes | ≥ 25 min continuous at 70‑78 % HRₘₐₓ | EF ↑ 3–5 % wk⁻¹; Rest‑HR ↓ 1–3 bpm |
| ↑ Capillary density & plasma volume | Long‑run 30→40 min @ 65‑75 % HRₘₐₓ | HR‑drift slope ↓ ≥ 0.3 % mi⁻¹ |
| Improved neuromuscular efficiency | 6×20″ strides / hill surges @ Gradual increase from baseline (~165 spm) towards 170+ spm; focus on improvement trend and consistency (low SD) | Cadence SD ↓; Flight‑time ↑ |
| Connective‑tissue robustness | Low‑impact Z2 + eccentric hills | sRPE ≤ 4 despite ↑ mi |

*Four weeks* allow mitochondrial and plasma‑volume adaptations to express while limiting over‑reach at low mileage.

---

## 2 · Weekly Micro‑cycle Blueprint    — *real HR & pace bands (min·mile⁻¹)*

> **Z2 pace band:** 6:51–8:43 min/km (≈ 11–14 min/mi) — empirically derived from HR-compliant window, see pace-zones.md for details.

| Day | Session | Duration ( min) | **HR target** (%) | Focus |
|-----|---------|----------------|----------------------|-------|
| **Mon** | OFF (+ HRV log) | — | — | Super‑compensation |
| **Tue** | <abbr title="Include 5‑min brisk walk + 5‑min jog warm‑up; 5‑min cool‑down">Z2 steady + 6×20″ strides*</abbr> | **25** | **72–78** | Mito flux + neuromuscular snap |
| **Wed** | Recovery jog + mobility | **15** | **65–70** | Capillary flush |
| **Thu** | Z2 steady (alt route) | **25** | **72–78** | Economy consistency |
| **Fri** | OFF / <abbr title="Suggested: 2× circuit – split‑squat, single‑leg RDL, calf‑raise, plank">light strength†</abbr> | — | — | Tendon stiffness |
| **Sat** | Long‑run progressive ‡ | **30 → 40** | **72–78** | Durability + fuel pathway |
| **Sun** | Walk / bike spin | 15 | **≤ 55** | Glycogen refill |

<small>*Strides on 3–4 % grade if possible for eccentric stimulus.<br>†Body‑weight or < 25 % 1‑RM loads to avoid DOMS.<br>‡Always precede with 5‑min walk + 5‑min jog; finish with 5‑min walk cool‑down.</small>

> *Execution Priority: Maintain target HR strictly for all Z2 and Recovery sessions. Pace is secondary and should be adjusted (slowed or walked) as needed to stay within the target HR zone. Initial pace may be significantly slower than typical Z2 pace bands.*

> *Initial pace at target Z2 HR needs to be established during the first few weeks.*

### 2.1 Load‑Ramp & Deload  – Mileage / Run‑time

| Week | Run‑time ( min) | Long‑run ( min) | Strides ( reps) | **Miles (low / high)** |
|------|---------------|----------------|----------------|------------------------|
| **1** | **95** | 30 | 6 | **9.5 / 10.2** |
| **2** | 105 | 35 | 6 | 10.5 / 11.2 |
| **3** | 115 | 40 | 8 | 11.5 / 12.3 |
| **4 *(deload)* **| 85 | 25 | 4 | 8.2 / 9.0 |

Run‑time grows ≈ 10 % per week, aligning with safe progression guidelines.

### 2.2 Equivalent Kilometres

| Week | Km ( low / high) |
|------|----------------|
| 1 | 15 / 16.4 |
| 2 | 16.9 / 18.0 |
| 3 | 18.5 / 19.8 |
| 4 | 13.2 / 14.5 |

### 2.3 · Warm‑up / Cool‑down & Logging Standards

| Phase | Action | Repository touch‑point |
|-------|--------|-----------------------|
| **Pre‑session** | `warmup_id=W01` flag stored in FIT notes; 5‑min brisk walk + dynamic drills template lives under `docs/training/warmups.md`. | Parsed in `parse_run_files.py` ➝ column `warmup_dur` |
| **Post‑session** | 5‑min walk; stretch log (checkbox in mobile form) | `data/recovery.csv` (schema: date,session_id,stretch_min) |
| **Hill grade** | Auto‑detected (`elev_gain/distance`); validator asserts 2–5 % for strides. | Fails CI if < 2 % or > 6 % |

---

## 3 · Plan CSV (integration‑ready) · Plan CSV (integration‑ready)

```csv
week,day,session_code,duration_min,intensity_pct_hrmax,comments
1,Mon,OFF,,,
1,Tue,Z2+STRIDES,25,0.75,"6x20s strides"
1,Wed,RECOVERY,15,0.68,
1,Thu,Z2,25,0.75,
1,Fri,STRENGTH,,,
1,Sat,LONG,30,"0.72→0.75","HR‑drift watch"
1,Sun,CROSS,15,0.55,"Bike / walk"
# Weeks 2‑4 duplicate pattern with durations: Tue 28/31/23, Wed 17/19/13, Thu 28/31/23, Sat 35/40/25
```
Commit as `training_plans/2025_Q2_BaseOx.csv`.

*(CSV shows Week‑1; weeks 2‑4 generated by scheduler script to respect new durations.)*

---

## 4 · KPI Computation & CI Hooks  (unchanged)

| KPI | Script call | Pass Threshold |
|-----|-------------|----------------|
| **Efficiency‑Factor Δ** | `python metrics.py --ef --lookback 2` | ≥ +5 % vs Week‑1 baseline |
| **HR‑drift** | `python metrics.py --drift --session long_run` | ≤ 0.07 |
| **Subjective RPE** | `data/subjective.csv` | ≤ 4 avg |
| **Cadence variance** | auto | SD ≤ 3 spm |

CI YAML remains valid; no change required.

---

## 5 · Instrumentation Guidelines

1. **Watch face fields** – Lap pace, Lap HR, Live EF (Garmin CIQ *efficiency‑factor* field), plus an HR‑drift alert triggered after 20 min on the Saturday long‑run.
2. **Fuel & hydration log** – record carbohydrate intake *(g · h⁻¹)* and fluids *(mL)* in `nutrition_log.csv` (schema: `date,session_id,carbs_g,fluid_ml`).
3. **Mobility & strength capture** – `recovery.csv` (schema: `date,session_id,stretch_min,strength_min`) populated automatically from the mobile form check‑boxes.
4. **Hill‑grade validator** – CI job `ci‑grade.yml` fails if the average grade during strides is < 2 % or > 6 % (parsed from FIT elevation data).
5. **Live Cadence**: Monitor cadence during runs, aiming for gradual increase and consistency.
6. **HR Zone Alerts**: Configure watch alerts for the Z2 upper limit (e.g., 160 bpm) to ensure intensity discipline.

---

## 6 · Validation Milestones

| Week | Mini‑test | Pass Criteria | Tool |
|------|-----------|---------------|------|
| 2 | 30 min sub‑LT constant‑pace | HR‑drift < 4 % | `notebooks/running/subLT_drift.ipynb` |
| 4 | Repeat Week‑1 long‑run | EF ↑ ≥ 5 %; Avg HR ↓ ≥ 6 bpm | `scripts/running/compare_runs.py --id week1_long week4_long` |

Failing either test auto‑extends Base‑Ox by one week (`pid_scheduler.py` checks the CI badge and re‑queues *Week 5a* rows).

---

## 7 · Potential‑Engine Coupling

```python
# inside calculate_synergy.py
p_run_base = zscore(ef_trend)
if p_run_base > 0:
    potential_tensor["Cognitive"].weight += 0.01  # aerobic fitness ⇢ better focus
```

The `p_run_base` channel is already normalised, so no change is required for the lower mileage baseline.

---

## 8 · Risk Flags & Automated Mitigations

| Trigger | Auto‑action |
|---------|------------|
| Rest‑HR ↑ > 8 bpm (3 d rolling) | Replace Thursday Z2 with 15 min recovery; raise `fatigue‑alert` issue |
| HRV‑VLF power ↓ > 20 % vs baseline | Remove Tuesday strides; schedule 30‑min additional sleep task |
| sRPE ≥ 6 on any Z2 run | 48‑h full rest + tag `fatigue‑alert` in Task‑Master |

Alerts are surfaced by `fatigue_watch.py` (daily cron) and block progression labels in GitHub.

---

## 9 · Lean‑Proof Placeholder

> **Lemma (to prove in Phase P2):** *If decoupling ≤ ε and EF trend ≥ δ > 0 across ≥ 3 consecutive Z2 sessions, then the aerobic‑economy score is monotone‑increasing over the Base‑Ox period.*
>
> File stub: `lean/Cultivation/Running/BaseOx.lean` (imported in CI but currently admitted with `by sorry`).

---

### 📎 Next Steps

1. **Implement plan updates (Pace guidance, Cadence focus, Instrumentation).**
2. **Execute Week 2 focusing strictly on Z2 HR adherence, letting pace fall where necessary.**
3. **Monitor pace achieved during correct Z2 HR execution to establish a realistic baseline aerobic pace.**
4. **Continue cadence improvement efforts.**
5. **Investigate Wk18 Tue HR override data sparsity.**

With the mileage correctly scaled *and* instrumentation / validation sections restored, the Base‑Ox plan is fully documented and self‑governing.
