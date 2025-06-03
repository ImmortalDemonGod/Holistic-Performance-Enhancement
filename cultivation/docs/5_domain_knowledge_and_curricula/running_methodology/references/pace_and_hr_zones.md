---
title: Heart-Rate & Pace Zones – Reference
status: v1 · 2025-05-01
authors: tomriddle1, openai-o3
---

> **Why this doc?**  
> Two popular zone systems (AeT-anchored vs. threshold-anchored) share the *same names* but assign **different BPM ranges**.  
> This file declares which one the Cultivation repo uses at any given training block, documents the alternative, and shows exactly where the values feed into code, CI, and devices.

---

## 0 · TL;DR

| Block | Zone model in force | Z2 ceiling | Key purpose |
|-------|--------------------|------------|-------------|
| **Base-Ox (Weeks 1-4)** | **AeT-anchored** | **160 bpm** | Aerobic-base, mitochondrial density |
| Tempo-Dur / later blocks | Threshold-anchored (Friel) | 0.89 × LTHR ≈ 166 bpm | LT & muscular endurance work |

Stay below **160 bpm** for *all* steady running during Base-Ox.  
You’ll switch to the threshold-anchored table only when the calendar enters Tempo-Dur.

---

## 1 · Zone Tables

### 1.1 AeT-Anchored Scheme *(CURRENT)*

| Zone (doc label) | BPM lo | BPM hi | %HR<sub>max</sub> | Typical usage |
|------------------|-------:|-------:|-------------------|---------------|
| **Z1 Recovery**  |  0 | 145 | ≤ 72 % | Walks, cooldowns |
| **Z2 Aerobic Base** | 145 | **160** | 72–80 % | Base-Ox steady runs |
| **Z3 Tempo “gray”** | 161 | 175 | 80–87 % | Progression finishers |
| **Z4 Threshold** | 176 | 186 | 88–92 % | Cruise-tempo, LT reps |
| **Z5 VO₂ / Anaerobic** | 187 | 201 | 93–100 % | Strides, VO₂ intervals |

*Pace bands* for 10 °C (flat terrain, 8 : 03 min·mi aerobic speed):

9 : 24  = Z1 ·  145 bpm
6 : 51 – 8 : 43 = Z2 · 145-160 bpm
8 : 28 – 6 : 50 = Z3 · 161-175 bpm
8 : 27 – 8 : 07 = Z4 · 176-186 bpm
< 8 : 07          Z5 · ≥187 bpm

### 1.2 Threshold-Anchored Scheme (Friel / Coggan)

| Zone | BPM lo | BPM hi | %HR<sub>max</sub> | Note |
|------|-------:|-------:|-------------------|------|
| Z1 (Recovery) | ≤ 158 | — | ≤ 79 % | *Maps to Z2 in many runner plans!* |
| Z2 (Aerobic)  | 158 | 166 | 79–83 % | Above AeT; “moderate endurance” |
| Z3 (Tempo)    | 167 | 175 | 83–87 % | Tempo maintenance |
| Z4 (Threshold)| 177 | 184 | 88–92 % | @ LTHR – long LT efforts |
| Z5 (VO₂)      | ≥ 186 | 201 | 93–100 % | Max-aerobic |

*Anchor values used*: **HR<sub>max</sub> 201 bpm** · **LTHR 186 bpm**

---

## 2 · Choosing the Right Model

| Question | AeT-anchored (Base-Ox) | Threshold-anchored (later) |
|----------|-----------------------|----------------------------|
| Main energy system trained | Oxidative, fat-max | Lactate clearance & buffering |
| Primary KPI | EF ↑ ≥ 5 %, HR-drift < 7 % | LT-pace ↓, decoupling < 5 % |
| Zone 2 upper limit | **160 bpm** | 166 bpm |
| Scripts depending on it | `metrics.py`, `fatigue_watch.py`, CI gates, Garmin alerts | none yet (to be wired for Tempo-Dur) |

**Rule of thumb:**  
*If the block objective is “more mitochondria”, stay with AeT zones.  
When the objective shifts to “push LT/VO₂”, swap to the threshold table.*

---

## 3 · Integration Points in the Repo

| File / Tool | What it does with zones | Action needed when switching |
|-------------|------------------------|------------------------------|
| `cultivation/data/zones_personal.yml` | Source-of-truth for scripts | Update `model:` key to `threshold` |
| `scripts/running/metrics.py` | Flags HR-cap breaches, calculates EF | Run with `--model aeT` (default) or `--model threshold` |
| `.github/workflows/run-metrics.yml` | Fails build if EF Δ negative *for active model* | No change unless model toggles |
| Garmin/Coros watch profile | HR high alert | Keep 160 bpm during Base-Ox, raise to 166 bpm later |
| Dashboard charts | Zone-dwell plots | Auto-update from `zones_personal.yml` |

---

## 4 · How to Switch Models (check-list)

1. Edit `zones_personal.yml` → `model: threshold`  
   (script will recalc break-points from stored HR<sub>max</sub> + LTHR)
2. Re-run `python scripts/running/metrics.py --recalc-zones`.
3. Push commit; CI will regenerate weekly metrics parquet.
4. Update watch HR alert to new Z2 ceiling (166 bpm).
5. Verify next run file shows new zone labels in the summary.

---

## 5 · Footnotes & References

* **AeT (VT1)** – first ventilatory threshold, marks maximal sustainable fat-oxidation intensity.  
* **LTHR (LT2)** – point where lactate ≥ 4 mmol·L⁻¹; Friel zones anchor here.  
* Seiler, S. (2010) “Intensity distribution…” *Scand. J. Med. Sci. Sports*.  
* Friel, J. (2012) *The Triathlete’s Training Bible*, 4th ed.

---

### Change-log

| Date | Author | Note |
|------|--------|------|
| 2025-05-01 | tomriddle1 / o3 | Initial commit – dual-model spec |
