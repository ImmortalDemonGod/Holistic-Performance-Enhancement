# Analysis Overview
🧙🏾‍♂️ **The “4_analysis/” layer = the living lab notebook of the repo**
Think of `cultivation/docs/4_analysis/` as the **results vault**: every time data flow runs, a test finishes, or a proof sheds light on real‑world performance, the distilled *story* lands here.

Below’s a practical blueprint of **what belongs there**, **how to structure it**, and **how it evolves phase‑by‑phase**.

---

## 1 Purpose

| Goal | What it means in practice |
|------|---------------------------|
| **Archive results** | Permanent record of each experiment/run (plots, tables, key numbers, verbal conclusions). |
| **Compare iterations** | Side‑by‑side snapshots: *baseline → intervention → post‑tuning* so we can see deltas. |
| **Drive decisions** | Each analysis file should end with a “Next action” or “Hypothesis confirmed/denied” block. |
| **Close the doc–code loop** | Link notebooks + scripts outputs back to narrative; update road‑map gates automatically. |

---

## 2 Folder Skeleton

```
4_analysis/
├── 0_index.md           # running TOC, links to latest reports
├── 1_running/
│   ├── baseline_week0.md
│   ├── intervention_A_interval_training.md
│   └── ...
├── 2_biology/
│   ├── reading_retention_q1.md
│   └── ...
├── 3_software/
│   ├── commit_metrics_Q12025.md
│   └── ...
├── 4_synergy/
│   ├── pilot_synergy_calc_r1.md
│   ├── rl_agent_vs_pid_ablation.md
│   └── ...
└── assets/              # static PNG/SVG exports referenced by the markdown
```

> **Convention:** filename = `<topic>_<date-or-version>.md` to keep git diffs readable.

---

## 3 Template for an Analysis File

```markdown
# ⏱️ [Running] Weekly Baseline – Week 0 (2025‑04‑20)

## Data Snapshot
| Metric | Value |
|--------|------:|
| Total distance (km) | 43.2 |
| Average pace (min/km) | 5:45 |
| Resting HR (bpm) | 57 |

*(CSV: `../../data/running/weekly_2025‑04‑20.csv`)*

## Visuals
![pace](../assets/pace_week0.png)

## Interpretation
1. HR drift suggests endurance ceiling ~10 % below target.
2. High variance in Tuesday intervals → candidate for form drills.

## Next Action
- **Plan:** introduce 2 × 12 min tempo on Tue/Thu, re‑measure Week 2.
- **CI Tag:** `analysis:running:baseline_W0` (lets GitHub Action link future runs).
```

Save the template as `analysis_template.md` for quick copy‑paste.

---

## 4 How Analyses Are Generated

| Source | Pipeline | Landing Spot |
|--------|----------|--------------|
| **Notebooks** (`notebooks/*/*.ipynb`) | executed via `nbconvert` in CI; export HTML or PNG plots | attach images to `assets/`, embed summary numbers in markdown |
| **Scripts** (`scripts/*/*.py`) | CLI flag `--report md` dumps a markdown snippet | snippet appended to the appropriate analysis file by CI |
| **Lean proofs** | `lake exe export_summary` (future) → JSON | human‑readable proof stats recorded in `software/` analyses |
| **Synergy score calc** | nightly action → CSV + bar chart | weekly roll‑up file in `4_synergy/` |

---

## 5 Phase‑by‑Phase Minimum Content

| Road‑map Phase | Must‑have analyses before gate ✓ |
|----------------|----------------------------------|
| **P0 – Bootstrap** | • Running *weekly baseline*<br>• RNA loader sanity stats (row count, NA %) |
| **P1 – Dynamics** | • VO₂/HR ODE fit residuals<br>• RNA structure EDA plots |
| **P2 – Control + Causal** | • PID schedule vs actual adherence plot<br>• DAG ATE table |
| **P3 – Optimisation** | • Time‑allocator before/after resource realloc<br>• PCA scree & biplot |
| **P4+** | escalate similarly (RL agent reward curves, GNN loss, PBH detector ROC, …) |

CI can assert these files exist and include a `## Next Action` section → green light the phase transition.

---

## 6 Tips & Best Practices

1. **One truth‑file per analysis** – don’t tuck results into notebooks only; export canonical numbers here.
2. **Static assets** – render plots to PNG/SVG once; keep markdown light so Git diffs stay small.
3. **Link upstream commit** – start each file with `> Analysis of commit abc123` for perfect provenance.
4. **Automate boring stuff** – a small `report_builder.py` can scan `data/`, make tables, and append to markdown.
5. **Interpretation first, math second** – numbers are useless without the 2‑3 bullet takeaway.

---

## Key Analyses and Reports

Below are the most important analysis documents and templates available in this section:

- [Running Analysis Template](../meta/templates/running_analysis_template.md): Use this template to generate new run or experiment reports.
- [Week 21 RPE10 Benchmark Analysis](week21_rpe10_benchmark_analysis.md): In-depth analysis of the week 21 benchmark run, including methodology, results, and actionable insights.
- [2025-04-27 Run Report](run_reports/2025_04_27_run_report.md): Example of a detailed run report for April 27, 2025.
- [2025-05-01 Run Report](run_reports/2025_05_01_run_report.md): Example of a detailed run report for May 1, 2025.
- [2025-05-11 Run Report](run_reports/2025_05_11_run_report.md): Example of a detailed run report for May 11, 2025.
- [2025-05-17 Run Report](run_reports/2025_05_17_run_report.md): Example of a detailed run report for May 17, 2025.
- [2025-05-24 Run Report](run_reports/2025_05_24_run_report.md): Example of a detailed run report for May 24, 2025.

---

### TL;DR

Put *every* finalized result, figure, and conclusion into `docs/4_analysis/`, organized by domain ⇒ date/version. Each file becomes a bite‑sized lab report that tells (a) what you did, (b) what you saw, (c) what you’ll do next. When CI or humans rerun things, they create a fresh file or update the existing one, keeping the project’s scientific memory intact.

Let me know if you’d like me to scaffold the folder, add a template file, or wire a CI snippet to auto‑publish notebook summaries 🚀
### Short answer
**Yes — but don’t drop the whole kitchen sink into `4_analysis/`**. Treat your old Kaggle comps and legacy software as **reference case‑studies and data sources**, then surface only the distilled *lessons* (and any reusable assets) inside the Cultivation tree.

---

## Where each artifact belongs

| Artifact type | Recommended home | Why |
|---------------|------------------|-----|
| **Raw notebooks / code** from a Kaggle comp | `archive/kaggle/<competition‑slug>/` <br>*or* link as a Git submodule | Keeps giant notebooks, CSVs, checkpoints out of the main history, but still retrievable. |
| **Finished model weights, datasets** | `data/legacy/kaggle_<slug>/` (if < 100 MB) <br>*otherwise:* reference an external storage URL in README | Large binaries shouldn’t live in Git LFS unless you’ll actually re‑use them. |
| **“Post‑mortem” write‑up** (what worked, what didn’t) | `docs/4_analysis/legacy/` <br>e.g. `kaggle_titanic_2019.md` | Fits the analysis pattern: experiment → metric → take‑away. |
| **Reusable helper libs / utilities** | Promote into `scripts/legacy_utils/` or a dedicated PyPI package | Anything you expect to call from current pipelines should be importable, test‑covered, CI’d. |
| **Old software projects** (complete repos) | Leave in their own repos; add links + high‑level summary in `docs/1_background/related_projects.md` | Avoid code rot & dependency conflicts inside Cultivation. |

> **Folder to add:**
> ```
> cultivation/
> └── archive/
>     ├── kaggle/
>     └── software/
> ```
> Git‑ignore large files; treat it as read‑only provenance.

---

## How to make legacy work *useful* for Cultivation

1. **Extract metrics that map to current goals**
   *Example:* Your Kaggle “predict heart‑rate variability” model → pull final leaderboard RMSE and any feature‑engineering tricks; cite them in `docs/4_analysis/legacy/` as a benchmark for the new running‑VO₂ project.

2. **Harvest proven code snippets**
   - Data loaders, cross‑validation wrappers, LightGBM tuning grids, etc.
   - Relocate to `scripts/common/` with tests so current CI keeps them alive.

3. **Convert lessons into *design constraints* or *requirements***
   - If a past project taught you “never trust early stopping without a hold‑out fold,” note that in `docs/2_requirements/testing-requirements.md`.

4. **Tag legacy analyses clearly**
   - Prefix filenames with `legacy_` or nest them under `legacy/` so future readers know these results don’t come from the new integrated pipeline.

5. **Set up CI to ignore legacy folders**
   - Add path filters so broken notebooks in `archive/` don’t fail the build.

---

## Decision checklist

| Question | If **YES** | If **NO** |
|----------|------------|-----------|
| Will I re‑use this model/data in the next 12 months? | Pull into `data/legacy/` + write a short analysis. | Leave in an external repo; just hyperlink it. |
| Does this code teach a unique lesson that informs the roadmap? | Summarise in `docs/4_analysis/legacy/`. | Skip or just note briefly in background. |
| Is the repository huge (> 500 MB) or has exotic dependencies? | Keep outside; use a submodule or ZIP link. | Safe to archive inside `archive/`. |

---

### TL;DR
*Archive the bulky stuff, surface the insights.*
- **Archive or link** full notebooks / repos so you can reproduce them.
- **Write concise “what we learned” reports** in `docs/4_analysis/legacy/` following the same template as new analyses.
- **Promote only reusable utilities** into `scripts/` and protect them with CI.

That way the Cultivation project stays lean, coherent, and forward‑focused—while still harvesting every drop of value from your Kaggle glory days and past software adventures. 🚀
