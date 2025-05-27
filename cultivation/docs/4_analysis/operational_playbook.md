<!--  
File: **docs/4_analysis/operational_playbook.md**  
Status: **✅ APPROVED — v Σ 1.0 (Full‑stack experience guide)**  
This supersedes every previous “playbook” draft and is now the canonical day‑to‑day / year‑to‑year usage manual for the fully‑integrated **Cultivation** platform.  
-->

# 🚀 Operational Playbook — *Life inside Cultivation*  
*A practitioner‑oriented walkthrough of what it feels like to **live** with the entire stack once P0 → P2 (and beyond) are in production.*

---

## 0 · Reading Compass

| Section | Why read it? | Links |
|---------|--------------|-------|
| **§ 1 Quick Mental Model** | 5‑minute orientation to how the whole engine breathes | C4 diagram → `docs/3_design/architecture_overview.md` |
| **§ 2 Personas** | What “using Cultivation” looks like for Dev, Researcher, Coach | — |
| **§ 3 Daily Loop** | Minute‑by‑minute UX (CI, dashboards, CLI, IDE) | integrates: Task Master, DocInsight, Flash‑Memory |
| **§ 4 Weekly → Monthly Rhythm** | Rituals that keep data, models, proofs, and people fresh | Synergy retro, Lean proof triage, PID weight update |
| **§ 5 Annual Evolution** | How phases unfold, how debt & complexity stay tamed | mapped to v Σ roadmap |
| **§ 6 Observability & Incident Flow** | Dashboards, alerts, SLOs, fire‑drills | Grafana, Slack automations |
| **§ 7 Maintenance Ledger** | Quarterly checklist: rotate keys, pin deps, archive data | — |
| **§ 8 Pain Points ↔ Mitigations** | Pre‑mortem table of likely frictions & built‑ins that solve them | — |
| **§ 9 On/Off‑Boarding** | Spin up a new teammate or shell out to sabbatical gracefully | template scripts |
| **§ 10 FAQ** | “Can I skip a run? What if DocInsight dies?” | quick answers |
| **§ A Glossary** | All internal acronyms in one spot | — |

---

## 1 · Quick Mental Model 🧠

1. **ETLs** (Running, Software, Literature, …) shovel raw artefacts → tidy Parquets in `data/`.  
2. **Synergy Engine** fuses cross‑domain deltas into weekly `synergy_score.parquet`.  
3. **Potential Π** converts raw KPIs + synergy into a scalar “growth capacity” snapshot.  
4. **PID / RL Scheduler** reads Π → emits `daily_plan.json`.  
5. **Task Master** exposes that plan as an actionable, version‑controlled backlog.  
6. **Flash‑Memory Layer** guarantees humans learn what code and proofs already “know”.  
7. **Lean proofs** compile on every push, acting as a maths CI guard‑rail.  
8. **DocInsight** turns PDFs into instant answers + novelty scores feeding cognition metrics.  
9. **Dashboards & Slack digest** surface the only numbers that matter: ΔΠ & today’s top task.

Everything is either **code‑reviewed** (Git) or **task‑reviewed** (Task Master); nothing
important hides in Notion or email.

---

## 2 · Personas & Touch‑points

| Persona | Primary Interface | Golden Path |
|---------|------------------|-------------|
| **Developer** | VS Code + GitHub PRs | Write code → CI green → Synergy badge shows ΔB\_pred drop → merge |
| **Researcher / Reader** | `lit-search` CLI + DocInsight UI | Query paper → summary & novelty → note TL;DR → reading_stats tick |
| **Athlete / Coach** | Garmin auto‑sync + Dashboard | Morning run auto‑ingested → pace rolling‑mean displayed → scheduler adapts afternoon coding load |
| **Formal Methods Reviewer** | Lean VS Code extension + CI | Prove lemma → push → Lean job passes → analytic script now trustable |

*(One human can wear all four hats in a solo setup; personas help isolate UX needs.)*

---

## 3 · The Daily Loop Example (adjust against scheduling) ⏰

| Slot | Action | System Autonomy | Human Pay‑off |
|------|--------|-----------------|---------------|
| **06:30** | Finish run; phone syncs | Garmin → `process_run_data.py` via cron hook | Breakfast dashboard already shows new VO₂ & fatigue score |
| **08:45** | `tm next` | PID plan rendered as Task #1412 “Read *RNA diffusion* (15 min)” | No mental thrash; first high‑leverage task is obvious |
| **09:15–11:30** | Deep‑focus coding | Git pre‑commit tags LOC Δ; Lean‑Copilot offers proof skeletons | CI comment predicts **+0.4 % Π** before PR review |
| **11:45** | `lit-search "RNA folding diffusion"` | DocInsight returns 3‑bullet summary & novelty 0.72 | 5‑minute skim captured in note; C(t) cognition channel spikes |
| **14:20** | Merge PR | GitHub Action runs Synergy calc & Flash‑Memory vet | Slack bot posts: “ΔΠ +0.27 % day‑over‑day” |
| **16:30** | Mark Task #1412 done | `post_task_complete.py` auto‑makes flashcard if `[[fc]]` tag | Knowledge enters spaced‑repetition queue |
| **23:00** | Nightly CI cron | Literature fetch, DocInsight re‑index, synergy retro draft | Tomorrow’s `tm next` already prioritised |

*End‑user UI surface area*: one terminal tab, one Slack channel, one dashboard tab.

---

## 4 · Weekly → Monthly Rhythm 📆

| Cadence | Trigger (CI job) | Artefact Reviewed | Decisions |
|---------|-----------------|-------------------|-----------|
| **Mon 06:00** | `literature-nightly` | New PDFs diff + LanceDB delta | Tag papers, spawn reading tasks |
| **Tue 09:00** | `synergy_weekly.md` | Bar‑chart of cross‑domain \(S_{A→B}\) | Re‑allocate scheduler weights if a channel flat‑lines |
| **Wed 16:00** | `lean.yml` proof report | List of failing lemmas | Hot‑fix algorithms or relax theorem in ADR |
| **Fri 14:00** | `flashcards-nightly` review stats | FSRS stability curve | Cull stale cards, add missing `[[fc]]` hooks |
| **Last work‑day / month** | `update_potential_weights.py` | Regression report + R² | Approve new Π weights; record in ADR‑03 |

Every cadence artefact is a Markdown file in `docs/4_analysis/` with *Next Action* bullets.
Passing all five checks is the Phase‑gate to proceed on roadmap.

---

## 5 · Year‑to‑Year Evolution 🌳

| Year 0 → 1 | Year 1 → 2 | Year 2 → 3 |
|------------|-----------|-----------|
| **Stability & CI**<br>ETLs, Lean core, PID baseline | **Optimisation**<br>RL agent beats PID, SARIMA baseline replaced by GNN, auto‑sharding LanceDB | **Scale & New Domains**<br>Astrodynamics ETL, ARC solver, cluster DB, 75 % Lean coverage |
| **Data Volume**: 5 k PDFs • 1 M GPS pts | 30 k PDFs • 6 M GPS pts | 120 k PDFs • 25 M GPS pts |
| **Tech‑debt posture**: strict CI gates, debt negligible | Debt appears in vector index & CI queue time → weekly pruning job | Move to distributed vector DB & queued notebook execution |
| **User sentiment**: “Everything just works.” | “We need dashboard curation.” | “The system suggests strategies I never imagined.” |

---

## 6 · Observability, SLOs & Incidents 🩺

| Subsystem | SLO | Alert Channel | Auto‑Remediation |
|-----------|-----|---------------|------------------|
| **CI** (lint+tests) | 95 % runs < 5 min | `#ci-slow` | Parallel matrix fan‑out |
| **DocInsight API** | 99 % p95 < 2 s | `#docinsight-alerts` | Kubernetes restart; ETL‑B switches to ingest‑only mode |
| **Synergy Calc** | weekly job < 10 min | `#synergy-fail` | Push job to bigger runner; downgrade to rolling‑mean |
| **Flash‑Memory Build** | nightly job success | `#flashcards-alerts` | Re‑order decks to cut build time |

*Grafana dashboards* aggregate Prometheus exporters; single “All green” banner embedded in home README.

---

## 7 · Maintenance Ledger (Quarterly)

1. **Rotate API keys** (Task Master, OpenAI, DocInsight).  
2. **Freeze & repin deps** (`pip‑compile --upgrade --resolver backtracking`).  
3. **Archive** raw GPS & LanceDB shards > 18 m old ⇒ S3 Glacier.  
4. **Schema drift audit** via Great Expectations `docs/2_requirements/schemas/`.  
5. **Coverage delta** — Lean LOC up by ≥ 5 % vs. last quarter.  
6. **Close / refresh ADR backlog** (label `adr/stale`).  

*CI template `maintenance-check.yml` enforces completion before new phase starts.*

---

## 8 · Pain Points → Built‑in Mitigations

| Pain | Guard‑rail / Fix |
|------|------------------|
| **DocInsight down** | ETL‑B sets `avg_novelty = 0`; Slack alert; daily plan bias reduced on cognition channel so schedule remains feasible. |
| **Data explosion** | Weekly Spark job prunes LanceDB embeddings older than 9 m; gzip GPS; Thanos tier‑down prom metrics. |
| **Model drift** | `drift_tests.py` in CI asserts forecast RMSE < threshold; on fail scheduler pins last known good PID weights. |
| **Proof rot** | `lake build` gate; failing proofs add `proof‑broken` label → merge blocked unless hot‑fixed or marked experimental. |
| **Human overload** | PID caps tasks/day; Flash‑Memory FSRS algorithm throttles new‑card rate; dashboards highlight Δ not raw counts. |

---

## 9 · On‑ & Off‑Boarding Scripts

```bash
# New contributor bootstrap
curl -sL https://cultivation.sh/install | bash
tm init --clone https://github.com/org/cultivation.git
make first-run           # installs venv, pre-commit, Lean tool‑chain

# Graceful off‑boarding
tm pause --user alice --handoff bob
tm-fc export --user alice > alice_cards.apkg
gh api -X DELETE /orgs/.../members/alice
```

CI auto‑tests that `install` completes on Ubuntu, macOS, Windows‑WSL.

---

## 10 · FAQ 🙋

| Q | A |
|---|---|
| “I skipped running for two weeks; will scheduler freak out?” | No. PID error integrates but clamps; RL explores compensatory tasks. |
| “Can I temporarily disable Lean proofs for a refactor?” | Label PR `skip‑lean`; CI soft‑fails but warns; must re‑enable before phase gate. |
| “DocInsight novelty looks noisy—tweakable?” | Adjust cosine threshold in `synergy_config.yaml`; contract tests still assert 0 ≤ novelty ≤ 1. |
| “Are big GPUs mandatory?” | Not until P4; PID + SARIMA run on CPU GitHub runner. |
| “How do I add a new wearable brand?” | Drop `parse_<brand>.py` in `scripts/running/`, register parser lookup. Schema test will flag missing fields. |

---

## A · Glossary (quick refs)

**Π** — global Potential scalar.  
**C(t)** — cognitive channel of Π fed by literature stats.  
**Task Master** — Git‑backed task tracker / CLI.  
**DocInsight** — vendored RAG micro‑service.  
**ETL‑R/S/B** — running / software / literature extract‑transform‑load jobs.  
**FSRS** — flashcard spaced repetition scheduler.  
**Lean** — proof assistant; CI gate.  
**ΔB\_pred** — baseline behaviour prediction for synergy delta.  

---

### ✨ What Success Feels Like

*Morning*: dashboard already knows today’s ideal mix of splits, commits, and papers.  
*Afternoon*: merge a PR → CI proves maths, recalculates synergy, nudges Potential upward.  
*Evening*: Slack digest says “+0.31 % growth capacity”; you close laptop guilt‑free.  
*Quarter*: phase gate passes; new domain spins up without chaos.  
*Year*: system suggests strategies you hadn’t imagined — and backs them with proofs.

That is **life inside Cultivation**: a self‑reinforcing ecosystem where every jog, line of code, and paper summary compounds into long‑horizon, quantifiable growth — with guard‑rails that keep entropy at bay.
