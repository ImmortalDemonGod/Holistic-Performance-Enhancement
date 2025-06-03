<!--  File: docs/2_requirements/math_stack.md  -->
# 📐 The Mathematics Stack for **Holistic Performance Enhancement (Cultivation)**  

> *“Mathematics is the connective tissue that lets running shoes talk to RNA helices and black‑hole orbits.”*  

---

### How to use this page  
1. **Index → Code** Each row shows *exactly* which repo paths or Lean imports need that math.  
2. **Timeline link** Pair this list with the live roadmap:  
   [`docs/3_design_and_architecture/roadmap_vSigma.md`](../../3_design_and_architecture/roadmap_vSigma.md).  
3. **Lean tags** Bullets in *lavender* ⊚ tell you which `mathlib4` modules (or upcoming community files) we’ll import when proofs start.

---

| Layer / Domain | Core Mathematics | Concrete Repo Touch‑Points | ⊚ Lean / Proof Imports |
|---|---|---|---|
| **1. Running‑Performance Analytics** | • Calculus & ODEs – HR‑recovery, VO₂ kinetics<br>• Numerical methods – Runge‑Kutta, LS parameter fit<br>• Time‑series stats – STL, ARIMA, change‑point<br>• Biomechanics physics – impulse–momentum | `scripts/running/`  `notebooks/running/*` | ⊚ `Mathlib.Analysis.ODE`, `Mathlib.Data.Real.Basic` |
| **2. Biological Modeling & Lab Data** | • Deterministic ODE/PDE – logistic, reaction–diffusion<br>• Stochastic processes – birth‑death, CME<br>• Linear algebra & PCA – omics reduction<br>• Bayesian inference – hierarchical wet‑lab models | `docs/5_mathematical_biology/*`  `scripts/biology/*` | ⊚ `Mathlib.MeasureTheory`, `Mathlib.Topology` |
| **3. Software‑Engineering Metrics** | • Discrete math & graph theory – call‑graph DAGs, cyclomatic complexity<br>• Information theory – Shannon entropy of code deltas<br>• Statistical process control – Shewhart/χ² charts | `scripts/software/commit_metrics.py`  `docs/4_analysis/analysis_overview.md` | ⊚ `Mathlib.Combinatorics.Graph`, `Mathlib.Data.Finset` |
| **4. Synergy / Potential Engine** | • Multivariate stats – PCA/CCA/ICA fusion<br>• Convex optimization – resource allocation via cvxpy<br>• Control theory – PID, MPC scheduling<br>• Causal inference – DAGs, do‑calculus | `scripts/synergy/calculate_synergy.py`  `notebooks/synergy/*`<br>*(Synergy‑entropy formula now lives in* [`synergy_concept.md`](../../0_vision_and_strategy/archive/synergy_concept.md)*).* | ⊚ `Mathlib.Analysis.Convex`, `Mathlib.Probability` |
| **5. Machine‑Learning Layer** | • Linear algebra / matrix calculus – grads, SVD<br>• Optimization – SGD, regularizers<br>• Information geometry – KL, Fisher | ML prototypes in `notebooks/*`  RL agent in `scripts/synergy/rl_agent.py` | ⊚ `Std.Data.Matrix`, *(future)* `Mathlib.Geometry.Manifold` |
| **6. ARC / Abstract‑Reasoning Toolkit**  🆕 | • Combinatorics & finite automata – grid / state machines<br>• SAT/SMT basics – constraint satisfaction<br>• Circuit complexity – Boolean algebra, graph flows | `scripts/arc/*`  `notebooks/arc/*` | ⊚ `Mathlib.Logic.Basic`, `Mathlib.Data.Bool`, `Mathlib.Tactic` |
| **7. Formal Verification (Lean 4)** | • Dependent type theory – Lean kernel<br>• Classical algebra, analysis, topology<br>• (Optional) Category theory – compositional proofs | `lean/` workspace – stability of logistic harvest, PID proofs, ARC solvers | ⊚ whole `mathlib4` universe 🌌 |
| **8. Long‑Horizon / Space & Game‑Theory** | • Celestial mechanics – two‑ & n‑body, Lambert<br>• High‑order ODE solvers – symplectic, adaptive RK<br>• Game theory & economics – multi‑agent resource allocation<br>• PBH perturbation statistics – gravitational signature models | Future `scripts/space/*`  `notebooks/space/*`  design refs in `docs/3_design/` | ⊚ custom theories + `Mathlib.Analysis.Calculus` |

---

## Cross‑Domain Glue Math  
*Entropy‑based synergy coefficient* and the *global potential function* are specified in **Background → Synergy Concept**.  
Deep‑link here when you need formulas, proofs, or derivative work.

---

## Next‑Up TODOs
1. **Anchor links** – add `<a id="running-ode">`‑style IDs per bullet for deep linking.  
2. **Kanban tags** – create *Needs‑Math* cards for upcoming PID proof & PCA dashboard.  
3. **CI badge** – embed GitHub‑Actions build status atop this file.

*Last updated: 2025‑04‑18 • synced to Road‑map v Σ*  
