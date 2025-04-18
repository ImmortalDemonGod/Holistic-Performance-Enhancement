# ✨ Lean 4 & Formal Verification Guide for *Cultivation*

> *“Proof is the ultimate integration test.”*

This guide lives in **`docs/2_requirements/lean_guide.md`** (rendered here in the canvas for live edits). It explains **why** we use Lean 4, **how** to set it up, **where** proofs belong in the repo, and the **coding conventions** that make proofs readable and reusable across running, biology, software‑metrics, PBH simulation, and ARC‑style reasoning.

---
## 1  Why Lean 4?
* **Unified mathematics** ― `mathlib4` already covers calculus, linear algebra, probability, graph theory… the same stack our project needs.
* **Executable docs** ― proofs double as checked documentation for algorithms in `scripts/`.
* **Lean Copilot** ― AI‑assist speeds scaffolding while keeping the kernel as single source of truth.

---
## 2  Folder & Namespace Layout
```
cultivation/
└─ lean/
   ├─ README.md          -- quick setup
   ├─ Cultivation/       -- top‑level namespace
   │   ├─ Running.lean   -- VO₂ ODE proofs
   │   ├─ Biology.lean   -- logistic / budworm theorems
   │   ├─ Synergy.lean   -- PID, convex optimisation
   │   ├─ ARC/*.lean     -- combinatorics helpers
   │   └─ Space/*.lean   -- n‑body error bounds
   └─ lakefile.lean      -- Lake build config
```
*One namespace per domain* keeps imports light. Cross‑domain lemmas go in `Cultivation/Common.lean`.

---
## 3  Setup Steps
1. **Install tool‑chain**  (≥ Lean 4.3):
   ```bash
   lake update
   lake exe cache get
   ```
2. **Editor support**: VS Code + `lean4` extension **or** Neovim + `lean.nvim`.
3. **Lean Copilot**: ensure `$OPENAI_API_KEY` is in env. Run:
   ```bash
   lake exe copilot login
   lake exe copilot enable
   ```
   Copilot suggestions appear as *ghost text* – always `Ctrl+Enter` to accept only after reading.

---
## 4  Proof Conventions
| Topic | Convention |
|---|---|
| **Imports** | Use `open Real BigOperators`; avoid `open Classical` in library files. |
| **Names** | `deriv_hr_recovery` not `lem1`. Use snake_case for lemmas, CamelCase for structures. |
| **Comments** | Top‑docstring + inline `--` for non‑obvious steps. |
| **Tactics Order** | `simp`, `ring`, `linarith`, `nlinarith`, `field_simp` before heavier tactics. |
| **Automation** | Wrap long tactic chains with `by` blocks; expose helper lemmas so Copilot can reuse them. |
| **Units** | State physical units in comments; proofs remain dimensionless unless necessary. |

---
## 5  Road‑map ↔ Lean
| Road‑map Phase | First Proof Targets |
|---|---|
| 0‑2 mo | arithmetic, list, matrix basics (`Common.lean`) |
| 2‑6 mo | **Running.lean** – existence & uniqueness of HR‑recovery ODE, logistic stability |
| 6‑10 mo | **Synergy.lean** – PID closed‑loop boundedness |
| 10‑16 mo | **Optimization.lean** – KKT conditions for time allocator |
| 18‑24 mo | **ARC/Grid.lean** – decidability of small automata problems |
| 24 + mo | **Space/TwoBody.lean** – error bound for symplectic integrator |

---
## 6  Interfacing Lean ⇄ Python
* Use `lake exe export_lean` → generates `.olean` & compile JSON schemas.
* For numerical algorithms: prove correctness in Lean, implement in Python, then unit‑test Python against Lean‑generated reference values.
* Experimental: `lean‑python` binding can call Lean kernel at runtime; we’ll evaluate when PID proofs are stable.

---
## 7  CI / CD
* **GitHub Action** `.github/workflows/lean.yml` runs `lake build` + `lake test`.
* Cache `mathlib4` to save minutes.
* Fail PR if any proof breaks (soft‑fail allowed on experimental namespaces `ARC/`, `Space/`).

---
## 8  LeanDojo Quick‑Start
LeanDojo lets us trace external Lean repos, mine premises, and spin up a **gym‑like REPL** so Python agents (or ReProver) can interact with proofs.

| Task | Command | Notes |
|---|---|---|
| **Install** | `pip install lean-dojo` | Needs Python ≥3.9 and `elan` tool‑chain. |
| **Trace repo** | `python -m lean_dojo.trace https://github.com/leanprover-community/mathlib4 <commit>` | Generates `traced_<repo>/` with AST, dependency graphs, proof states. |
| **Launch REPL** | `python -m lean_dojo.repl <Theorem>` | Opens an interactive shell; try tactics and inspect goals programmatically. |
| **Retrieve premises** | `from lean_dojo import PremiseRetriever` | Uses LeanDojo’s embeddings to get candidates for ReProver / Copilot. |
| **Cache dir** | Set `CACHE_DIR=~/.cache/lean_dojo` | Traced repos are cached + shareable (see project README). |

> **Tip — Build once, use everywhere:** Add a Makefile target `make trace` that hits all repos in `requirements/repos.txt` so CI populates the cache.

---

## 9  Learning Resources
* *Mathematics in Lean* (Lean 4 edition) – quick intro.
* `mathlib4` docs – https://leanprover-community.github.io
* Zulip – `#new members` friendly help.

---
### 🔗 Changelog
| Date | Change |
|---|---|
| 2025‑04‑18 | Initial draft – imported to canvas by ChatGPT o3 |

---
> **Next** → open an issue *Lean‑Onboarding‑Tasks* and tag `good first proof` for each Phase‑0 lemma.

