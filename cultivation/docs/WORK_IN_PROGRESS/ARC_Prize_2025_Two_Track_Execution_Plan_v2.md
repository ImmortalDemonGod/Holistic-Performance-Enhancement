# **ARC Prize 2025: The Two-Track Execution Plan**

**Document Version:** 2.0
**Date:** 2025-06-13
**Status:** Canonical Tactical Plan
**Point of Contact:** The Architect (You) & The `george` Creator

### I. Executive Summary

This document is the official, tactical execution plan for the five-month ARC Prize 2025 sprint. It details the phased milestones, team structure, specific goals, and risk mitigation strategies for the two-track development approach.

This plan is the direct implementation of the high-level vision and justification outlined in the project's master strategic charter: **`Strategic_Integration_of_ARC_Prize_2025_v2.0.md`**. All strategic "Why" questions should be referred to that document; this document focuses exclusively on the tactical "What" and "When."

Our mission is to execute a **two-track, parallel development strategy** to produce a final, hybrid solver codenamed **ARCHIMEDES**. This system will combine the high-performance, end-to-end `JARC-Reactor` model with the deliberate, compositional reasoning of the `george` engine, creating a system more capable than the sum of its parts.

### II. The Five-Month Two-Track Execution Plan

This plan outlines the parallel workstreams, deliverables, and timelines for the duration of the competition.

#### **2.1. Key Milestones & Deliverables Summary**

| Phase       | End Date (Approx) | Key Milestone / Deliverable                               | Purpose                                                 |
| :---------- | :---------------- | :-------------------------------------------------------- | :------------------------------------------------------ |
| **Phase 1** | End of Month 2    | **SUBMIT:** `JARC-Reactor` v1                             | Establish baseline, de-risk Kaggle environment.         |
|             |                   | **DELIVERABLE:** `circuits.db` v0.1                       | Create foundational library of skills for Track B.      |
|             |                   | **SUBMIT:** `george` v1                                   | Benchmark symbolic engine performance.                  |
| **Phase 2** | End of Month 4    | **SUBMIT:** `JARC-Reactor` v2                             | Test performance of a symbolically-enhanced model.      |
|             |                   | **SUBMIT:** `george` v2                                   | Test performance of a symbolic-neural model.            |
| **Phase 3** | End of Month 5    | **SUBMIT:** `ARCHIMEDES` Hybrid Solver v1.0               | Final competition entry combining the best of both tracks. |

---

#### **2.2. Team Structure**

*   **Track A: The "Production" Team (Lead: You)**
    *   **Mission:** Build the most powerful, practical, and reliable solver possible. Ensure a strong entry on the leaderboard at all times. This track is the **"floor-raiser."**
    *   **Repositories:** `JARC-Reactor`, `simplest_arc_agi`

*   **Track B: The "R&D" Team (Lead: `george` Creator)**
    *   **Mission:** Fully implement the visionary cognitive architecture of `george`. Create a system that can solve problems via novel reasoning. This track is the **"ceiling-raiser."**
    *   **Repository:** `george`

---

#### **2.3. Phase 1: Foundation & Baseline (Months 1-2)**

*   **Track A Goals:**
    *   **Weeks 1-3:** Execute the "Pragmatic Entry" plan for `JARC-Reactor`.
        1.  Run a focused Optuna HPO study to find the best hyperparameters for the `TransformerModel`.
        2.  Train the "Golden v1" model on the full training dataset.
        3.  Thoroughly validate the Kaggle submission pipeline with the trained model.
        4.  **SUBMIT `JARC-Reactor` v1.**
    *   **Weeks 4-8:** Begin building the `simplest_arc_agi` "System 2 Toolbox."
        1.  Generalize the `simplest_arc_agi` data generation framework to be task-agnostic.
        2.  Begin batch-training a library of 10-15 core ARC primitives (e.g., rotation, object finding), populating the `circuits.db`. This library is a critical shared asset, forming the **Foundational Library of Verifiable Skills**.

*   **Track B Goals:**
    *   **Weeks 1-8:** Implement the core `george` reasoning engine.
        1.  Replace the placeholder CNN in `NeuralProgramLattice` with a genuine program composition engine.
        2.  Evolve the `AdaptingNeuralTree` from a brute-force checker into a more intelligent search or synthesis algorithm.
        3.  **SUBMIT `george` v1.**

*   **Phase 1 Milestone:** Two independent models submitted. We will have a data-driven understanding of the performance gap between a powerful end-to-end model (`JARC-Reactor`) and a nascent symbolic-composition engine (`george`).

---

#### **2.4. Phase 2: Integration & Hybridization (Months 3-4)**

*   **Track A Goals:**
    *   **Weeks 9-16:** Create the **"Symbolic-Enhanced `JARC-Reactor`."**
        1.  Integrate the library of symbolic primitives from `george`'s `config.py` as a feature extractor.
        2.  Enrich `JARC-Reactor`'s `ContextEncoder` to accept this symbolic feature vector as an additional input.
        3.  Retrain the model with this new, more powerful context.
        4.  **SUBMIT `JARC-Reactor` v2.**

*   **Track B Goals:**
    *   **Weeks 9-16:** Upgrade `george` with learned neural skills.
        1.  Modify the `george` engine to query and execute the pre-trained neural circuits from the `circuits.db` being built by Track A.
        2.  This transforms `george` from a purely symbolic reasoner into a **symbolic-neural reasoner.**
        3.  Refine the `george` planner to work effectively with this new library of neural tools.
        4.  **SUBMIT `george` v2.**

*   **Phase 2 Milestone:** We will have two highly advanced, competing architectures. We can perform a detailed analysis of which tasks each architecture solves better, informing the final synthesis.

---

#### **2.5. Phase 3: The Final Synthesis (Month 5)**

*   **Unified Team Goals (You & `george` Creator):**
    *   **Task 5.1: Architect and Validate the Hybrid Selection Logic.** Create and rigorously test a `selection_logic.py` module. This includes defining the confidence score formula, setting the escalation threshold, and establishing a final fallback strategy.
    *   **Task 5.2: Build and Refine the ARCHIMEDES Hybrid Solver.** Integrate the validated selection logic into a master `solver.py` script that orchestrates the two systems.
        1.  **System 1:** For each ARC task, the solver will first run the **Symbolic-Enhanced `JARC-Reactor` v2 model (from Track A).**
        2.  **Confidence Check:** The model will output its prediction and a confidence score (e.g., derived from the output logits' entropy).
        3.  **System 2 Fallback:** If the confidence score is low, the solver discards the answer and instead calls the full **`george` v2 symbolic-neural engine (from Track B).**
    *   **Final Submission:** The final entry is not a single model, but a **two-stage cognitive system** that dynamically chooses between a powerful intuitive engine and a precise logical reasoner.

### III. Risk Mitigation

The two-track approach is our primary risk mitigation strategy, designed to balance innovation with pragmatism.

*   **Technical Risk Mitigation:** If Track B's ambitious R&D on the `george` engine is delayed or proves less effective than hoped, the project is not at risk of failure. Track A's robust, enhanced `JARC-Reactor` v2 will still provide a very strong, competitive final entry.
*   **Performance Ceiling Mitigation:** If Track A's end-to-end model hits a performance plateau on tasks requiring complex, multi-step logic, Track B's novel compositional architecture provides an alternative path to solving those difficult problems, raising the overall performance ceiling of our final hybrid system.

| **Risk ID** | **Risk Description**                                                                                                                                     | **Likelihood** | **Impact** | **Mitigation Strategy**                                                                                                                                                                                                                         |
| :---------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------- | :--------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **R-01**    | **Interface Drift:** The interfaces between Track A's `circuits.db` and Track B's `george` engine diverge, making final integration in Month 5 difficult. | Medium         | High       | **Primary:** Define a preliminary, versioned schema for the `circuits.db` and the circuit execution API early in Phase 1. **Secondary:** Schedule bi-weekly, 30-minute integration syncs between the two track leads starting in Month 3. |
| **R-02**    | **Kaggle Environment Issues:** Unforeseen issues with the offline Kaggle notebook environment (e.g., dependency conflicts, memory limits) hinder deployment. | Medium         | High       | **Primary:** The early submission of `JARC-Reactor` v1 in Phase 1 is designed specifically to de-risk this. Any environment issues will be discovered and solved early, not during the final deadline crunch.                                      |
| **R-03**    | **Developer Burnout:** The intensity of the parallel sprint leads to fatigue and reduced productivity for the two-person team.                               | Medium         | High       | **Primary:** Strict adherence to the Cultivation system's own wellness protocols (`My_Optimized_Performance_Schedule_v1.0.md`). **Secondary:** Use the `fatigue-watch.yml` system and conduct bi-weekly check-ins to monitor for signs of overtraining/burnout. |

### IV. Post-Competition Harvesting

Upon conclusion of the competition, the focus will shift to the "harvesting" phase as detailed in the strategic initiative document. This includes:

*   Finalizing and submitting the Paper Award submission (`DW_ARC_PAPER_001`).
*   Integrating the final `CircuitDatabase` and `ARCHIMEDES` solver logic into the Cultivation KCV layer (`DW_ARC_HARVEST_KCV_001`).
*   Developing the ETL to feed final ARC performance metrics into the Π-Engine's Aptitude (A) domain (`DW_ARC_HARVEST_APTITUDE_001`).