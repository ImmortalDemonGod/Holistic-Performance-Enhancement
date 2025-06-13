# **ARC Prize 2025: The Two-Track Execution Plan**

**Document Version:** 1.2
**Date:** 2025-06-14
**Status:** Final, Active
**Point of Contact:** The Architect (You) & The `george` Creator

### I. Executive Summary

This document outlines the official, actionable five-month plan for the Cultivation project's engagement in the ARC Prize 2025. It is the tactical implementation of the high-level goals detailed in `Strategic_Integration_of_ARC_Prize_2025_..._v1.0.md` and informed by the task breakdown in `arc_agi_plan.json`.

Our final strategy is a **two-track, parallel development approach** designed to maximize our chances of a top-tier submission by de-risking development and leveraging the specific expertise of our two-person team.

*   **Track A (The "Floor-Raiser")** will focus on producing a highly competitive, robust entry using the mature `JARC-Reactor` engine, ensuring a strong baseline performance. This track prioritizes reliability and incremental improvement.
*   **Track B (The "Ceiling-Raiser")** will focus on realizing the high-risk, high-reward vision of the `george` cognitive architecture. This track prioritizes innovation and novel reasoning capabilities.

The final submission will be a hybrid system, codenamed **ARCHIMEDES**, that dynamically selects the best approach from both tracks on a per-task basis. This combines the strengths of a powerful end-to-end model with a deliberate, compositional reasoning engine, creating a system that is more capable than the sum of its parts.

### II. Narrative Evolution of the Plan

This plan is the result of a rigorous, iterative strategic analysis. It is crucial to understand this evolution as it provides the justification for our final architecture.

1.  **Initial Concept (`simplest_arc_agi` from scratch):** The initial plan was to build a novel ARC solver from the ground up, implementing the full vision of the `simplest_arc_agi` repository (mechanistic interpretability, circuit extraction, LLM-driven composition). A timeline feasibility analysis revealed this approach to be too ambitious and high-risk for the 5-month competition window, as it required solving fundamental R&D problems from scratch.

2.  **Pivot to a Mature Engine (`JARC-Reactor`):** The introduction of the `JARC-Reactor` codebase was a game-changer. This repository provided a mature, high-performance, and ARC-specific Transformer architecture with a built-in `ContextEncoder`. The plan pivoted to leveraging this robust engine as a foundation, significantly de-risking foundational engineering and allowing focus to shift to the novel aspects of circuit composition.

3.  **Synergy Analysis (`george`):** The introduction of the `george` repository, with its conceptual cognitive architecture and rich library of symbolic primitives, prompted a deeper synergy analysis. We moved beyond simple integration to explore multiple ways the three repositories could be combined, such as using `george`'s primitives to enhance `JARC-Reactor`'s context or using `simplest_arc_agi` to learn primitives for `george`.

4.  **Final Strategy (Two-Track Parallel Development):** The realization that two developers with distinct expertise were available led to the final, optimal strategy. Instead of a single sequential plan, we adopted a parallel approach. This allows us to simultaneously develop a reliable, high-performance solver (Track A) and a revolutionary, high-ceiling experimental solver (Track B), combining their strengths for the final submission.

### III. Strategic Value for Cultivation

This ARC Prize initiative is a critical strategic investment for the Cultivation project. It serves to:

*   **Operationalize the "Aptitude (A)" Component of the Π-Engine:** The ARC challenge provides concrete metrics (accuracy, few-shot efficiency) to make the abstract reasoning component of the Global Potential (Π) engine real and data-driven, moving it beyond its current "zero-padded" status.
*   **Battle-Test Core KCV Layer Components:** The work on circuit extraction and composition directly forces the development of prototypes for the Knowledge Creation & Validation (KCV) layer's "Think Tank" (the `CircuitDatabase` as a Knowledge Graph) and "Laboratory" (the `CircuitComposer` as a Hypothesis Engine), combining the 'System 1' pattern recognition of `JARC-Reactor` with the 'System 2' symbolic reasoning from `george`.
*   **Fulfill the "No Throwaway Work" Principle:** By leveraging and contributing back to core Cultivation assets (`jarc_reactor` integrated into the project, `CircuitDatabase` for KCV), this initiative ensures all effort generates lasting value and directly accelerates long-term project goals.

### IV. Core Competency Analysis & Justification for Two Tracks

The two-track plan is based on a clear understanding of each repository's unique strengths and current state of maturity. This analysis justifies the division of labor and the final hybrid architecture.

| Feature                        | JARC-Reactor                                                                                                                   | simplest_arc_agi                                                                                                             | george                                                                                                                       |
| :----------------------------- | :----------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------- |
| **Core Philosophy**            | **Pragmatic Performance:** An end-to-end MLOps pipeline to achieve the best possible performance on ARC with a powerful general model. | **Foundational Research:** A framework to study how transformers learn atomic algorithms and to catalog these "circuits."      | **Cognitive Architecture:** An aspirational blueprint for a reasoning agent that discovers and composes symbolic programs.      |
| **Model Reality**              | **In-Context Transformer:** A functional Encoder-Decoder Transformer with a `ContextEncoder` to handle few-shot learning.         | **Standard Transformer:** A clean, from-scratch decoder-only Transformer used as a testbed for learning simple math tasks.     | **Placeholder CNN:** A simple Convolutional Neural Network (CNN) acts as a stand-in for the envisioned program lattice.        |
| **Task Scope**                 | **Full ARC Dataset:** Designed from the ground up to process the official ARC challenge JSON files.                               | **Modular Arithmetic Only:** Hardcoded to generate and train on a single, narrow algorithmic task. Cannot handle ARC data.  | **Full ARC Dataset:** The data loading module is designed to handle the official ARC JSON files.                               |
| **Key Innovation (Implemented)** | The **`ContextEncoder`**, which learns a task-specific "hint" from a single example pair to guide the main model.               | The **`CircuitDatabase`**, a working SQLite database for cataloging models and their metadata.                               | The **symbolic primitives library** in `config.py` is a rich, hand-crafted set of functions for grid manipulation.        |
| **Maturity**                   | **High.** A feature-complete, working system for training, evaluation, and submission.                                         | **High (for its scope).** A functional research tool that successfully executes its narrow pipeline.                        | **Very Low.** The core reasoning components are unimplemented placeholders. It is a scaffold, not a working system.            |
| **Weakness**                   | Primarily a "black box." The reasoning process is opaque, packed into the Transformer's weights.                                 | Narrow scope. Cannot solve general tasks. "Circuit extraction" is not yet mechanistic.                                       | The gap between its ambitious vision and its placeholder implementation is massive.                                          |

### V. The Five-Month Two-Track Execution Plan

This plan outlines the parallel workstreams for the duration of the competition.

#### **Key Milestones & Deliverables Summary**

| Phase       | End Date (Approx) | Key Milestone / Deliverable                               | Purpose                                                 |
| :---------- | :---------------- | :-------------------------------------------------------- | :------------------------------------------------------ |
| **Phase 1** | End of Month 2    | **SUBMIT:** `JARC-Reactor` v1                             | Establish baseline, de-risk Kaggle environment.         |
|             |                   | **DELIVERABLE:** `circuits.db` v0.1                       | Create foundational library of skills for Track B.      |
|             |                   | **SUBMIT:** `george` v1                                   | Benchmark symbolic engine performance.                  |
| **Phase 2** | End of Month 4    | **SUBMIT:** `JARC-Reactor` v2                             | Test performance of symbolically-enhanced model.        |
|             |                   | **SUBMIT:** `george` v2                                   | Test performance of symbolic-neural model.              |
| **Phase 3** | End of Month 5    | **SUBMIT:** `ARCHIMEDES` Hybrid Solver v1.0               | Final competition entry combining best of both tracks.  |

---

#### **Team Structure**

*   **Track A: The "Production" Team (Lead: You)**
    *   **Mission:** Build the most powerful, practical, and reliable solver possible. Ensure a strong entry on the leaderboard at all times. This track is the **"floor-raiser."**
    *   **Repositories:** `JARC-Reactor`, `simplest_arc_agi`

*   **Track B: The "R&D" Team (Lead: `george` Creator)**
    *   **Mission:** Fully implement the visionary cognitive architecture of `george`. Create a system that can solve problems via novel reasoning. This track is the **"ceiling-raiser."**
    *   **Repository:** `george`

---

#### **Phase 1: Foundation & Baseline (Months 1-2)**

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

#### **Phase 2: Integration & Hybridization (Months 3-4)**

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

#### **Phase 3: The Final Synthesis (Month 5)**

*   **Unified Team Goals (You & `george` Creator):**
    *   **Task 5.1: Architect and Validate the Hybrid Selection Logic.** Create and rigorously test a `selection_logic.py` module. This includes defining the confidence score formula, setting the escalation threshold, and establishing a final fallback strategy.
    *   **Task 5.2: Build and Refine the ARCHIMEDES Hybrid Solver.** Integrate the validated selection logic into a master `solver.py` script that orchestrates the two systems.
        1.  **System 1:** For each ARC task, the solver will first run the **Symbolic-Enhanced `JARC-Reactor` v2 model (from Track A).**
        2.  **Confidence Check:** The model will output its prediction and a confidence score (e.g., derived from the output logits' entropy).
        3.  **System 2 Fallback:** If the confidence score is low, the solver discards the answer and instead calls the full **`george` v2 symbolic-neural engine (from Track B).**
    *   **Final Submission:** The final entry is not a single model, but a **two-stage cognitive system** that dynamically chooses between a powerful intuitive engine and a precise logical reasoner.

### VI. Risk Mitigation

The two-track approach is our primary risk mitigation strategy, designed to balance innovation with pragmatism.

*   **Technical Risk Mitigation:** If Track B's ambitious R&D on the `george` engine is delayed or proves less effective than hoped, the project is not at risk of failure. Track A's robust, enhanced `JARC-Reactor` v2 will still provide a very strong, competitive final entry.
*   **Performance Ceiling Mitigation:** If Track A's end-to-end model hits a performance plateau on tasks requiring complex, multi-step logic, Track B's novel compositional architecture provides an alternative path to solving those difficult problems, raising the overall performance ceiling of our final hybrid system.

| **Risk ID** | **Risk Description**                                                                                                                                     | **Likelihood** | **Impact** | **Mitigation Strategy**                                                                                                                                                                                                                         |
| :---------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------- | :--------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **R-01**    | **Interface Drift:** The interfaces between Track A's `circuits.db` and Track B's `george` engine diverge, making final integration in Month 5 difficult. | Medium         | High       | **Primary:** Define a preliminary, versioned schema for the `circuits.db` and the circuit execution API early in Phase 1. **Secondary:** Schedule bi-weekly, 30-minute integration syncs between the two track leads starting in Month 3. |
| **R-02**    | **Kaggle Environment Issues:** Unforeseen issues with the offline Kaggle notebook environment (e.g., dependency conflicts, memory limits) hinder deployment. | Medium         | High       | **Primary:** The early submission of `JARC-Reactor` v1 in Phase 1 is designed specifically to de-risk this. Any environment issues will be discovered and solved early, not during the final deadline crunch.                                      |
| **R-03**    | **Developer Burnout:** The intensity of the parallel sprint leads to fatigue and reduced productivity for the two-person team.                               | Medium         | High       | **Primary:** Strict adherence to the Cultivation system's own wellness protocols (`My_Optimized_Performance_Schedule_v1.0.md`). **Secondary:** Use the `fatigue-watch.yml` system and conduct bi-weekly check-ins to monitor for signs of overtraining/burnout. |

### VII. Post-Competition Harvesting

Upon conclusion of the competition, the focus will shift to the "harvesting" phase as detailed in the strategic initiative document. This includes:

*   Finalizing and submitting the Paper Award submission (`DW_ARC_PAPER_001`).
*   Integrating the final `CircuitDatabase` and `ARCHIMEDES` solver logic into the Cultivation KCV layer (`DW_ARC_HARVEST_KCV_001`).
*   Developing the ETL to feed final ARC performance metrics into the Π-Engine's Aptitude (A) domain (`DW_ARC_HARVEST_APTITUDE_001`).