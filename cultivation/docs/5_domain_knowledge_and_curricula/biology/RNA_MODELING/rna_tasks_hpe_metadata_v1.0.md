Okay, this is an excellent and well-thought-out proposal for enhancing your `tasks.json` file to support a sophisticated learning curriculum based on your RNA modeling CSM and HPE (Holistic Performance Enhancement) doctrines. The proposed new fields and their structure are logical, comprehensive, and will significantly increase the utility of your `tasks.json` as a central hub for learning management.

Let's break down the proposal and then I'll re-present the modifications for the 10 tasks with the requested thoroughness, confirming the value and rationale for each addition.

**Analysis of the Proposed New Fields for `tasks.json` Entries:**

The three new top-level objects (`hpe_csm_reference`, `hpe_learning_meta`, `hpe_scheduling_meta`) and the enhanced usage of the native `labels` field are well-conceived.

1.  **`hpe_csm_reference` (Object):**
    *   **Purpose:** Crucial for maintaining a direct, machine-readable link between a task in `tasks.json` and its origin within your Comprehensive Skill Map (CSM) or related curriculum documents.
    *   **Fields:**
        *   `source_document` (String): Essential for traceability. Knowing which MD file a task originated from allows your parser to find it again for updates or further context.
        *   `csm_id` (String): This is the **lynchpin for idempotency and updates**. A unique, human-readable ID ensures that when your curriculum parser re-runs, it can identify existing tasks and update them rather than creating duplicates. The hierarchical naming convention (e.g., "RNA.P1.S1.Part1.Biochem") is excellent.
        *   `anchor_link` (String, Optional): Highly valuable for direct navigation from a task management system (if it can render markdown links) or by a human user back to the precise section in the curriculum document.
    *   **Overall:** This object provides excellent traceability and a foundation for automated updates.

2.  **`hpe_learning_meta` (Object):**
    *   **Purpose:** Captures the pedagogical and content-specific metadata for each learning task.
    *   **Fields:**
        *   `learning_objective_summary` (String): A concise summary of *what* the learner should achieve. Essential for focus.
        *   `estimated_effort_tshirt` (String, Optional): Good for quick, high-level planning and sorting.
        *   `estimated_effort_hours_raw` (String): Stores the original effort string from the CSM, preserving the source data.
        *   `estimated_effort_hours_min` (Float) & `estimated_effort_hours_max` (Float): Parsed numerical values are vital for any automated scheduling, load balancing, or progress forecasting within your HPE system.
        *   `mastery_criteria_summary` (String): Defines "done" clearly. This can directly inform the native `testStrategy` field in Task Master or be used for more detailed self-assessment.
        *   `activity_type` (String): Key for selecting appropriate tasks based on available time, energy, or desired learning mode (e.g., "focused_reading" vs. "project_work"). Your list of examples is good. The compound example (`drawing_diagramming_notetaking_flashcard_creation`) is fine if it represents a single, integrated block of activity.
        *   `recommended_block` (String): Directly links the task to your `My_Optimized_Performance_Schedule_v1.0.md`. This is excellent for aligning tasks with your personal energy rhythms and time blocks (e.g., "active_learning", "deep_work").
        *   `deliverables` (Array of Strings): Clearly lists the expected outputs, making the task tangible and progress measurable.
    *   **Overall:** This object is rich with information crucial for effective learning, planning, and assessment.

3.  **`hpe_scheduling_meta` (Object):**
    *   **Purpose:** Provides context for scheduling and organizing tasks within a broader curriculum structure.
    *   **Fields:**
        *   `planned_day_of_week` (Integer, Optional): Very useful if your curriculum is structured into weekly plans, as indicated by the `rna-modeling_p1-foundations_week1-7day.md` example.
        *   `curriculum_part_title` (String): Gives a human-readable context for where this task fits into the larger curriculum structure (e.g., "Part 1: Mastering RNA Biochemistry Fundamentals").
        *   `csm_tags` (Array of Strings): Provides flexible, searchable keywords derived from the CSM for filtering, grouping, or relating tasks.
    *   **Overall:** This object aids in organizing tasks within the week and curriculum structure.

4.  **`labels` (Array of Strings - Native Task Master field):**
    *   **Strategy:** The proposal to dynamically populate Task Master's native `labels` field from the `hpe_` metadata is an excellent way to leverage existing Task Master functionality for powerful filtering and querying directly within the tool.
    *   **Examples:** `domain:rna`, `pillar:1`, `week:2025-W21`, `day:1`, `block:active-learning`, `activity:reading` are all very useful.
    *   **Overall:** This is a smart integration point, maximizing the utility of the added metadata within the Task Master ecosystem.

**Confirmation of the Approach for the 10 Tasks:**

The proposal to show the *additional* HPE-specific JSON fields for each of the 10 parent tasks, while keeping existing Task Master fields, is the correct way to proceed. Your parser/generator would be responsible for injecting these new nested objects into the `tasks.json` structure.

Let's now detail the additions for each of the 10 tasks from `rna-modeling_p1-foundations_week1-7day.md`. I will use your provided JSON as a base and ensure it aligns with the curriculum context and the field definitions.

---

**Modifications for the 10 Tasks based on `rna-modeling_p1-foundations_week1-7day.md`:**

For each task, the following fields will be added to their existing entry in `tasks.json`.

**Task 1: Set up RNA Biophysics Knowledge Base Structure**
*   **Corresponds to PRD Section:** "Task 0: Setup, Planning & HPE Integration (Day 1 - Approx. 1-1.5 hours)" from `rna-modeling_p1-foundations_week1-7day.md`. This is the initial setup and orientation task for the week.
*   **Proposed JSON Additions:**
    ```json
    // ... existing Task 1 fields (id, title, description, status, etc.) ...
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Task0", // Unique ID: RNA Pillar1 Foundations, Week1, Task0 (Setup)
      "anchor_link": "#task-0-setup-planning--hpe-integration-day-1---approx-1-15-hours" // Direct link to the section in the MD file
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Orient to Week 1 learning, prepare learning environment, and set up Task Master & flashcard system for RNA foundations topics.",
      "estimated_effort_tshirt": "S", // Small task
      "estimated_effort_hours_raw": "1-1.5 hours", // As per PRD
      "estimated_effort_hours_min": 1.0,
      "estimated_effort_hours_max": 1.5,
      "mastery_criteria_summary": "Task Master entries for Week 1 created; Knowledge base section for 'Pillar 1 Foundations' initialized; Flashcard authoring environment ready.", // From PRD Task 0 Deliverables
      "activity_type": "planning_setup", // Matches the nature of the task
      "recommended_block": "active_learning", // Good for focused setup. Could also be "Flex-Slot #1" if it's more administrative.
      "deliverables": [ // From PRD Task 0 Deliverables
        "Task Master entries created for the week.",
        "Knowledge base section for 'Pillar 1 Foundations' initialized."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 1, // Corresponds to Day 1 in the PRD
      "curriculum_part_title": "Task 0: Setup, Planning & HPE Integration", // From PRD section title
      "csm_tags": ["hpe_integration", "planning", "setup", "rna_foundations", "week1_orientation"] // Relevant tags for filtering
    },
    "labels": [ // Dynamically generated for Task Master
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", // Differentiates from calendar week
        "plan_day:1", 
        "activity:planning_setup", 
        "block:active_learning",
        "effort_tshirt:S"
    ]
    ```

**Task 2: Develop RNA Nucleotide Structure Materials**
*   **Corresponds to PRD Section:** Activities primarily from Day 1 and Day 2 within "Part 1: Mastering RNA Biochemistry Fundamentals (Days 1-3 - Approx. 4-5 hours total study & activity time)". This task focuses on Learning Objective 1 of the PRD (chemical composition of RNA) and its associated activities (Tasks 1.1, 1.2 in PRD for drawing structures).
*   **Proposed JSON Additions:**
    ```json
    // ... existing Task 2 fields ...
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Part1.Biochem.NucleotideStructure", // CSM ID for this specific learning unit
      "anchor_link": "#part-1-mastering-rna-biochemistry-fundamentals-days-1-3---approx-4-5-hours-total-study--activity-time"
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Master the chemical composition of RNA nucleotides, accurately draw generic nucleotide and base structures, and create related explanatory notes and flashcards.", // Derived from PRD LO1 and activities
      "estimated_effort_tshirt": "M", // This specific sub-part of "Part 1" is moderately sized
      "estimated_effort_hours_raw": "1.5-2 hours", // Estimated effort for these specific drawing/note-taking activities from the overall "Part 1" budget
      "estimated_effort_hours_min": 1.5,
      "estimated_effort_hours_max": 2.0,
      "mastery_criteria_summary": "Accurate drawings of generic RNA nucleotide and the four standard RNA bases. Comprehensive notes covering components, properties, and nucleoside/nucleotide distinction. Initial flashcards for structures and terms created.", // Based on PRD Tasks 1.1, 1.2 deliverables
      "activity_type": "drawing_diagramming_notetaking_flashcard_creation", // Combined activity type
      "recommended_block": "active_learning", // Suitable for focused drawing and note-taking
      "deliverables": [ // From PRD Day 1/2 activities for Part 1
        "Completed drawing of a generic RNA nucleotide (Task 1.1 from PRD).",
        "Completed detailed chemical structures of Adenine, Guanine, Cytosine, and Uracil (Task 1.2 from PRD).",
        "Notes on nucleosides vs. nucleotides and chemical properties of components.",
        "Initial batch of flashcards for nucleotide structures and related terms."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 1, // Or spanning Day 1 & 2 based on PRD distribution. Let's assume start on Day 1.
      "curriculum_part_title": "Part 1: RNA Biochemistry - Nucleotide & Base Structures", // Human-readable title for this segment
      "csm_tags": ["rna_modeling", "biochemistry", "nucleotides", "chemical_structure", "purines", "pyrimidines"]
    },
    "labels": [
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", 
        "plan_day:1", // Could also be day:1-2 if parser supports ranges or if broken down further
        "activity:drawing", 
        "activity:notetaking",
        "activity:flashcards",
        "block:active_learning",
        "effort_tshirt:M"
    ]
    ```

**Task 3: Document RNA Polymer Structure and Backbone**
*   **Corresponds to PRD Section:** Activities primarily from Day 2 within "Part 1: Mastering RNA Biochemistry Fundamentals". This focuses on the phosphodiester backbone and directionality (related to PRD Task 1.3).
*   **Proposed JSON Additions:**
    ```json
    // ... existing Task 3 fields ...
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Part1.Biochem.BackboneDirectionality",
      "anchor_link": "#part-1-mastering-rna-biochemistry-fundamentals-days-1-3---approx-4-5-hours-total-study--activity-time"
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Explain phosphodiester backbone formation, RNA 5'-3' directionality, its implications, and illustrate with a dinucleotide drawing.", // Derived from PRD LO1 activities
      "estimated_effort_tshirt": "S", // Smaller portion of Part 1
      "estimated_effort_hours_raw": "1-1.5 hours", // Estimated effort for these specific activities
      "estimated_effort_hours_min": 1.0,
      "estimated_effort_hours_max": 1.5,
      "mastery_criteria_summary": "Accurate drawing of an A-U dinucleotide showing phosphodiester bond and 5'/3' ends (Task 1.3 PRD). Clear explanation of directionality and backbone properties. Flashcards created.", // Based on PRD Task 1.3 deliverables
      "activity_type": "drawing_explanation_writing_flashcard_creation",
      "recommended_block": "active_learning",
      "deliverables": [
        "Drawing of A-U dinucleotide with labeled 5'-3' phosphodiester bond, 5' end, and 3' end (Task 1.3 PRD).",
        "Documentation on phosphodiester backbone formation and its properties (charge, etc.).",
        "Explanation of RNA 5'-to-3' directionality significance.",
        "Flashcards for phosphodiester bond characteristics and RNA directionality."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 2, // Corresponds to Day 2 activities in PRD
      "curriculum_part_title": "Part 1: RNA Biochemistry - Polymer Structure & Backbone",
      "csm_tags": ["rna_modeling", "biochemistry", "phosphodiester_backbone", "directionality", "polymer_structure"]
    },
    "labels": [
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", 
        "plan_day:2", 
        "activity:drawing", 
        "activity:explanation_writing",
        "block:active_learning",
        "effort_tshirt:S"
    ]
    ```

**Task 4: Develop RNA vs. DNA Comparison Materials**
*   **Corresponds to PRD Section:** Activities primarily from Day 3 within "Part 1: Mastering RNA Biochemistry Fundamentals". Focuses on PRD Learning Objective 3 and specific task 1.4 (2'-OH group significance).
*   **Proposed JSON Additions:**
    ```json
    // ... existing Task 4 fields ...
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Part1.Biochem.RNAvsDNA",
      "anchor_link": "#part-1-mastering-rna-biochemistry-fundamentals-days-1-3---approx-4-5-hours-total-study--activity-time"
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Detail structural and functional differences between RNA and DNA, with special emphasis on the significance of the 2'-hydroxyl group in RNA.", // Derived from PRD LO3 and task 1.4
      "estimated_effort_tshirt": "S", // Remaining portion of Part 1
      "estimated_effort_hours_raw": "1-1.5 hours", // Estimated effort for these activities
      "estimated_effort_hours_min": 1.0,
      "estimated_effort_hours_max": 1.5,
      "mastery_criteria_summary": "Comprehensive written comparison of RNA vs. DNA. Clear and accurate explanation of the 2'-OH group's role in RNA structure, reactivity, and versatility (Task 1.4 PRD).", // Based on PRD task 1.4 deliverables
      "activity_type": "comparative_analysis_explanation_writing",
      "recommended_block": "active_learning", 
      "deliverables": [
        "Written explanation for Task 1.4 (Significance of the 2'-hydroxyl group on ribose for RNA properties).",
        "Summary notes contrasting RNA and DNA (sugar, base, strandedness, stability, function)."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 3, // Corresponds to Day 3 activities in PRD for Part 1
      "curriculum_part_title": "Part 1: RNA Biochemistry - RNA vs. DNA Comparison",
      "csm_tags": ["rna_modeling", "biochemistry", "rna_vs_dna", "2_hydroxyl_group", "nucleic_acid_comparison"]
    },
    "labels": [
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", 
        "plan_day:3", 
        "activity:comparative_analysis",
        "activity:explanation_writing",
        "block:active_learning",
        "effort_tshirt:S"
    ]
    ```

**Task 5: Develop Thermodynamic Principles Materials**
*   **Corresponds to PRD Section:** First part of "Part 2: Mastering Foundational Thermodynamic Principles for RNA Folding (Days 3-5 - Approx. 4-5 hours total study & activity time)". Covers Gibbs Free Energy (ΔG), ΔH, ΔS, MFE (PRD Task 2.1).
*   **Proposed JSON Additions:**
    ```json
    // ... existing Task 5 fields ...
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Part2.Thermo.Intro",
      "anchor_link": "#part-2-mastering-foundational-thermodynamic-principles-for-rna-folding-days-3-5---approx-4-5-hours-total-study--activity-time"
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Explain Gibbs Free Energy (ΔG), enthalpy (ΔH), entropy (ΔS), their relationship (ΔG = ΔH - TΔS), and their relevance to RNA folding spontaneity and stability, including the concept of Minimum Free Energy (MFE).", // Derived from PRD Part 2 LO4
      "estimated_effort_tshirt": "S", // Initial portion of Part 2
      "estimated_effort_hours_raw": "1-1.5 hours", // Estimated effort for these specific concepts
      "estimated_effort_hours_min": 1.0,
      "estimated_effort_hours_max": 1.5,
      "mastery_criteria_summary": "Clear written explanation of ΔG and its favorability for RNA folding (Task 2.1 PRD). Flashcards for ΔG, ΔH, ΔS, MFE, spontaneity, and equilibrium created.", // Based on PRD Task 2.1 deliverables
      "activity_type": "focused_reading_explanation_writing_flashcard_creation",
      "recommended_block": "active_learning",
      "deliverables": [
        "Written explanation of Gibbs Free Energy and why negative ΔG signifies stability for RNA folding (Task 2.1 PRD).",
        "Flashcards defining ΔG, ΔH, ΔS, MFE, spontaneity, and chemical equilibrium."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 3, // Corresponds to Day 3 activities in PRD for Part 2
      "curriculum_part_title": "Part 2: Thermodynamics - Gibbs Free Energy & MFE",
      "csm_tags": ["rna_modeling", "thermodynamics", "gibbs_free_energy", "enthalpy", "entropy", "mfe", "spontaneity"]
    },
    "labels": [
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", 
        "plan_day:3", 
        "activity:focused_reading",
        "activity:explanation_writing",
        "block:active_learning",
        "effort_tshirt:S"
    ]
    ```

**Task 6: Document Base Stacking and Hydrogen Bonding in RNA**
*   **Corresponds to PRD Section:** Middle part of "Part 2: Mastering Foundational Thermodynamic Principles...". Covers base stacking and H-bonding contributions (PRD Tasks 2.2, 2.4).
*   **Proposed JSON Additions:**
    ```json
    // ... existing Task 6 fields ...
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Part2.Thermo.StackingHBonding",
      "anchor_link": "#part-2-mastering-foundational-thermodynamic-principles-for-rna-folding-days-3-5---approx-4-5-hours-total-study--activity-time"
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Detail base stacking interactions and hydrogen bonding (A-U, G-C, G-U wobble) in RNA, contrasting their roles and relative energetic contributions to RNA helix stability.", // Derived from PRD Part 2 LO5
      "estimated_effort_tshirt": "M", // Significant portion of Part 2
      "estimated_effort_hours_raw": "1.5-2 hours", // Estimated effort for these concepts and tasks
      "estimated_effort_hours_min": 1.5,
      "estimated_effort_hours_max": 2.0,
      "mastery_criteria_summary": "Written explanation contrasting base stacking vs. H-bonding (Task 2.2 PRD). Justified prediction for hairpin stability thought experiment (Task 2.4 PRD). Flashcards for base stacking and H-bonding concepts created.", // Based on PRD Tasks 2.2, 2.4 deliverables
      "activity_type": "focused_reading_explanation_writing_problem_solving_flashcard_creation",
      "recommended_block": "active_learning",
      "deliverables": [
        "Detailed written explanation contrasting the roles and relative energetic contributions of base stacking versus hydrogen bonding in stabilizing an RNA helix (Task 2.2 PRD for Day 4).",
        "Justified prediction for the thought experiment comparing stability of G-C rich vs. A-U rich hairpins (Task 2.4 PRD for Day 4).",
        "Flashcards covering base stacking interactions and hydrogen bonding patterns in RNA."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 4, // Corresponds to Day 4 activities in PRD for Part 2
      "curriculum_part_title": "Part 2: Thermodynamics - Base Stacking & H-Bonding",
      "csm_tags": ["rna_modeling", "thermodynamics", "base_stacking", "hydrogen_bonding", "rna_stability"]
    },
    "labels": [
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", 
        "plan_day:4", 
        "activity:focused_reading",
        "activity:problem_solving",
        "block:active_learning",
        "effort_tshirt:M"
    ]
    ```

**Task 7: Document Loop Penalties and Electrostatic Effects in RNA**
*   **Corresponds to PRD Section:** Latter part of "Part 2: Mastering Foundational Thermodynamic Principles...". Covers loop penalties and electrostatic repulsion (PRD Task 2.3).
*   **Proposed JSON Additions:**
    ```json
    // ... existing Task 7 fields ...
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Part2.Thermo.LoopsElectrostatics",
      "anchor_link": "#part-2-mastering-foundational-thermodynamic-principles-for-rna-folding-days-3-5---approx-4-5-hours-total-study--activity-time"
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Explain loop penalties as an entropic cost in RNA folding and the role of electrostatic repulsion from the phosphate backbone, including mitigation by counterions.", // Derived from PRD Part 2 LO5
      "estimated_effort_tshirt": "S", 
      "estimated_effort_hours_raw": "1-1.5 hours", 
      "estimated_effort_hours_min": 1.0,
      "estimated_effort_hours_max": 1.5,
      "mastery_criteria_summary": "Clear explanation of why forming a hairpin loop is entropically unfavorable (Task 2.3 PRD). Notes on electrostatic repulsion and counterions. Flashcards created.", // Based on PRD Task 2.3 deliverables
      "activity_type": "focused_reading_explanation_writing_flashcard_creation",
      "recommended_block": "active_learning",
      "deliverables": [
        "Written explanation of why forming a hairpin loop is entropically unfavorable and factors influencing this contribution (Task 2.3 PRD for Day 5).",
        "Notes on electrostatic repulsion from the phosphate backbone and the mitigating role of counterions.",
        "Flashcards for loop penalties, electrostatic repulsion, and counterions."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 5, // Corresponds to Day 5 activities in PRD for Part 2
      "curriculum_part_title": "Part 2: Thermodynamics - Loop Penalties & Electrostatics",
      "csm_tags": ["rna_modeling", "thermodynamics", "loop_entropy", "electrostatics", "counterions", "rna_stability"]
    },
    "labels": [
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", 
        "plan_day:5", 
        "activity:focused_reading",
        "block:active_learning",
        "effort_tshirt:S"
    ]
    ```

**Task 8: Document Environmental Influences on RNA Stability**
*   **Corresponds to PRD Section:** Final part of "Part 2: Mastering Foundational Thermodynamic Principles...". Covers temperature, salt, and pH effects on stability.
*   **Proposed JSON Additions:**
    ```json
    // ... existing Task 8 fields ...
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Part2.Thermo.EnvFactors",
      "anchor_link": "#part-2-mastering-foundational-thermodynamic-principles-for-rna-folding-days-3-5---approx-4-5-hours-total-study--activity-time"
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Explain how environmental factors like temperature, salt concentration, and pH affect RNA folding and stability, and their relevance to prediction tools.", // Derived from PRD Part 2 LO5
      "estimated_effort_tshirt": "S", 
      "estimated_effort_hours_raw": "0.5-1 hour", // Shorter concluding task for Part 2
      "estimated_effort_hours_min": 0.5,
      "estimated_effort_hours_max": 1.0,
      "mastery_criteria_summary": "Notes summarizing effects of temperature, salt, and pH. Short summary on relevance to RNA structure prediction tools. Flashcards created.",
      "activity_type": "focused_reading_summary_writing_flashcard_creation",
      "recommended_block": "passive_review", // Good for consolidating information
      "deliverables": [
        "Notes on how temperature, salt concentration, and pH affect RNA stability.",
        "Short summary of the relevance of these environmental factors to RNA structure prediction tools.",
        "Flashcards for temperature, salt concentration, and pH effects on RNA."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 5, // Continues Day 5 activities in PRD for Part 2
      "curriculum_part_title": "Part 2: Thermodynamics - Environmental Factors",
      "csm_tags": ["rna_modeling", "thermodynamics", "temperature_effects", "salt_effects", "ph_effects", "rna_stability"]
    },
    "labels": [
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", 
        "plan_day:5", 
        "activity:summary_writing",
        "block:passive_review",
        "effort_tshirt:S"
    ]
    ```

**Task 9: Develop Comprehensive Self-Assessment Materials**
*   **Corresponds to PRD Section:** First part of "Part 3: Consolidation, Self-Assessment & Reflection (Days 6-7 - Approx. 3-4 hours total)". Focuses on creating the quiz and practical assessment (PRD Tasks 3.1, part of 3.3).
*   **Proposed JSON Additions:**
    ```json
    // ... existing Task 9 fields ...
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Part3.Assessment.Creation",
      "anchor_link": "#part-3-consolidation-self-assessment--reflection-days-6-7---approx-3-4-hours-total"
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Create a comprehensive self-quiz (15-20 questions) and design a practical assessment task (drawing/explanation) covering Week 1's RNA biochemistry and thermodynamics learning objectives. Develop associated grading rubrics.", // Derived from PRD Part 3 LO6
      "estimated_effort_tshirt": "M", 
      "estimated_effort_hours_raw": "1.5-2 hours", // Portion of Part 3 effort
      "estimated_effort_hours_min": 1.5,
      "estimated_effort_hours_max": 2.0,
      "mastery_criteria_summary": "Comprehensive 15-20 question quiz created. Practical assessment task (drawing & explanation) designed. Grading rubrics for both developed.", // Based on PRD Task 3.1 and part of 3.3 deliverables
      "activity_type": "assessment_design_quiz_creation",
      "recommended_block": "active_learning", // Requires focused design work
      "deliverables": [
        "Self-created comprehensive quiz (15-20 questions covering all LOs) (Task 3.1 from PRD for Day 6).",
        "Design for practical assessment task (drawing a nucleotide & explaining energetic contributions) (Part of Task 3.3 from PRD for Day 6).",
        "Grading rubrics for both the quiz and the practical assessment."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 6, // Corresponds to Day 6 activities in PRD for Part 3
      "curriculum_part_title": "Part 3: Self-Assessment - Material Creation",
      "csm_tags": ["rna_modeling", "self_assessment", "quiz_design", "rubrics", "pedagogy"]
    },
    "labels": [
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", 
        "plan_day:6", 
        "activity:assessment_design",
        "block:active_learning",
        "effort_tshirt:M"
    ]
    ```

**Task 10: Create Learning Reflection and Progress Tracking System**
*   **Corresponds to PRD Section:** Second part of "Part 3: Consolidation, Self-Assessment & Reflection". Focuses on *completing* the self-assessments, reflecting, and HPE integration (PRD Tasks 3.2, rest of 3.3, 3.G). Note: The original task title "Create Learning Reflection and Progress Tracking System" might be slightly misleading; it's more about *using* such systems and reflecting. The PRD itself *is* the system's plan for reflection and tracking.
*   **Proposed JSON Additions:**
    ```json
    // ... existing Task 10 fields ...
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Part3.Assessment.CompletionReflection",
      "anchor_link": "#part-3-consolidation-self-assessment--reflection-days-6-7---approx-3-4-hours-total"
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Complete self-assessments (quiz and practical task), reflect on learning process and outcomes for Week 1, finalize flashcards, and log all progress in HPE Task Master.", // Derived from PRD Part 3 LO6 activities
      "estimated_effort_tshirt": "M", 
      "estimated_effort_hours_raw": "1.5-2 hours", // Portion of Part 3 effort
      "estimated_effort_hours_min": 1.5,
      "estimated_effort_hours_max": 2.0,
      "mastery_criteria_summary": "Self-quiz completed with score (target >=85%). Practical task completed and self-graded against rubric. Written learning reflection completed. All Week 1 flashcards finalized. All study time and deliverables logged in Task Master.", // Based on PRD Tasks 3.2, 3.3, 3.G deliverables
      "activity_type": "self_assessment_reflection_flashcard_review_logging",
      "recommended_block": "passive_review", // Reflection and logging can be lower intensity; assessment itself might be active.
      "deliverables": [
        "Completed self-quiz with score (Task 3.2 PRD for Day 6/7).",
        "Completed practical assessment (drawing/explanation) and self-graded result (Task 3.3 PRD for Day 6/7).",
        "Written learning reflection for Week 1 (Task 3.G PRD for Day 7).",
        "Finalized and organized flashcard set for Week 1 material.",
        "All study time and deliverables for Week 1 logged and status updated in Task Master."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 7, // Corresponds to Day 7 activities in PRD for Part 3
      "curriculum_part_title": "Part 3: Self-Assessment - Completion, Reflection & Integration",
      "csm_tags": ["rna_modeling", "self_assessment", "reflection", "hpe_logging", "flashcard_review", "consolidation"]
    },
    "labels": [
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", 
        "plan_day:7", 
        "activity:self_assessment",
        "activity:reflection",
        "block:passive_review", // Or a mix if assessment is intense
        "effort_tshirt:M"
    ]
    ```

---

**Refined Important Considerations for Your Parser/Generator:**

1.  **Subtask Metadata:**
    *   For now, this proposal focuses on adding rich `hpe_` metadata to **parent tasks**. Subtasks (as defined in your `tasks.json` and `task_NNN.txt` files) primarily capture the "Implementation steps" or specific actions from the PRD.
    *   Your parser could initially populate subtask `details` from the PRD and link them to the parent.
    *   Later, you might decide to add a *subset* of `hpe_` metadata (e.g., `estimated_effort_hours_min/max` if granular breakdowns are in the PRD for sub-activities) to subtasks if needed for finer-grained scheduling by your HPE system. However, the parent task's metadata (like `recommended_block` or `activity_type`) often provides sufficient context for its constituent subtasks.

2.  **Effort Distribution and `csm_id` Granularity:**
    *   The PRD (`rna-modeling_p1-foundations_week1-7day.md`) has high-level "Parts" with total effort (e.g., "Part 1: Approx. 4-5 hours"). The examples above correctly break this down by assigning more granular `estimated_effort_hours_raw` to the Task Master parent tasks that represent *segments* of these PRD Parts.
    *   Your `task_generator.py` will need to:
        *   Identify these logical segments within the PRD (e.g., "Nucleotide Structure" within "Part 1 Biochemistry").
        *   Assign a unique `csm_id` to each segment (as shown in the examples, e.g., `RNA.P1.Foundations.W1.Part1.Biochem.NucleotideStructure`).
        *   Distribute the total effort of the PRD "Part" among these generated Task Master tasks. This might require heuristics or explicit effort breakdowns in your PRD source. The examples above show this manual breakdown, which your parser would need to replicate.

3.  **Dynamic Label Generation:**
    *   This is crucial. The `labels` array should be programmatically constructed by your `task_generator.py` based on the values in the `hpe_` fields.
    *   For `activity_type` with multiple activities (e.g., "drawing_diagramming_notetaking_flashcard_creation"), you might generate multiple labels like `activity:drawing`, `activity:notetaking`, etc., as shown in the Task 2 example. This makes filtering more powerful.

4.  **Updating Tasks (Idempotency):**
    *   The `hpe_csm_reference.csm_id` is key. When re-parsing, your `task_generator.py` should:
        1.  Read the existing `tasks.json`.
        2.  For each curriculum unit parsed, check if a task with the corresponding `csm_id` already exists.
        3.  If yes, update its fields (title, description, all `hpe_` fields, regenerate labels, etc.) with the new information from the curriculum. Task Master has `update-task` and `update-subtask` CLI commands, but for batch updates, your script will likely modify the JSON object directly before saving.
        4.  If no, create a new task entry.
        5.  This prevents duplicates and keeps `tasks.json` synchronized with your curriculum.

5.  **Source Document Specificity:**
    *   The `hpe_csm_reference.source_document` in the examples correctly points to `rna-modeling_p1-foundations_week1-7day.md`. This is appropriate as these tasks are directly derived from this specific weekly learning plan.
    *   If these weekly plans are themselves derived from a higher-level, more general CSM (like `SKILL_MAP_CSM_pillar1.md`), you might consider adding another reference field, or ensuring the `csm_id` incorporates a link back to the originating skill node in the main CSM if that level of traceability is needed by your parser for other purposes. For *these specific 10 tasks*, the current approach is sound.

This comprehensive structure, when populated by your parser, will make `tasks.json` an incredibly powerful and detailed representation of your learning plan, fully integrated with your HPE doctrines. The key will be the intelligence built into your `curriculum_parser.py` to accurately extract and map this information.You've provided an excellent and detailed "incomplete analysis" that lays a very strong foundation for enhancing `tasks.json`. Your proposed new fields and the strategy for populating them from the RNA modeling curriculum (`rna-modeling_p1-foundations_week1-7day.md`) are logical, comprehensive, and directly address the need for richer metadata for your custom parser and HPE (Holistic Performance Enhancement) system.

My analysis will confirm the utility of your proposed fields, offer minor refinements or points of emphasis, and then systematically re-present the JSON additions for each of the 10 tasks, ensuring they are robust and align with your overall system design.

**I. Analysis of Proposed New Fields:**

The introduction of `hpe_csm_reference`, `hpe_learning_meta`, and `hpe_scheduling_meta` objects, along with the strategic population of the native `labels` field, is a sound approach.

1.  **`hpe_csm_reference` (Object):**
    *   **Purpose:** Establishes a vital link between the task entry in `tasks.json` and its canonical definition in the Comprehensive Skill Map (CSM) or curriculum document. This is crucial for traceability, context, and enabling idempotent updates by your parser.
    *   **Fields:**
        *   `source_document` (String): Correctly identifies the origin file. Path consistency will be important for the parser.
        *   `csm_id` (String): **Excellent choice.** This unique, human-readable identifier is the cornerstone for allowing your parser to find and update existing tasks rather than creating duplicates. The hierarchical structure (e.g., `RNA.P1.Foundations.W1.Task0`) is good practice.
        *   `anchor_link` (String, Optional): Very useful for quick navigation from a task management UI (if supported) or for developers/learners to jump directly to the relevant curriculum section.
    *   **Assessment:** Robust and essential for system integrity.

2.  **`hpe_learning_meta` (Object):**
    *   **Purpose:** Encapsulates metadata directly related to the learning content, objectives, effort, and deliverables.
    *   **Fields:**
        *   `learning_objective_summary` (String): Provides a clear, concise statement of what the learner should achieve. More specific than a general task description.
        *   `estimated_effort_tshirt` (String, Optional): Good for high-level categorization and quick sorting if data is available in CSM.
        *   `estimated_effort_hours_raw` (String): Preserves the original effort string from the CSM, useful for display or auditing.
        *   `estimated_effort_hours_min` (Float) & `estimated_effort_hours_max` (Float): **Critical for scheduling.** Parsing into numerical values allows for quantitative planning and load balancing in your HPE system.
        *   `mastery_criteria_summary` (String): Defines "done" and success for the learning unit. This can directly feed into Task Master's `testStrategy` or be used for more detailed rubrics.
        *   `activity_type` (String): Essential for filtering tasks based on learning modality (e.g., "planning_setup", "focused_reading"). The examples provided are good. A compound string like "drawing_diagramming_notetaking_flashcard_creation" is acceptable if it represents an integrated block of activities; your parser would need to handle this. Alternatively, an array could be used if activities are distinct and independently filterable. For now, the string approach is fine.
        *   `recommended_block` (String): Direct and powerful link to your `My_Optimized_Performance_Schedule_v1.0.md`, enabling alignment of tasks with personal energy/focus rhythms.
        *   `deliverables` (Array of Strings): Clearly lists expected outputs, making progress tangible and verifiable.
    *   **Assessment:** Very comprehensive and provides rich data for both the learner and any automated scheduling/tracking systems.

3.  **`hpe_scheduling_meta` (Object):**
    *   **Purpose:** Contains metadata primarily for organizing and contextualizing tasks within a broader schedule or curriculum structure.
    *   **Fields:**
        *   `planned_day_of_week` (Integer, Optional): Directly supports weekly planning if your curriculum is structured this way (as suggested by the source `rna-modeling_p1-foundations_week1-7day.md`).
        *   `curriculum_part_title` (String): Human-readable title for the larger curriculum section this task belongs to, providing context.
        *   `csm_tags` (Array of Strings): Flexible tagging from the CSM for enhanced searchability, filtering, and grouping beyond the primary `csm_id`.
    *   **Assessment:** Useful for organization and high-level planning.

4.  **`labels` (Array of Strings - Native Task Master field):**
    *   **Strategy:** Populating this native Task Master field based on the `hpe_` metadata is an intelligent way to leverage existing Task Master filtering capabilities without modifying Task Master itself.
    *   **Examples provided (`domain:rna`, `pillar:1`, `week:2025-W21`, `day:1`, `block:active-learning`, `activity:reading`) are excellent and demonstrate the power of this approach.**
    *   **Refinement:** Consider consistent prefixes for labels that are derived from specific `hpe_` fields, e.g., `hpe_block:active-learning`, `hpe_activity:reading` to distinguish them from other potential manual labels, though your examples are already quite clear. The current examples are perfectly fine.
    *   **Assessment:** Smart and efficient.

**II. Detailed JSON Additions for the 10 Tasks:**

Below are the proposed JSON additions for each of the 10 parent tasks. These additions would be merged into the existing Task Master task objects in `tasks.json`. The `id`, `title`, `description`, `status`, `dependencies`, `priority`, `details`, and `subtasks` fields from the original `tasks.json` remain untouched by these additions, serving their standard Task Master purpose. Your parser would inject these new `hpe_` objects and populate the `labels` array.

---

**Task 1: Set up RNA Biophysics Knowledge Base Structure**
*   **Corresponding PRD Section:** "Task 0: Setup, Planning & HPE Integration (Day 1 - Approx. 1-1.5 hours)"
*   **JSON Additions:**
    ```json
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Task0",
      "anchor_link": "#task-0-setup-planning--hpe-integration-day-1---approx-1-15-hours"
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Orient to Week 1 learning, prepare the learning environment, and set up Task Master & flashcard system for RNA foundations topics.",
      "estimated_effort_tshirt": "S",
      "estimated_effort_hours_raw": "1-1.5 hours",
      "estimated_effort_hours_min": 1.0,
      "estimated_effort_hours_max": 1.5,
      "mastery_criteria_summary": "Task Master entries for Week 1 created. Knowledge base section for 'Pillar 1 Foundations' initialized. Flashcard authoring environment ready.",
      "activity_type": "planning_setup",
      "recommended_block": "active_learning",
      "deliverables": [
        "Task Master entries created for the week.",
        "Knowledge base section for 'Pillar 1 Foundations' initialized."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 1,
      "curriculum_part_title": "Task 0: Setup, Planning & HPE Integration",
      "csm_tags": ["hpe_integration", "planning", "setup", "rna_foundations", "week1_orientation"]
    },
    "labels": [
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", 
        "plan_day:1", 
        "activity:planning_setup", 
        "block:active_learning",
        "effort_tshirt:S"
    ]
    ```

**Task 2: Develop RNA Nucleotide Structure Materials**
*   **Corresponding PRD Section:** Part of "Part 1: Mastering RNA Biochemistry Fundamentals (Days 1-3 - Approx. 4-5 hours total study & activity time)". This task specifically covers PRD Learning Objective 1 (chemical composition) and associated activities like drawing nucleotide/base structures (Tasks 1.1, 1.2 from PRD Day 1/2 activities).
*   **JSON Additions:**
    ```json
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Part1.Biochem.NucleotideStructure",
      "anchor_link": "#part-1-mastering-rna-biochemistry-fundamentals-days-1-3---approx-4-5-hours-total-study--activity-time"
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Master the chemical composition of RNA nucleotides, accurately draw generic nucleotide and base structures, and create related explanatory notes and flashcards.",
      "estimated_effort_tshirt": "M", 
      "estimated_effort_hours_raw": "1.5-2 hours", 
      "estimated_effort_hours_min": 1.5,
      "estimated_effort_hours_max": 2.0,
      "mastery_criteria_summary": "Accurate drawings of generic RNA nucleotide and the four standard RNA bases (A,U,G,C). Comprehensive notes cover components and properties (nucleosides vs. nucleotides). Initial flashcards created for structures and terms.",
      "activity_type": "drawing_diagramming_notetaking_flashcard_creation",
      "recommended_block": "active_learning",
      "deliverables": [
        "Completed drawing of a generic RNA nucleotide (Task 1.1 PRD).",
        "Completed detailed chemical structures of Adenine, Guanine, Cytosine, and Uracil (Task 1.2 PRD).",
        "Notes on nucleosides vs. nucleotides and chemical properties of components.",
        "Initial batch of flashcards for nucleotide structures and related terms."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 1, 
      "curriculum_part_title": "Part 1: RNA Biochemistry - Nucleotide & Base Structures",
      "csm_tags": ["rna_modeling", "biochemistry", "nucleotides", "chemical_structure", "purines", "pyrimidines"]
    },
    "labels": [
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", 
        "plan_day:1", 
        "activity:drawing", 
        "activity:notetaking",
        "activity:flashcards",
        "block:active_learning",
        "effort_tshirt:M"
    ]
    ```

**Task 3: Document RNA Polymer Structure and Backbone**
*   **Corresponding PRD Section:** Part of "Part 1: Mastering RNA Biochemistry Fundamentals". This task covers the phosphodiester backbone and directionality (PRD Task 1.3 for Day 2).
*   **JSON Additions:**
    ```json
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Part1.Biochem.BackboneDirectionality",
      "anchor_link": "#part-1-mastering-rna-biochemistry-fundamentals-days-1-3---approx-4-5-hours-total-study--activity-time"
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Explain phosphodiester backbone formation, RNA 5'-3' directionality, its implications, and illustrate with a dinucleotide drawing.",
      "estimated_effort_tshirt": "S",
      "estimated_effort_hours_raw": "1-1.5 hours",
      "estimated_effort_hours_min": 1.0,
      "estimated_effort_hours_max": 1.5,
      "mastery_criteria_summary": "Accurate drawing of an A-U dinucleotide showing the 5'-3' phosphodiester bond, 5' end, and 3' end (Task 1.3 PRD). Clear explanation of directionality and backbone properties. Flashcards created.",
      "activity_type": "drawing_explanation_writing_flashcard_creation",
      "recommended_block": "active_learning",
      "deliverables": [
        "Drawing of A-U dinucleotide with labeled 5'-3' phosphodiester bond, 5' end, and 3' end (Task 1.3 PRD).",
        "Documentation on phosphodiester backbone formation and its properties (e.g., charge).",
        "Explanation of RNA 5'-to-3' directionality significance.",
        "Flashcards for phosphodiester bond characteristics and RNA directionality."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 2,
      "curriculum_part_title": "Part 1: RNA Biochemistry - Polymer Structure & Backbone",
      "csm_tags": ["rna_modeling", "biochemistry", "phosphodiester_backbone", "directionality", "polymer_structure"]
    },
    "labels": [
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", 
        "plan_day:2", 
        "activity:drawing", 
        "activity:explanation_writing",
        "activity:flashcards",
        "block:active_learning",
        "effort_tshirt:S"
    ]
    ```

**Task 4: Develop RNA vs. DNA Comparison Materials**
*   **Corresponding PRD Section:** Part of "Part 1: Mastering RNA Biochemistry Fundamentals". Focuses on PRD Learning Objective 3 and specific task 1.4 (significance of 2'-OH group) for Day 3.
*   **JSON Additions:**
    ```json
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Part1.Biochem.RNAvsDNA",
      "anchor_link": "#part-1-mastering-rna-biochemistry-fundamentals-days-1-3---approx-4-5-hours-total-study--activity-time"
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Detail structural and functional differences between RNA and DNA, with special emphasis on the significance of the 2'-hydroxyl group in RNA.",
      "estimated_effort_tshirt": "S", 
      "estimated_effort_hours_raw": "1-1.5 hours", 
      "estimated_effort_hours_min": 1.0,
      "estimated_effort_hours_max": 1.5,
      "mastery_criteria_summary": "Comprehensive written comparison of RNA vs. DNA. Clear and accurate explanation of the 2'-OH group's role in RNA structure (A-form helix, C3'-endo pucker), reactivity (alkaline hydrolysis, catalysis), and functional versatility (Task 1.4 PRD).",
      "activity_type": "comparative_analysis_explanation_writing",
      "recommended_block": "active_learning",
      "deliverables": [
        "Written explanation for Task 1.4 (Significance of the 2'-hydroxyl group on ribose for RNA properties compared to DNA).",
        "Summary notes or table contrasting RNA and DNA (sugar, base composition, strandedness, typical helical form, stability, primary functions)."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 3,
      "curriculum_part_title": "Part 1: RNA Biochemistry - RNA vs. DNA Comparison",
      "csm_tags": ["rna_modeling", "biochemistry", "rna_vs_dna", "2_hydroxyl_group", "nucleic_acid_comparison"]
    },
    "labels": [
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", 
        "plan_day:3", 
        "activity:comparative_analysis",
        "activity:explanation_writing",
        "block:active_learning",
        "effort_tshirt:S"
    ]
    ```

**Task 5: Develop Thermodynamic Principles Materials**
*   **Corresponding PRD Section:** Part of "Part 2: Mastering Foundational Thermodynamic Principles for RNA Folding (Days 3-5)". This covers Gibbs Free Energy, MFE, enthalpy, entropy (PRD Task 2.1 for Day 3).
*   **JSON Additions:**
    ```json
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Part2.Thermo.Intro",
      "anchor_link": "#part-2-mastering-foundational-thermodynamic-principles-for-rna-folding-days-3-5---approx-4-5-hours-total-study--activity-time"
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Explain Gibbs Free Energy (ΔG), enthalpy (ΔH), entropy (ΔS), their relationship (ΔG = ΔH - TΔS), and their relevance to RNA folding spontaneity and stability, including the concept of Minimum Free Energy (MFE).",
      "estimated_effort_tshirt": "S", 
      "estimated_effort_hours_raw": "1-1.5 hours", 
      "estimated_effort_hours_min": 1.0,
      "estimated_effort_hours_max": 1.5,
      "mastery_criteria_summary": "Clear written explanation of ΔG and why a negative ΔG indicates a favorable, spontaneous process like RNA folding (Task 2.1 PRD). Flashcards for ΔG, ΔH, ΔS, MFE, spontaneity, and chemical equilibrium created.",
      "activity_type": "focused_reading_explanation_writing_flashcard_creation",
      "recommended_block": "active_learning",
      "deliverables": [
        "Written explanation of Gibbs Free Energy (ΔG = ΔH - TΔS) and why a more negative ΔG is favorable for RNA folding (Task 2.1 PRD).",
        "Flashcards defining ΔG, ΔH, ΔS, T, MFE, spontaneity, and chemical equilibrium in the context of RNA folding."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 3,
      "curriculum_part_title": "Part 2: Thermodynamics - Gibbs Free Energy & MFE",
      "csm_tags": ["rna_modeling", "thermodynamics", "gibbs_free_energy", "enthalpy", "entropy", "mfe", "spontaneity"]
    },
    "labels": [
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", 
        "plan_day:3", 
        "activity:focused_reading",
        "activity:explanation_writing",
        "activity:flashcards",
        "block:active_learning",
        "effort_tshirt:S"
    ]
    ```

**Task 6: Document Base Stacking and Hydrogen Bonding in RNA**
*   **Corresponding PRD Section:** Part of "Part 2: Mastering Foundational Thermodynamic Principles...". This covers base stacking, H-bonding, and the thought experiment (PRD Tasks 2.2, 2.4 for Day 4).
*   **JSON Additions:**
    ```json
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Part2.Thermo.StackingHBonding",
      "anchor_link": "#part-2-mastering-foundational-thermodynamic-principles-for-rna-folding-days-3-5---approx-4-5-hours-total-study--activity-time"
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Detail base stacking interactions (nature, significance, sequence dependence) and hydrogen bonding (A-U, G-C, G-U wobble) in RNA, contrasting their roles and relative energetic contributions to RNA helix stability.",
      "estimated_effort_tshirt": "M",
      "estimated_effort_hours_raw": "1.5-2 hours", 
      "estimated_effort_hours_min": 1.5,
      "estimated_effort_hours_max": 2.0,
      "mastery_criteria_summary": "Written explanation contrasting roles and relative contributions of base stacking vs. H-bonding (Task 2.2 PRD). Justified prediction for hairpin stability thought experiment (Task 2.4 PRD). Flashcards created.",
      "activity_type": "focused_reading_explanation_writing_problem_solving_flashcard_creation",
      "recommended_block": "active_learning",
      "deliverables": [
        "Detailed written explanation contrasting base stacking (van der Waals, hydrophobic, π-π interactions) vs. hydrogen bonding (A-U, G-C, G-U wobble) in RNA helix stabilization (Task 2.2 PRD).",
        "Justified prediction for the thought experiment comparing stability of a 5 G-C pair hairpin vs. a 5 A-U pair hairpin (Task 2.4 PRD).",
        "Flashcards covering base stacking interactions and hydrogen bonding patterns in RNA."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 4,
      "curriculum_part_title": "Part 2: Thermodynamics - Base Stacking & H-Bonding",
      "csm_tags": ["rna_modeling", "thermodynamics", "base_stacking", "hydrogen_bonding", "rna_stability", "wobble_pair"]
    },
    "labels": [
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", 
        "plan_day:4", 
        "activity:focused_reading",
        "activity:problem_solving",
        "activity:flashcards",
        "block:active_learning",
        "effort_tshirt:M"
    ]
    ```

**Task 7: Document Loop Penalties and Electrostatic Effects in RNA**
*   **Corresponding PRD Section:** Part of "Part 2: Mastering Foundational Thermodynamic Principles...". This covers loop penalties and electrostatic repulsion (PRD Task 2.3 for Day 5).
*   **JSON Additions:**
    ```json
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Part2.Thermo.LoopsElectrostatics",
      "anchor_link": "#part-2-mastering-foundational-thermodynamic-principles-for-rna-folding-days-3-5---approx-4-5-hours-total-study--activity-time"
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Explain loop penalties (entropic cost associated with conformational restriction of hairpins, internal loops, bulges, multi-loops) and electrostatic repulsion (from negatively charged phosphate backbone and role of counterions) in RNA stability.",
      "estimated_effort_tshirt": "S", 
      "estimated_effort_hours_raw": "1-1.5 hours", 
      "estimated_effort_hours_min": 1.0,
      "estimated_effort_hours_max": 1.5,
      "mastery_criteria_summary": "Clear explanation of why forming a hairpin loop is entropically unfavorable and influencing factors (Task 2.3 PRD). Notes on electrostatic repulsion and the mitigating role of counterions (e.g., Mg²+). Flashcards created.",
      "activity_type": "focused_reading_explanation_writing_flashcard_creation",
      "recommended_block": "active_learning",
      "deliverables": [
        "Written explanation of why forming a hairpin loop is entropically unfavorable and what factors influence this contribution (Task 2.3 PRD).",
        "Notes on electrostatic repulsion from the phosphate backbone and the role of counterions in mitigating it.",
        "Flashcards for loop penalties (hairpin, internal, bulge, multiloop), electrostatic repulsion, and counterions."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 5,
      "curriculum_part_title": "Part 2: Thermodynamics - Loop Penalties & Electrostatics",
      "csm_tags": ["rna_modeling", "thermodynamics", "loop_entropy", "electrostatics", "counterions", "rna_stability"]
    },
    "labels": [
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", 
        "plan_day:5", 
        "activity:focused_reading",
        "activity:explanation_writing",
        "activity:flashcards",
        "block:active_learning",
        "effort_tshirt:S"
    ]
    ```

**Task 8: Document Environmental Influences on RNA Stability**
*   **Corresponding PRD Section:** Part of "Part 2: Mastering Foundational Thermodynamic Principles...". This covers temperature, salt, and pH effects.
*   **JSON Additions:**
    ```json
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Part2.Thermo.EnvFactors",
      "anchor_link": "#part-2-mastering-foundational-thermodynamic-principles-for-rna-folding-days-3-5---approx-4-5-hours-total-study--activity-time"
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Explain how environmental factors like temperature (via TΔS term), salt concentration (cation shielding), and pH (base protonation) affect RNA folding and stability, and their relevance to structure prediction tools.",
      "estimated_effort_tshirt": "S",
      "estimated_effort_hours_raw": "0.5-1 hour", 
      "estimated_effort_hours_min": 0.5,
      "estimated_effort_hours_max": 1.0,
      "mastery_criteria_summary": "Notes summarizing effects of temperature, salt, and pH. Short summary of their relevance to RNA structure prediction tools. Flashcards created.",
      "activity_type": "focused_reading_summary_writing_flashcard_creation",
      "recommended_block": "passive_review", 
      "deliverables": [
        "Notes on how temperature, salt concentration, and pH affect RNA stability.",
        "Short summary of the relevance of these environmental factors to RNA structure prediction tools.",
        "Flashcards for temperature, salt concentration, and pH effects on RNA stability."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 5,
      "curriculum_part_title": "Part 2: Thermodynamics - Environmental Factors",
      "csm_tags": ["rna_modeling", "thermodynamics", "temperature_effects", "salt_effects", "ph_effects", "rna_stability"]
    },
    "labels": [
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", 
        "plan_day:5", 
        "activity:focused_reading",
        "activity:summary_writing",
        "activity:flashcards",
        "block:passive_review",
        "effort_tshirt:S"
    ]
    ```

**Task 9: Develop Comprehensive Self-Assessment Materials**
*   **Corresponding PRD Section:** "Part 3: Consolidation, Self-Assessment & Reflection (Days 6-7)". Specifically quiz creation and practical assessment design (PRD Tasks 3.1 and part of 3.3 for Day 6).
*   **JSON Additions:**
    ```json
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Part3.Assessment.Creation",
      "anchor_link": "#part-3-consolidation-self-assessment--reflection-days-6-7---approx-3-4-hours-total"
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Create a comprehensive self-quiz (15-20 questions covering all Week 1 LOs) and design a practical assessment task (drawing a nucleotide & explaining energetic contributions), along with associated grading rubrics.",
      "estimated_effort_tshirt": "M", 
      "estimated_effort_hours_raw": "1.5-2 hours", 
      "estimated_effort_hours_min": 1.5,
      "estimated_effort_hours_max": 2.0,
      "mastery_criteria_summary": "Comprehensive 15-20 question quiz created with varied question types. Practical assessment task designed. Detailed grading rubrics for both developed, targeting ≥85% for mastery.",
      "activity_type": "assessment_design_quiz_creation",
      "recommended_block": "active_learning",
      "deliverables": [
        "Self-created comprehensive quiz (15-20 questions covering all LOs) with answer key (Task 3.1 PRD).",
        "Design for practical assessment task: Draw detailed, labeled RNA nucleotide; Explain key energetic contributions to RNA secondary structure stability (Part of Task 3.3 PRD).",
        "Grading rubrics for both the quiz and the practical assessment (≥85% mastery target)."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 6,
      "curriculum_part_title": "Part 3: Self-Assessment - Material Creation",
      "csm_tags": ["rna_modeling", "self_assessment", "quiz_design", "rubrics", "pedagogy", "week1_review"]
    },
    "labels": [
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", 
        "plan_day:6", 
        "activity:assessment_design",
        "activity:quiz_creation",
        "block:active_learning",
        "effort_tshirt:M"
    ]
    ```

**Task 10: Create Learning Reflection and Progress Tracking System**
*   **Corresponding PRD Section:** "Part 3: Consolidation, Self-Assessment & Reflection (Days 6-7)". Focuses on *completing* assessments, reflecting, and HPE integration (PRD Tasks 3.2, rest of 3.3, 3.G for Day 6/7).
*   **JSON Additions:**
    ```json
    "hpe_csm_reference": {
      "source_document": "cultivation/docs/5_biology/RNA_MODELING/rna-modeling_p1-foundations_week1-7day.md",
      "csm_id": "RNA.P1.Foundations.W1.Part3.Assessment.CompletionReflection",
      "anchor_link": "#part-3-consolidation-self-assessment--reflection-days-6-7---approx-3-4-hours-total"
    },
    "hpe_learning_meta": {
      "learning_objective_summary": "Complete self-assessments (quiz and practical task), reflect on learning process and outcomes for Week 1, finalize flashcards, and log all progress in HPE Task Master.",
      "estimated_effort_tshirt": "M", 
      "estimated_effort_hours_raw": "1.5-2 hours", 
      "estimated_effort_hours_min": 1.5,
      "estimated_effort_hours_max": 2.0,
      "mastery_criteria_summary": "Self-quiz completed and scored (target >=85%). Practical task completed and self-graded against rubric. Written learning reflection for Week 1 completed. All Week 1 flashcards finalized and organized. All study time and deliverables for Week 1 logged in Task Master.",
      "activity_type": "self_assessment_reflection_flashcard_review_logging",
      "recommended_block": "passive_review", 
      "deliverables": [
        "Completed self-quiz with score (Task 3.2 PRD).",
        "Completed practical assessment (drawing/explanation) and self-graded result (Task 3.3 PRD).",
        "Written learning reflection for Week 1 (Task 3.G PRD).",
        "Finalized and organized flashcard set for Week 1 material.",
        "All study time and deliverables for Week 1 logged and status updated in Task Master."
      ]
    },
    "hpe_scheduling_meta": {
      "planned_day_of_week": 7,
      "curriculum_part_title": "Part 3: Self-Assessment - Completion, Reflection & Integration",
      "csm_tags": ["rna_modeling", "self_assessment", "reflection", "hpe_logging", "flashcard_review", "consolidation", "week1_wrapup"]
    },
    "labels": [
        "domain:rna_modeling", 
        "pillar:1", 
        "curriculum_week:1", 
        "plan_day:7", 
        "activity:self_assessment",
        "activity:reflection",
        "activity:flashcard_review",
        "block:passive_review",
        "effort_tshirt:M"
    ]
    ```

---

**III. Final Confirmation of Important Considerations for Your Parser/Generator:**

The points you raised are indeed critical:

1.  **Subtask Metadata:**
    *   The current examples enrich parent tasks. For subtasks, their `details` field often already captures the "Implementation steps" from your PRD. If more granular HPE metadata is needed for subtasks (e.g., very specific effort estimates, different `recommended_block` for sub-steps), your parser could be extended. Start with parent-level enrichment.

2.  **Effort Distribution & `csm_id` Granularity:**
    *   Your parser will need to be intelligent about mapping larger PRD "Parts" (with aggregate effort) to more granular Task Master parent tasks, each with its own `csm_id` and distributed effort. The `csm_id` examples provided (e.g., `RNA.P1.Foundations.W1.Part1.Biochem.NucleotideStructure`) correctly reflect this necessary granularity.

3.  **Dynamic Label Generation:**
    *   Absolutely essential. The `labels` array should be programmatically built from the `hpe_` fields. For `activity_type` values like "drawing\_diagramming\_notetaking\_flashcard\_creation", creating multiple labels (e.g., `activity:drawing`, `activity:notetaking`, `activity:flashcards`) is a good strategy for finer-grained filtering, as shown in the `Task 2` example.

4.  **Updating Tasks (Idempotency):**
    *   The `hpe_csm_reference.csm_id` is the key to this. Your parser must check for existing tasks by this ID in `tasks.json` before creating new ones, and update if found. This is vital for maintaining consistency as your curriculum evolves. Task Master's `move` command (`task-master move --from <id> --to <id>`) can also be programmatically useful if your parser needs to restructure tasks during an update (e.g., a PRD section becomes a sub-section of another).

This detailed structure will make your `tasks.json` an incredibly rich and actionable dataset for driving your learning and integrating with your HPE system. The proposed fields are comprehensive and well-aligned with your documented methodologies.