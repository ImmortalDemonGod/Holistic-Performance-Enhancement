---
title: "Strength Training Block: [BLOCK_NAME_PLACEHOLDER] - V[VERSION_PLACEHOLDER]"
author: "[AUTHOR_PLACEHOLDER]"
date_created: "2025-06-04"
date_modified: "2025-06-04"
status: "template"
version: "1.0"
# block_focus: "e.g., Hypertrophy, Max Strength, Peaking"
# block_duration_weeks: X
# target_lifts_for_improvement: "[Lift1, Lift2]"
---

# Strength Training Block: [BLOCK_NAME_PLACEHOLDER]

**Author:** [AUTHOR_PLACEHOLDER]
**Last Edit:** [DATE_MODIFIED_PLACEHOLDER]
**Scope:** [e.g., 4-week hypertrophy-focused block, 3-day/week full body]
**Primary Goal(s) for this Block:** [e.g., Increase estimated 1RM on Squat by 5kg, Add 2kg lean mass]

---

## 0. Block Philosophy & Overall Strategy

<!-- What is the overarching approach for this block? E.g., High volume, moderate intensity? Low volume, high intensity? Specific periodization model being followed? -->

## 1. Athlete Profile & Context (As of Block Start)

*   **Training Age/Experience Level:** [e.g., Early Intermediate]
*   **Current Key Lifts (e1RMs or recent bests):**
    *   Squat: [Value] kg/lbs
    *   Bench: [Value] kg/lbs
    *   Deadlift: [Value] kg/lbs
    *   OHP: [Value] kg/lbs
*   **Bodyweight:** [Value] kg/lbs
*   **Concurrent Training:** [e.g., Base-Ox Running Block Week X, High cognitive load from Project Y]
*   **Constraints/Considerations:** [e.g., Limited gym access on certain days, specific joint sensitivities]

## 2. Volume Landmarks & Progression Model

*   **Target Weekly Sets per Major Muscle Group:**
    *   Quads: MEV [X], MAV [Y], MRV [Z]
    *   Hamstrings: MEV [X], MAV [Y], MRV [Z]
    *   Chest: ...
    *   Back (Horizontal Pull): ...
    *   Back (Vertical Pull): ...
    *   Shoulders (Anterior/Lateral/Posterior): ...
    *   Biceps: ...
    *   Triceps: ...
    *   Core: ...
    *   Calves: ...
*   **Progression Strategy within Block:** [e.g., Linear periodization on main lifts, double progression on accessories, volume ramp week 1-3, deload week 4]
*   **Load Management Strategy:** [e.g., RPE-based, %1RM-based, RIR-based]

## 3. Weekly Microcycle Template

<!-- Repeat for each training day of the week -->
### Day 1: [Day_Name_Placeholder, e.g., Monday] - Focus: [e.g., Full Body - Squat Emphasis]

*   **A. Warm-up Protocol:**
    *   General: [e.g., 5 min light cardio]
    *   Specific: [e.g., Dynamic stretches, activation drills for target muscles]
*   **B. Workout:**
    | Exercise                         | Sets x Reps             | Load (%1RM / RPE / RIR) | Rest (min) | Tempo    | Notes / Coaching Cues                               |
    | :------------------------------- | :---------------------- | :---------------------- | :--------- | :------- | :-------------------------------------------------- |
    | [Exercise_1_Name_Placeholder]    | [e.g., 3 x 5]           | [e.g., 80% 1RM]         | [e.g., 3]  | [e.g., 20X0] | [e.g., Focus on bar path]                         |
    | [Exercise_2_Name_Placeholder]    | [e.g., 4 x 8-12]        | [e.g., RPE 8]           | [e.g., 2]  |          | [e.g., Controlled eccentric]                      |
    | ...                              |                         |                         |            |          |                                                     |
*   **C. Cool-down Protocol:**
    *   [e.g., Static stretches for worked muscles]

<!-- Add templates for other training days (Day 2, Day 3, etc.) -->

## 4. Exercise Selection & Alternatives

*   **Primary Lifts:** [List core exercises for the block]
*   **Accessory Lifts:** [List key accessory exercises]
*   **Approved Substitutions (e.g., for equipment limitations or specific needs):**
    *   If [Exercise_A] is unavailable, substitute with [Exercise_B] targeting similar muscle groups/patterns.
    *   No-Gym/Bodyweight Alternatives: [Link to or list bodyweight variations as in source doc]

## 5. Injury Mitigation & Monitoring Strategy

*   **Key Areas of Focus/Concern (based on history or block design):** [e.g., Lower back, shoulders]
*   **Specific Warm-up/Mobility Drills:** [List any mandatory prehab/mobility work]
*   **Signs to Watch For / Autoregulation Rules:** [e.g., Sharp pain = stop exercise, persistent ache > 3/10 = deload next session]

## 6. Logging, Tracking & Review

*   **Data to Log Each Session:** Sets, Reps, Weight, RPE, RIR (if used), Rest Times, Session Duration, Subjective Feel, Any Pain/Discomfort.
*   **Weekly Volume Audit:** Use `cultivation/scripts/strength_audit.py` (or similar) to check planned vs. actual sets per muscle group against MEV/MAV/MRV.
*   **Block Review Cadence:** [e.g., End of each week for micro-adjustments, full review at end of block]

## 7. Plan CSV (for `cultivation/training_plans/`)

<!-- 
Provide a sample CSV structure or instructions on how this plan translates to the 
cultivation/training_plans/[block_name].csv file for scheduler/Task Master integration.
Example columns: week, day_of_week, session_focus, exercise_name, sets_planned, reps_planned, load_target_type, load_target_value, rest_planned_sec
-->

```csv
week,day_of_week,session_focus,exercise_name,sets_planned,reps_planned,load_target_type,load_target_value,rest_planned_sec,tempo
1,Mon,Strength_SquatFocus,Barbell Back Squat,3,5,%1RM,80,180,20X0
1,Mon,Strength_SquatFocus,Romanian Deadlift,3,8,RPE,8,120,3010
... 
```

---
*Instructions for Use: Fill in placeholders. This template provides a comprehensive structure for designing and documenting a new strength training block. Adapt as needed for specific goals and periodization strategies.*
