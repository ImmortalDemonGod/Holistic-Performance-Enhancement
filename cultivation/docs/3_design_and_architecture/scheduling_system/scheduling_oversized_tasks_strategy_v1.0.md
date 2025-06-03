# Taskmaster HPE: Strategy and Design for Scheduling Oversized Learning Tasks (v1.0)

**Date:** 2025-05-26
**Author/Maintainer:** [Your Name/Handle]
**Location:** `cultivation/docs/6_scheduling/scheduling_oversized_tasks_strategy_v1.0.md`

---

## 1. Introduction & Problem Statement

Our Taskmaster-based HPE learning system uses predefined time blocks (e.g., a 60-minute "Active Acquisition & Practice" block) for focused learning. The `tasks.json` file, enriched with HPE metadata, provides `estimated_effort_hours_min` and `estimated_effort_hours_max` for each learning task.

**The core problem:** A learning task, as defined in the curriculum and `tasks.json`, may have an `estimated_effort_hours_min` that exceeds the duration of its `recommended_block`. Strict exclusion of such tasks can lead to important curriculum items being perpetually unscheduled, disrupting learning flow and curriculum adherence.

**Goal:** To define a comprehensive strategy and design modifications for our schedulers to intelligently handle oversized tasks, ensuring curriculum progression while respecting HPE doctrines and time block integrity.

---

## 2. Guiding Principles

1. **Curriculum Adherence:** Strive to schedule all defined learning tasks from the CSM.
2. **HPE Doctrine Compliance:** Align scheduling with the intended cognitive load and purpose of each time block.
3. **Task Integrity & Focus:** Avoid excessive fragmentation that disrupts learning quality.
4. **Automation & Minimized Manual Intervention:** Schedulers should make intelligent decisions autonomously.
5. **Taskmaster Compatibility:** Integrate with Taskmaster's structure (statuses, dependencies).
6. **Iterative Implementation:** Favor pragmatic, implementable solutions first, with pathways for more advanced logic later.

---

## 3. Analysis of Potential Strategies

### 3.1. Strict Exclusion (Baseline)
- **Logic:** If `task.min_effort_minutes > block_duration`, task is not scheduled in this block.
- **Pros:** Simplicity; ensures scheduled tasks are nominally fittable.
- **Cons:** Major curriculum gaps if tasks are not granular enough.
- **Current Status:** Used by `active_learning_block_scheduler.py`'s initial filter.

### 3.2. Flexible Fitting (Average/Max Effort)
- **Logic:** Schedule if `task.avg_effort_minutes <= block_duration` or `task.max_effort_minutes <= block_duration`.
- **Pros:** Schedules more tasks than strict min-effort fitting.
- **Cons:** Risk of overcommitment; still excludes some tasks.
- **Current Status:** Used by `passive_learning_block_scheduler.py` (average effort).

### 3.3. Task Chunking / Partial Scheduling (Conceptual)
- **Logic:** Schedule a fixed-duration "chunk" of a too-large task.
- **Pros:** Allows progress on any task.
- **Cons:** Complex status tracking; not natively supported by Taskmaster.

### 3.4. Subtask Promotion (Pragmatic Partial Scheduling)
- **Logic:** If a parent task is too large, but has pending subtasks with fittable effort, schedule these subtasks.
- **Pros:** Leverages Taskmaster's subtask structure; maintains task integrity; granular progress.
- **Cons:** Requires subtasks to have meaningful effort estimates.
- **Design Implication:** Schedulers must inspect subtasks and handle effort estimation heuristics.

### 3.5. Alternative Block Recommendation / Intelligent Deferral
- **Logic:** If a task is too large for its primary block, check if it is suitable for another (longer) block (e.g., "deep_work").
- **Pros:** Optimizes block usage; aligns tasks with appropriate cognitive states/durations.
- **Cons:** Requires new metadata and cross-scheduler logic.

### 3.6. Time-Slicing with Explicit Progress Tracking (Advanced)
- **Logic:** Allocate a portion of any block to any task, tracking `effort_completed_hours`.
- **Pros:** Ultimate flexibility.
- **Cons:** Requires fundamental changes to task state model.

---

## 4. Proposed Integrated Strategy for Oversized Tasks

### Layer 1: Enhance Block Schedulers with Subtask Promotion (Immediate Priority)
- If a parent task is too large:
  1. Check for pending subtasks.
  2. Use explicit subtask effort if present; otherwise, divide parent effort or use a default (e.g., 30 min).
  3. If a subtask fits, add it as a candidate for scheduling, referencing its parent.
  4. Prioritize and schedule as usual, mixing parent and subtask candidates.

### Layer 2: Intelligent Deferral and Alternative Block Suggestion (Mid-Term)
- If a task is too large for its recommended block and lists `acceptable_blocks`, attempt to schedule it in a longer block (e.g., "deep_work").
- If still unschedulable, flag for manual review.

### Layer 3: Enhancing Curriculum Granularity (Ongoing)
- Improve `curriculum_parser.py` and `task_generator.py` to generate tasks/subtasks at schedulable granularity with appropriate effort estimates.
- Encourage subtask-level HPE metadata.

---

## 5. Scheduler & Test Suite Impact

- **Schedulers:**
  - Must implement subtask promotion logic and effort heuristics.
  - Prioritization and scheduling must handle both parent and subtask candidates.
- **Test Suite:**
  - Add tests for parent-too-large/subtask-fits, all-too-large, explicit/inferred subtask effort, and correct reporting.

---

## 6. Future Enhancements

- Dynamic chunking of tasks based on available block time.
- User feedback loop for splitting oversized tasks or adjusting effort estimates.
- Predictive adjustment of effort estimation logic based on user/task type trends.

---

## 7. Conclusion

Handling oversized tasks requires a multi-layered approach. Immediate subtask promotion, mid-term intelligent deferral, and long-term curriculum granularity improvements together ensure curriculum adherence, HPE doctrine compliance, and robust automation.

---

*This document should be referenced by all developers and curriculum designers working on Taskmaster HPE scheduling logic and curriculum ingestion scripts.*
