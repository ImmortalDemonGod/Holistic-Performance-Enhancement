# cultivation/scripts/task_management/weekly_schedule_simulator.py
"""
Simulate a full week's schedule using the active and passive learning block schedulers.
For each day:
  - Schedule active and passive blocks
  - Mark scheduled tasks as 'done' in-memory
  - Output a consolidated weekly plan

This script does NOT modify tasks.json on disk.
"""
import copy
from datetime import datetime, timedelta
import json
import os
import sys

# Import scheduler functions
from active_learning_block_scheduler import generate_active_plan, format_schedule_for_print, load_tasks
from passive_learning_block_scheduler import generate_passive_plan

# --- Config ---
TASKS_JSON_PATH = '/Users/tomriddle1/Holistic-Performance-Enhancement/tasks/tasks.json'
WEEKLY_OUTPUT_MD = os.path.join(os.path.dirname(__file__), '../../outputs/learning_curricula_parsed/weekly_schedule_simulated.md')
START_DATE_STR = "2025-05-26"  # Monday (can be parameterized)
DAYS_TO_SIMULATE = 7

# --- Helper: Mark scheduled tasks as done in in-memory list ---
def mark_tasks_done(sim_tasks, scheduled_tasks):
    """
    Mark tasks as 'done' in-place in sim_tasks based on scheduled_tasks.
    Handles both parent tasks and promoted subtasks.
    """
    for sched in scheduled_tasks:
        # Skip default passive review
        if sched.get("id") == "Default.Passive.Review":
            continue
        csm_id = sched.get("id")
        # Try to find by CSM ID in parent tasks
        for t in sim_tasks:
            if t.get("hpe_csm_reference", {}).get("csm_id") == csm_id:
                t["status"] = "done"
                break
            # Check subtasks if present
            for st in t.get("subtasks", []):
                if st.get("hpe_csm_reference", {}).get("csm_id") == csm_id:
                    st["status"] = "done"
                    # If all subtasks done, mark parent done
                    if all(sub.get("status") == "done" for sub in t.get("subtasks", [])):
                        t["status"] = "done"
                    break
    return sim_tasks

# --- Main Simulation ---
def main():
    # Load initial tasks
    initial_tasks = load_tasks(TASKS_JSON_PATH)
    if not initial_tasks:
        print(f"ERROR: Could not load tasks from {TASKS_JSON_PATH}. Exiting.")
        sys.exit(1)
    sim_tasks = copy.deepcopy(initial_tasks)
    weekly_output = []
    start_date = datetime.strptime(START_DATE_STR, "%Y-%m-%d")

    # Compute date range for header
    end_date = start_date + timedelta(days=DAYS_TO_SIMULATE-1)
    header = f"# Simulated Weekly Schedule ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})\n"

    for day_offset in range(DAYS_TO_SIMULATE):
        current_sim_date = start_date + timedelta(days=day_offset)
        current_sim_date_str = current_sim_date.strftime("%Y-%m-%d")
        day_of_week_iso = current_sim_date.isoweekday()
        daily_plan_parts = [f"--- Schedule for {current_sim_date_str} (Day {day_of_week_iso}) ---"]

        # Active block
        scheduled_active = generate_active_plan(sim_tasks, current_sim_date_str)
        active_str = format_schedule_for_print(scheduled_active)
        if not isinstance(active_str, str):
            active_str = "No active tasks scheduled for this block."
        daily_plan_parts.append(active_str)
        sim_tasks = mark_tasks_done(sim_tasks, scheduled_active)

        # Passive block
        scheduled_passive = generate_passive_plan(sim_tasks, current_sim_date_str)
        # Use the print_schedule logic from passive scheduler for formatting
        passive_lines = []
        passive_lines.append("Learning Block (Passive Review & Consolidation) | 23:15 – 00:45 CT (90 minutes)")
        passive_lines.append("-" * 80)
        for i, task in enumerate(scheduled_passive, 1):
            passive_lines.append(f"{i}. Task ID: [{task['id']}]")
            passive_lines.append(f"   - Title: {task['title']}")
            passive_lines.append(f"   - Activity: {task['activity']}")
            passive_lines.append(f"   - Est. Effort: {task['effort']} ({task['effort_minutes_planned']:.0f} min planned)")
            passive_lines.append(f"   - Notes: {task['notes']}")
            passive_lines.append("")
        daily_plan_parts.append("\n".join(passive_lines))
        sim_tasks = mark_tasks_done(sim_tasks, scheduled_passive)

        weekly_output.append("\n".join(daily_plan_parts))

    # Write to file
    with open(WEEKLY_OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write("\n\n".join(weekly_output))
    print(f"Weekly simulated schedule written to {WEEKLY_OUTPUT_MD}")

if __name__ == "__main__":
    main()
