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
def mark_tasks_done(sim_tasks_list, scheduled_tasks_details, current_day_str):
    """
    Mark tasks as 'done' in-place in sim_tasks_list based on scheduled_tasks_details.
    Handles both parent tasks and promoted subtasks.
    """
    import logging
    sim_logger = logging.getLogger("WeeklyScheduleSimulator")
    if not scheduled_tasks_details:
        return

    for scheduled_item in scheduled_tasks_details:
        if scheduled_item.get("id") == "Default.Passive.Review":
            continue

        is_promoted_sub = scheduled_item.get("_is_promoted_subtask", False)

        if is_promoted_sub:
            parent_numeric_id = scheduled_item.get("_parent_numeric_id")
            subtask_local_id = scheduled_item.get("_subtask_local_id")

            if parent_numeric_id is None or subtask_local_id is None:
                sim_logger.warning(f"Day {current_day_str} - Promoted subtask detail missing parent/local ID: {scheduled_item.get('id')}")
                continue

            parent_task_found = False
            for parent_task_in_sim in sim_tasks_list:
                if parent_task_in_sim.get("id") == parent_numeric_id:
                    parent_task_found = True
                    subtask_found_in_parent = False
                    for sub_in_parent in parent_task_in_sim.get("subtasks", []):
                        if sub_in_parent.get("id") == subtask_local_id:
                            sub_in_parent["status"] = "done"
                            subtask_found_in_parent = True
                            sim_logger.info(f"SIM_INFO: Subtask {parent_numeric_id}.{subtask_local_id} ('{sub_in_parent.get('title')[:20]}...') marked 'done'.")
                            break
                    if not subtask_found_in_parent:
                        sim_logger.warning(f"Promoted subtask {parent_numeric_id}.{subtask_local_id} not found in parent's subtask list during status update.")
                    # Check if all subtasks of this parent are now done
                    all_subs_done = True
                    if not parent_task_in_sim.get("subtasks"):
                        all_subs_done = False
                    else:
                        for sub_in_parent_check in parent_task_in_sim.get("subtasks", []):
                            if sub_in_parent_check.get("status", "pending") != "done":
                                all_subs_done = False
                                break
                    if all_subs_done:
                        parent_task_in_sim["status"] = "done"
                        sim_logger.info(f"SIM_INFO: Parent task {parent_numeric_id} ('{parent_task_in_sim.get('title')[:20]}...') marked 'done' as all its subtasks are complete.")
                    break
            if not parent_task_found:
                sim_logger.warning(f"Parent task ID {parent_numeric_id} for promoted subtask not found in sim_tasks_list.")
        else:
            task_csm_id_to_complete = scheduled_item.get("id")
            if not task_csm_id_to_complete:
                sim_logger.warning(f"Scheduled parent task missing 'id' (CSM ID): {scheduled_item.get('title')}")
                continue
            task_found_for_completion = False
            for task_in_state in sim_tasks_list:
                if task_in_state.get("hpe_csm_reference", {}).get("csm_id") == task_csm_id_to_complete:
                    task_in_state["status"] = "done"
                    task_found_for_completion = True
                    sim_logger.info(f"SIM_INFO: Parent task {task_csm_id_to_complete} ('{task_in_state.get('title')[:20]}...') marked 'done'.")
                    for st_in_state in task_in_state.get("subtasks", []):
                        st_in_state["status"] = "done"
                    break
            if not task_found_for_completion:
                sim_logger.warning(f"Parent task with CSM ID {task_csm_id_to_complete} not found in sim_tasks_list for marking done.")
    # No return needed; sim_tasks_list is modified in-place

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
        import logging
        sim_logger = logging.getLogger("WeeklyScheduleSimulator")
        sim_logger.info(f"Day {current_sim_date_str} - Active Plan Returned by generate_active_plan: {scheduled_active}")
        # --- DEBUG: Print scheduled_active and formatted output for each day ---
        sim_logger.info(f"DEBUG: scheduled_active for Day {current_sim_date_str}: {scheduled_active}")
        active_str = format_schedule_for_print(scheduled_active)
        sim_logger.info(f"DEBUG: format_schedule_for_print output for Day {current_sim_date_str}: {active_str}")
        if not isinstance(active_str, str):
            active_str = "No active tasks scheduled for this block."
        daily_plan_parts.append(active_str)
        mark_tasks_done(sim_tasks, scheduled_active, current_sim_date_str)

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
        mark_tasks_done(sim_tasks, scheduled_passive, current_sim_date_str)

        weekly_output.append("\n".join(daily_plan_parts))

    # Write to file
    with open(WEEKLY_OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write("\n\n".join(weekly_output))
    print(f"Weekly simulated schedule written to {WEEKLY_OUTPUT_MD}")

if __name__ == "__main__":
    main()
