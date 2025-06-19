# cultivation/scripts/task_management/active_learning_block_scheduler.py
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import argparse


# --- Configuration ---
ACTIVE_BLOCK_MINUTES: int = 60
MIN_REQUIRED_TASKS_FOR_DAY_FOCUS: int = 1 # Min on-day tasks before pulling from other days

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ActiveLearningBlockScheduler")

# --- Helper Functions (Potentially refactor into a shared utility module) ---

def load_tasks(tasks_json_path: str) -> List[Dict[str, Any]]:
    """Loads tasks from the specified JSON file."""
    try:
        with open(tasks_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("tasks", [])
    except FileNotFoundError:
        logger.error(f"Tasks file not found: {tasks_json_path}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {tasks_json_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading tasks: {e}")
    return []

def get_day_of_week(target_date_str: Optional[str] = None) -> int:
    """Gets the ISO day of the week (Monday=1, Sunday=7) for a target date string or current date."""
    try:
        if target_date_str:
            dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        else:
            dt = datetime.now()
        return dt.isoweekday()
    except ValueError:
        logger.warning(f"Invalid date format: {target_date_str}. Defaulting to today's date.")
        return datetime.now().isoweekday()

def build_task_status_map(all_tasks: List[Dict[str, Any]]) -> Dict[str, str]:
    """Builds a map from task identifier to task status for quick lookups."""
    status_map = {}
    for task in all_tasks:
        # Prioritize csm_id as the canonical identifier if it exists
        if "hpe_csm_reference" in task and "csm_id" in task["hpe_csm_reference"]:
            task_id = task["hpe_csm_reference"]["csm_id"]
        else:
            task_id = task.get("id")

        if task_id:
            # Ensure the key is a string for consistency
            status_map[str(task_id)] = str(task.get("status", "pending"))
    return status_map

def dependencies_met(task: Dict[str, Any], task_statuses: Dict[str, str]) -> bool:
    """Check if all dependencies for a task are met."""
    dependency_ids = task.get("dependencies", [])
    if not dependency_ids:
        return True
    for dep_id in dependency_ids:
        # The status map uses csm_id if available, so we check that first.
        # This logic needs to be robust.
        dep_status = task_statuses.get(str(dep_id)) # Ensure key is string
        if dep_status != "done":
            task_id = task.get('id', 'Unknown')
            logger.debug(f"Task {task_id} dependency {dep_id} not met (status: {dep_status}).")
            return False
    return True

# --- Active Learning Scheduler Specific Functions ---

def filter_active_tasks(
    all_tasks: List[Dict[str, Any]],
    current_day_of_week: int,
    task_statuses: Dict[Any, str],
    min_tasks_for_day_focus: int = MIN_REQUIRED_TASKS_FOR_DAY_FOCUS
) -> List[Dict[str, Any]]:
    """
    Filters tasks suitable for the Active Learning Block, including subtask promotion for oversized tasks.
    Considers 'recommended_block', 'activity_type', dependencies, status, and minimum effort fitting.
    Uses a two-pass system for day preference. If a parent task is too large, considers eligible subtasks.
    """
    candidate_tasks: List[Dict[str, Any]] = []
    
    active_activity_keywords = [
        "planning_setup", "drawing", "diagramming", "notetaking", 
        "explanation_writing", "comparative_analysis", "problem_solving", 
        "assessment_design", "quiz_creation", "self_assessment_execution", 
        "coding_exercise", "active_study", "interactive_learning", "skill_practice",
        "focused_reading"
    ]

    logger.info("Filtering for active tasks (with subtask promotion)...")
    for task in all_tasks:
        task_id = task.get('id', 'Unknown')
        if task.get("status") != "pending":
            logger.debug(f"Task {task_id} skipped: status is not 'pending' (is '{task.get('status')}').")
            continue
        if not dependencies_met(task, task_statuses):
            logger.debug(f"Task {task_id} skipped: dependencies not met.")
            continue

        hpe_learning_meta = task.get("hpe_learning_meta", {})
        recommended_block = hpe_learning_meta.get("recommended_block", "").lower()
        activity_type = hpe_learning_meta.get("activity_type", "").lower()
        
        # Robust exclusion: Never consider tasks with explicit passive/deep blocks as active
        is_active_task_type = False
        if recommended_block == "active_learning":
            is_active_task_type = True
        elif recommended_block in ("passive_review", "deep_work"):
            logger.debug(f"Task {task_id} skipped: recommended_block is explicitly '{recommended_block}'.")
            is_active_task_type = False
        elif not recommended_block and any(keyword in activity_type for keyword in active_activity_keywords):
            is_active_task_type = True
            logger.debug(f"Task {task_id} considered active due to activity_type ('{activity_type}'), recommended_block is empty.")
        # If recommended_block is non-empty but not active_learning or passive/deep, do NOT fallback to activity_type.

        if is_active_task_type:
            min_effort_hours = hpe_learning_meta.get("estimated_effort_hours_min")
            if min_effort_hours is None:
                min_effort_hours = task.get('estimated_effort_hours_min', float('inf'))
            if not isinstance(min_effort_hours, (int, float)):
                logger.warning(f"Task {task_id} has invalid min_effort_hours type: {type(min_effort_hours)}. Skipping.")
                continue

            is_oversized = (min_effort_hours * 60 > ACTIVE_BLOCK_MINUTES)

            if is_oversized:
                logger.debug(f"Task {task_id} (min_effort: {min_effort_hours}hr) inherently too large for {ACTIVE_BLOCK_MINUTES}min active block. Checking subtasks for promotion.")
                # --- Subtask Promotion Logic ---
                # Only promote subtasks that are pending and whose sibling dependencies are met.
                # Fallback effort is distributed among only these eligible subtasks.
                eligible_subtasks_for_promotion: List[Dict[str, Any]] = []
                all_subtasks_in_parent = task.get("subtasks", [])
                for subtask in all_subtasks_in_parent:
                    if subtask.get("status") != "pending":
                        continue
                    subtask_id_local = subtask.get("id")
                    sibling_dependencies = subtask.get("dependencies", [])
                    if all(
                        next((st for st in all_subtasks_in_parent if st.get("id") == dep_id), {"status": "done"}).get("status") == "done"
                        for dep_id in sibling_dependencies
                    ):
                        eligible_subtasks_for_promotion.append(subtask)

                num_eligible_pending_subtasks = len(eligible_subtasks_for_promotion)
                # NEW: Calculate total pending subtasks for fallback effort distribution
                all_pending_subtasks_for_effort_calc = [
                    st for st in all_subtasks_in_parent if st.get("status", "pending") == "pending"
                ]
                num_total_pending_subtasks_for_effort_fallback = len(all_pending_subtasks_for_effort_calc)
                if num_total_pending_subtasks_for_effort_fallback == 0:
                    num_total_pending_subtasks_for_effort_fallback = 1  # Avoid division by zero
                logger.debug(f"  Parent Task {task_id}: Found {num_eligible_pending_subtasks} pending subtasks with met sibling dependencies.")

                for subtask in eligible_subtasks_for_promotion:
                    subtask_id_local = subtask.get("id", "UnknownSub")
                    subtask_hpe_learning_meta = subtask.get("hpe_learning_meta", {})
                    sub_min_effort_hours = subtask_hpe_learning_meta.get("estimated_effort_hours_min")
                    if sub_min_effort_hours is None:
                        logger.debug(f"Subtask {task_id}.{subtask_id_local} has no explicit min_effort_hours. Using fallback: parent_min_effort {min_effort_hours} / {num_total_pending_subtasks_for_effort_fallback} total pending subtasks.")
                        if num_total_pending_subtasks_for_effort_fallback > 0 and isinstance(min_effort_hours, (int, float)) and min_effort_hours != float('inf'):
                            sub_min_effort_hours = min_effort_hours / num_total_pending_subtasks_for_effort_fallback
                            logger.debug(f"    Subtask {task_id}.{subtask_id_local}: No explicit effort. Inferred min_effort: {sub_min_effort_hours:.2f}hr (parent {min_effort_hours}hr / {num_total_pending_subtasks_for_effort_fallback} pending subtasks).")
                        else:
                            logger.warning(f"    Subtask {task_id}.{subtask_id_local}: No explicit effort and no pending subtasks for fallback calculation. Skipping.")
                            continue
                    if sub_min_effort_hours is None or not isinstance(sub_min_effort_hours, (int, float)) or sub_min_effort_hours <= 0:
                        logger.warning(f"    Subtask {task_id}.{subtask_id_local} has invalid or non-positive calculated min_effort_hours: {sub_min_effort_hours}. Skipping.")
                        continue
                    logger.debug(f"    Subtask {task_id}.{subtask_id_local}: Final min_effort_hours to check: {sub_min_effort_hours}hr (block limit: {ACTIVE_BLOCK_MINUTES/60}hr)")
                    if (sub_min_effort_hours * 60) <= ACTIVE_BLOCK_MINUTES:
                        # Promote the subtask by constructing a new dictionary
                        # to avoid inheriting unwanted parent fields like dependencies or csm_id.
                        promoted_subtask = {
                            "id": subtask.get("id"),
                            "title": f"{task.get('title', 'Task')} -> {subtask.get('title', 'Subtask')}",
                            "status": "pending",
                            "_parent_task_id": task_id,
                            "_original_parent_id": task_id,
                            # Carry over essential metadata from the parent
                            "hpe_learning_meta": task.get("hpe_learning_meta", {}),
                            "priority": task.get("priority"),
                        }
                        # Update effort from the subtask's metadata
                        promoted_subtask["hpe_learning_meta"]["estimated_effort_hours_min"] = subtask_hpe_learning_meta.get("estimated_effort_hours_min")
                        promoted_subtask["hpe_learning_meta"]["estimated_effort_hours_max"] = subtask_hpe_learning_meta.get("estimated_effort_hours_max")

                        candidate_tasks.append(promoted_subtask)
                        logger.info(f"    Promoted Subtask {task_id}.{subtask_id_local} added as candidate (min_effort: {sub_min_effort_hours:.2f}hr).")
                    else:
                        logger.info(f"    Subtask {task_id}.{subtask_id_local} (min_effort: {sub_min_effort_hours:.2f}hr) too large for block. Not promoted.")
                # --- END Subtask Promotion Logic ---
                        
                logger.info(f"Parent task {task_id} skipped for scheduling due to being oversized. No parent appended.")
            # If the task itself is not oversized and is an active learning task, add it.
            elif not is_oversized:
                candidate_tasks.append(task)
                logger.info(f"Task {task_id} added to initial candidates (min_effort: {min_effort_hours}hr). Candidate_tasks now: {[t.get('id', t.get('_parent_task_id')) for t in candidate_tasks]}")
        else:
            logger.debug(f"Task {task_id} skipped: not an active_learning task (block: '{recommended_block}', activity: '{activity_type}').")

    # Pass 1: Tasks planned for the current day
    on_day_tasks: List[Dict[str, Any]] = []
    for task in candidate_tasks:
        planned_day = task.get("hpe_scheduling_meta", {}).get("planned_day_of_week")
        # For promoted subtasks, inherit planned_day from parent if not present
        if not planned_day and "_parent_hpe_scheduling_meta" in task:
            planned_day = task["_parent_hpe_scheduling_meta"].get("planned_day_of_week")
        if planned_day == current_day_of_week:
            on_day_tasks.append(task)
            
    logger.info(f"{len(on_day_tasks)} on-day active tasks found.")
    for t in on_day_tasks: 
        logger.debug(f"  On-day candidate: {t.get('id')} (parent: {t.get('_parent_task_id', None)})")

    if len(on_day_tasks) >= min_tasks_for_day_focus:
        logger.info(f"Sufficient on-day active tasks ({len(on_day_tasks)}). Prioritizing these.")
        logger.info(f"[filter_active_tasks] Final candidate task IDs: {[t.get('id', t.get('_parent_task_id')) for t in candidate_tasks]}")
        return candidate_tasks
    
    # Pass 2: If not enough on-day tasks, add other eligible active tasks
    logger.info(f"Insufficient on-day tasks ({len(on_day_tasks)} found, {min_tasks_for_day_focus} required). Considering other eligible active tasks.")
    
    # Tasks not planned for today but still eligible active tasks
    off_day_tasks = [task for task in candidate_tasks if task not in on_day_tasks]
    logger.info(f"{len(off_day_tasks)} off-day active tasks found as potential additions.")
    for t in off_day_tasks: 
        logger.debug(f"  Off-day candidate: {t.get('id')} (parent: {t.get('_parent_task_id', None)})")

    # Combine and ensure no duplicates (though logic should prevent it)
    # The order matters for prioritization later if planned_day_of_week is a primary sort key
    final_candidates = on_day_tasks + off_day_tasks
            
    logger.info(f"Total active tasks after two passes: {len(final_candidates)}.")
    return final_candidates


def prioritize_active_tasks(tasks: List[Dict[str, Any]], current_day_of_week: int) -> List[Dict[str, Any]]:
    """Prioritizes active tasks: 1. On-day, 2. Taskmaster Priority, 3. Min Effort."""
    
    def get_priority_score(priority_str: Optional[str]) -> int:
        return {"high": 3, "medium": 2, "low": 1}.get(str(priority_str).lower(), 2) # Default to medium

    def sort_key(task: Dict[str, Any]):
        is_on_day = task.get("hpe_scheduling_meta", {}).get("planned_day_of_week") == current_day_of_week
        priority_score = get_priority_score(task.get("priority"))
        min_effort = task.get("hpe_learning_meta", {}).get("estimated_effort_hours_min", float('inf'))
        
        # Sort criteria:
        # 1. On-day tasks first (True > False, so use 'not is_on_day' for ascending sort to put True first)
        # 2. Higher Taskmaster priority first (larger score is better, so use negative for ascending sort)
        # 3. Smaller minimum effort first (smaller is better)
        # 4. Task ID as a tie-breaker (consistent ordering)
        return (not is_on_day, -priority_score, min_effort, task.get("id", 0))

    sorted_tasks = sorted(tasks, key=sort_key)
    logger.info(f"Prioritized active tasks (IDs): {[task.get('id') for task in sorted_tasks]}")
    return sorted_tasks

def schedule_active_tasks_into_block(
    prioritized_tasks: List[Dict[str, Any]],
    total_block_minutes: int = ACTIVE_BLOCK_MINUTES
) -> List[Dict[str, Any]]:
    """Schedules tasks into the active learning block based on their minimum effort."""
    scheduled_tasks_for_output: List[Dict[str, Any]] = []
    time_left_in_block = float(total_block_minutes) # Use float for precision
    
    logger.info("Attempting to schedule active tasks. Candidates:")
    for t_idx, t in enumerate(prioritized_tasks):
        logger.debug(f"[SCHEDULE_DEBUG] Candidate {t_idx + 1} for scheduling (Full Data): {t}")
        logger.debug(f"[SCHEDULE_DEBUG]   Candidate {t_idx + 1} hpe_learning_meta: {t.get('hpe_learning_meta')}")
        logger.info(f"  Candidate: id={t.get('id')} parent={t.get('_parent_task_id', None)} effort={t.get('hpe_learning_meta',{}).get('estimated_effort_hours_min')}hr")
    logger.info(f"Total block time: {total_block_minutes} min.")

    for task in prioritized_tasks:
        task_id = task.get("id", "Unknown")
        task_title = task.get("title", "No Title")
        hpe_learning_meta = task.get("hpe_learning_meta", {})
        min_effort_hours = hpe_learning_meta.get("estimated_effort_hours_min")
        logger.debug(f"[SCHEDULE_DEBUG] Task {task_id} ('{task_title}'): Initial min_effort_hours from hpe_learning_meta: {min_effort_hours}")

        if min_effort_hours is None:
            min_effort_hours = task.get('estimated_effort_hours_min', float('inf'))
            logger.debug(f"[SCHEDULE_DEBUG] Task {task_id}: min_effort_hours was None, fallback from task.get('estimated_effort_hours_min'): {min_effort_hours}")

        if min_effort_hours is None: # Should not happen if float('inf') is default
            logger.warning(f"Task {task_id} ('{task_title}') still has None for min_effort_hours. Defaulting to infinity.")
            min_effort_hours = float('inf')
        
        # Convert hours to minutes for comparison
        try:
            min_effort_minutes = float(min_effort_hours) * 60
        except (ValueError, TypeError):
            logger.warning(f"Task {task_id} ('{task_title}') has invalid min_effort_hours value: {min_effort_hours}. Skipping.")
            continue

        logger.debug(f"[SCHEDULE_DEBUG] Task {task_id}: Final min_effort_minutes for scheduling: {min_effort_minutes:.2f}")

        if min_effort_minutes == float('inf'):
            logger.info(f"  SKIPPED: Task {task_id} ('{task_title[:30]}...') (min effort: inf min) due to effectively infinite effort.")
            continue

        if min_effort_minutes <= 0:
            logger.info(f"  SKIPPED: Task {task_id} ('{task_title[:30]}...') (min effort: {min_effort_minutes:.2f} min) due to zero or negative effort.")
            continue

        if min_effort_minutes <= time_left_in_block:
            # Preserve all fields, especially subtask promotion metadata
            scheduled_task = dict(task)  # shallow copy
            scheduled_task["effort_minutes_planned"] = min_effort_minutes
            # Set 'id' to CSM ID string if present, else fallback to integer
            scheduled_task["id"] = task.get("hpe_csm_reference", {}).get("csm_id", task_id)
            scheduled_tasks_for_output.append(scheduled_task)
            time_left_in_block -= min_effort_minutes
            logger.info(f"  SCHEDULED: Task {task_id} ('{task_title[:30]}...') for {min_effort_minutes:.0f} min. Time left: {time_left_in_block:.0f} min.")
            if time_left_in_block <= 0: # Using "<= 0" as even 0 time left means block is full
                logger.info("Active learning block is now full.")
                break
        else:
            logger.info(f"  SKIPPED: Task {task_id} ('{task_title[:30]}...') (min effort: {min_effort_minutes:.2f} min) too large for remaining time ({time_left_in_block:.2f} min).")
            # Flag oversized tasks and subtasks for manual review
            if min_effort_hours > 1.0:
                task['flagged_for_review'] = True
                logger.warning(f"Task {task_id} ('{task_title}') flagged for review: estimated effort {min_effort_hours} exceeds block size.")
            
    if not scheduled_tasks_for_output:
        logger.info("No tasks were scheduled for the active learning block.")
        
    return scheduled_tasks_for_output

def format_schedule_for_print(scheduled_tasks: List[Dict[str, Any]]) -> str:
    """Formats the list of scheduled tasks into a string for printing or file output."""
    lines = []
    header = "Learning Block (Active Acquisition & Practice) | 22:00 – 23:00 CT (60 minutes)"
    separator = "-" * len(header)
    
    lines.append(header)
    lines.append(separator)
    
    if not scheduled_tasks:
        lines.append("No active tasks scheduled for this block.")
    else:
        total_planned_time = sum(task.get('effort_minutes_planned', 0) for task in scheduled_tasks)
        lines.append(f"Total Planned Time: {total_planned_time:.0f} min / {ACTIVE_BLOCK_MINUTES} min")
        lines.append(separator)
        for i, task_detail in enumerate(scheduled_tasks, 1):
            lines.append(f"{i}. Task ID: [{task_detail.get('id', 'N/A')}]")
            lines.append(f"   - Title: {task_detail.get('title', 'No Title')}")
            hpe_meta = task_detail.get("hpe_learning_meta", {})
            lines.append(f"   - Activity Type: {hpe_meta.get('activity_type', 'N/A')}")
            lines.append(f"   - Est. Effort (Raw): {hpe_meta.get('estimated_effort_hours_raw', 'N/A')}")
            lines.append(f"   - Planned Duration: {task_detail.get('effort_minutes_planned', 0):.0f} min")
            task_notes = task_detail.get('notes')
            if task_notes:
                 lines.append(f"   - Notes: {str(task_notes)[:100]}...") # Truncate long notes
            lines.append("") 
    return "\n".join(lines)

# --- Main Execution Logic ---

def generate_active_plan(
    all_tasks_data: List[Dict[str, Any]],
    target_date_str: Optional[str],
    min_focus_tasks: int = MIN_REQUIRED_TASKS_FOR_DAY_FOCUS
) -> List[Dict[str, Any]]:
    """
    Generates the active learning block plan for a given date and task list.
    Returns a list of scheduled task dicts (not printed output).
    """
    current_day = get_day_of_week(target_date_str)
    task_completion_statuses = build_task_status_map(all_tasks_data)
    eligible_active_tasks = filter_active_tasks(
        all_tasks_data,
        current_day,
        task_completion_statuses,
        min_focus_tasks
    )
    if not eligible_active_tasks:
        logger.info("No eligible active tasks found after filtering for active plan.")
        return []
    final_prioritized_tasks = prioritize_active_tasks(eligible_active_tasks, current_day)
    scheduled_block_plan = schedule_active_tasks_into_block(final_prioritized_tasks)
    return scheduled_block_plan

def main():
    """Main function to orchestrate the scheduling process."""
    parser = argparse.ArgumentParser(description="Schedule Active Learning Block Tasks.")
    parser.add_argument(
        "--tasks", 
        type=str, 
        default="/Users/tomriddle1/Holistic-Performance-Enhancement/tasks/tasks.json", 
        help="Path to the enriched tasks.json file. Default: %(default)s"
    )
    parser.add_argument(
        "--date", 
        type=str, 
        default=None, 
        help="Target date in YYYY-MM-DD format. Defaults to today."
    )
    parser.add_argument(
        "--output-md", 
        type=str, 
        default=None, 
        help="If set, write output to Markdown file."
    )
    parser.add_argument(
        "--min-focus-tasks",
        type=int,
        default=MIN_REQUIRED_TASKS_FOR_DAY_FOCUS,
        help="Minimum on-day tasks before pulling from other days. Default: %(default)s"
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true', help='Enable debug logging'
    )
    
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logging.root.handlers: # Apply to all existing handlers
            handler.setLevel(logging.DEBUG)

    all_tasks = load_tasks(args.tasks)
    if not all_tasks:
        logger.warning("No tasks loaded. Exiting.")
        schedule_output_str = format_schedule_for_print([])
        print(schedule_output_str)
        if args.output_md:
            try:
                with open(args.output_md, 'w', encoding='utf-8') as f:
                    f.write(schedule_output_str)
                logger.info(f"Empty schedule written to {args.output_md}")
            except IOError as e:
                logger.error(f"Failed to write empty schedule to Markdown file {args.output_md}: {e}")
        return

    current_day = get_day_of_week(args.date)
    logger.info(f"Target day of week: {current_day} (Monday=1, Sunday=7)")
    
    task_completion_statuses = build_task_status_map(all_tasks)
    
    eligible_active_tasks = filter_active_tasks(
        all_tasks, 
        current_day, 
        task_completion_statuses,
        args.min_focus_tasks
    )
    
    if not eligible_active_tasks:
        logger.info("No eligible active tasks found after filtering.")
        final_schedule_str = format_schedule_for_print([])
    else:
        final_prioritized_tasks = prioritize_active_tasks(eligible_active_tasks, current_day)
        scheduled_block_plan = schedule_active_tasks_into_block(final_prioritized_tasks)
        final_schedule_str = format_schedule_for_print(scheduled_block_plan)
    
    print(final_schedule_str)
    if args.output_md:
        try:
            with open(args.output_md, 'w', encoding='utf-8') as f:
                f.write(final_schedule_str)
            logger.info(f"Schedule successfully written to {args.output_md}")
        except IOError as e:
            logger.error(f"Failed to write schedule to Markdown file {args.output_md}: {e}")

    logger.info("Active Learning Block Scheduler finished.")

if __name__ == "__main__":
    main()
