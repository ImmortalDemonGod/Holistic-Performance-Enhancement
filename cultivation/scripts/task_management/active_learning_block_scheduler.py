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

def build_task_status_map(tasks: List[Dict[str, Any]]) -> Dict[Any, str]:
    """Builds a map of task IDs to their statuses."""
    status_map: Dict[Any, str] = {}
    for task in tasks:
        task_id = task.get("id")
        if task_id is not None: # Task IDs can be numbers or strings
            status_map[task_id] = str(task.get("status", "pending"))
    return status_map

def dependencies_met(task: Dict[str, Any], task_statuses: Dict[Any, str]) -> bool:
    """Checks if all dependencies for a given task are met (status: 'done')."""
    deps = task.get("dependencies", [])
    if not deps:
        return True
    for dep_id in deps:
        if task_statuses.get(dep_id) != "done":
            logger.debug(f"Task '{task.get('id')}' (Title: {task.get('title', '')[:30]}...) dependency {dep_id} not met (status: {task_statuses.get(dep_id)}).")
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
            if min_effort_hours is None or not isinstance(min_effort_hours, (int, float)):
                logger.debug(f"Task {task_id} skipped: 'estimated_effort_hours_min' is missing or invalid ({min_effort_hours}).")
                continue
            
            if min_effort_hours * 60 > ACTIVE_BLOCK_MINUTES:
                logger.debug(f"Task {task_id} (min_effort: {min_effort_hours}hr) inherently too large for {ACTIVE_BLOCK_MINUTES}min active block. Checking subtasks for promotion.")
                # --- Subtask Promotion Logic ---
                subtasks = task.get("subtasks", [])
                pending_subtasks = [st for st in subtasks if st.get("status") == "pending" and dependencies_met(st, task_statuses)]
                num_pending = len(pending_subtasks)
                for subtask in pending_subtasks:
                    subtask_effort = None
                    # Prefer explicit subtask effort if present
                    subtask_meta = subtask.get("hpe_learning_meta", {})
                    subtask_effort = subtask_meta.get("estimated_effort_hours_min")
                    if subtask_effort is None:
                        # Fallback: divide parent min effort by number of pending subtasks
                        if num_pending > 0:
                            subtask_effort = min_effort_hours / num_pending
                        else:
                            subtask_effort = 0.5 # Default 30 min
                        logger.debug(f"Subtask {subtask.get('id')} of parent {task_id} lacks explicit effort; using fallback: {subtask_effort} hr.")
                    subtask_minutes = float(subtask_effort) * 60
                    logger.debug(f"Evaluating subtask {subtask.get('id')} (parent {task_id}): effort={subtask_effort}hr, minutes={subtask_minutes}")
                    if subtask_minutes <= ACTIVE_BLOCK_MINUTES:
                        # Augment subtask for scheduling
                        promoted = {
                            **subtask,
                            "_parent_task_id": task_id,
                            "_parent_title": task.get("title", ""),
                            "_parent_priority": task.get("priority"),
                            "_parent_hpe_csm_reference": task.get("hpe_csm_reference", {}),
                            "_parent_hpe_scheduling_meta": task.get("hpe_scheduling_meta", {}),
                            "hpe_learning_meta": {**hpe_learning_meta, **subtask_meta, "estimated_effort_hours_min": subtask_effort},
                        }
                        candidate_tasks.append(promoted)
                        logger.info(f"Promoted subtask {subtask.get('id')} (parent {task_id}) as candidate (min_effort: {subtask_effort}hr). Candidate_tasks now: {[t.get('id', t.get('_parent_task_id')) for t in candidate_tasks]}")
                    else:
                        logger.info(f"Subtask {subtask.get('id')} (parent {task_id}) too large for block (min_effort: {subtask_effort}hr), not promoted.")
                logger.info(f"Parent task {task_id} skipped for scheduling due to being oversized. No parent appended.")
                continue
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
    for t in on_day_tasks: logger.debug(f"  On-day candidate: {t.get('id')} (parent: {t.get('_parent_task_id', None)})")

    if len(on_day_tasks) >= min_tasks_for_day_focus:
        logger.info(f"Sufficient on-day active tasks ({len(on_day_tasks)}). Prioritizing these.")
        return on_day_tasks
    
    # Pass 2: If not enough on-day tasks, add other eligible active tasks
    logger.info(f"Insufficient on-day tasks ({len(on_day_tasks)} found, {min_tasks_for_day_focus} required). Considering other eligible active tasks.")
    
    # Tasks not planned for today but still eligible active tasks
    off_day_tasks = [task for task in candidate_tasks if task not in on_day_tasks]
    logger.info(f"{len(off_day_tasks)} off-day active tasks found as potential additions.")
    for t in off_day_tasks: logger.debug(f"  Off-day candidate: {t.get('id')} (parent: {t.get('_parent_task_id', None)})")

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
    
    logger.info(f"Attempting to schedule active tasks. Candidates:")
    for t in prioritized_tasks:
        logger.info(f"  Candidate: id={t.get('id')} parent={t.get('_parent_task_id', None)} effort={t.get('hpe_learning_meta',{}).get('estimated_effort_hours_min')}hr")
    logger.info(f"Total block time: {total_block_minutes} min.")

    for task in prioritized_tasks:
        task_id = task.get("id", "Unknown")
        task_title = task.get("title", "No Title")
        hpe_learning_meta = task.get("hpe_learning_meta", {})
        min_effort_hours = hpe_learning_meta.get("estimated_effort_hours_min")

        if min_effort_hours is None:
            logger.warning(f"Task {task_id} ('{task_title[:30]}...') missing 'estimated_effort_hours_min'. Skipping.")
            continue
        
        task_duration_minutes = float(min_effort_hours * 60)

        if task_duration_minutes <= 0:
            logger.debug(f"Task {task_id} ('{task_title[:30]}...') has non-positive min effort ({task_duration_minutes} min). Skipping.")
            continue
            
        if task_duration_minutes <= time_left_in_block:
            # Preserve all fields, especially subtask promotion metadata
            scheduled_task = dict(task)  # shallow copy
            scheduled_task["effort_minutes_planned"] = task_duration_minutes
            # Set 'id' to CSM ID string if present, else fallback to integer
            scheduled_task["id"] = task.get("hpe_csm_reference", {}).get("csm_id", task_id)
            scheduled_tasks_for_output.append(scheduled_task)
            time_left_in_block -= task_duration_minutes
            logger.info(f"  SCHEDULED: Task {task_id} ('{task_title[:30]}...') for {task_duration_minutes:.0f} min. Time left: {time_left_in_block:.0f} min.")
            if time_left_in_block <= 0: # Using "<= 0" as even 0 time left means block is full
                logger.info("Active learning block is now full.")
                break
        else:
            logger.info(f"  SKIPPED: Task {task_id} ('{task_title[:30]}...') (min effort: {task_duration_minutes:.0f} min) too large for remaining time ({time_left_in_block:.0f} min).")
            
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
        lines.append("No tasks scheduled for this block.")
    else:
        total_planned_time = sum(task.get('effort_minutes_planned', 0) for task in scheduled_tasks)
        lines.append(f"Total Planned Time: {total_planned_time:.0f} min / {ACTIVE_BLOCK_MINUTES} min")
        lines.append(separator)
        for i, task_detail in enumerate(scheduled_tasks, 1):
            lines.append(f"{i}. Task ID: [{task_detail.get('id', 'N/A')}]")
            lines.append(f"   - Title: {task_detail.get('title', 'No Title')}")
            lines.append(f"   - Activity Type: {task_detail.get('activity_type', 'N/A')}")
            lines.append(f"   - Est. Effort (Raw): {task_detail.get('estimated_effort_hours_raw', 'N/A')}")
            lines.append(f"   - Planned Duration: {task_detail.get('effort_minutes_planned', 0):.0f} min")
            task_notes = task_detail.get('notes')
            if task_notes:
                 lines.append(f"   - Notes: {str(task_notes)[:100]}...") # Truncate long notes
            lines.append("") 

    return "\n".join(lines)

# --- Main Execution Logic ---

def main():
    """Main function to orchestrate the scheduling process."""
    parser = argparse.ArgumentParser(description="Schedule Active Learning Block Tasks.")
    parser.add_argument(
        "--tasks", 
        type=str, 
        default="tasks/tasks.json", 
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
        help="Optional path to write the schedule to a Markdown file."
    )
    parser.add_argument(
        "--min-focus-tasks",
        type=int,
        default=MIN_REQUIRED_TASKS_FOR_DAY_FOCUS,
        help="Minimum number of on-day tasks required before pulling from other days. Default: %(default)s"
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true', help='Enable debug logging'
    )
    
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logging.root.handlers: # Apply to all existing handlers
            handler.setLevel(logging.DEBUG)


    logger.info(f"Running Active Learning Block Scheduler for date: {args.date or 'today'}")
    
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
