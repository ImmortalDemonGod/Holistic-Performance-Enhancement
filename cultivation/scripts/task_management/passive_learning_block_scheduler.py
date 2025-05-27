# cultivation/scripts/task_management/passive_learning_block_scheduler.py
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

PASSIVE_BLOCK_MINUTES = 90
DEFAULT_TASK = {
    "id": "Default.Passive.Review",
    "title": "General review of recent learning notes",
    "activity": "note_review_general",
    "effort": "15-30 min",
    "effort_minutes_planned": 20,
    "notes": "Consolidate recent learning through general review."
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PassiveLearningBlockScheduler")

def load_tasks(tasks_json_path: str) -> List[Dict[str, Any]]:
    """
    Loads a list of tasks from a JSON file at the specified path.
    
    If the file cannot be read or parsed, returns an empty list.
    """
    try:
        with open(tasks_json_path, 'r') as f:
            return json.load(f)["tasks"]
    except Exception as e:
        logger.error(f"Error loading tasks.json: {e}")
        return []

def get_today_day_of_week(target_date: str = None) -> int:
    """
    Returns the ISO weekday number for a given date or today.
    
    Args:
        target_date: Optional date string in 'YYYY-MM-DD' format. If not provided, uses the current date.
    
    Returns:
        An integer representing the ISO weekday (Monday=1, Sunday=7).
    """
    if target_date:
        dt = datetime.strptime(target_date, "%Y-%m-%d")
    else:
        dt = datetime.now()
    return dt.isoweekday()  # Monday=1, Sunday=7

def build_task_status_map(tasks: List[Dict[str, Any]]) -> Dict[Any, str]:
    """
    Creates a mapping of task IDs to their status from a list of task dictionaries.
    
    Args:
        tasks: List of task dictionaries, each containing at least "id" and "status" keys.
    
    Returns:
        A dictionary mapping each task's ID to its status.
    """
    return {task.get("id"): task.get("status") for task in tasks}

def dependencies_met(task: Dict[str, Any], task_statuses: Dict[Any, str]) -> bool:
    """
    Checks whether all dependencies of a task are marked as completed.
    
    Args:
        task: The task dictionary, potentially containing a "dependencies" list.
        task_statuses: A mapping of task IDs to their current status.
    
    Returns:
        True if all dependencies are marked "done" in the status map, False otherwise.
    """
    deps = task.get("dependencies", [])
    for dep_id in deps:
        if task_statuses.get(dep_id) != "done":
            logger.debug(f"Task {task.get('id')} dependency {dep_id} not done.")
            return False
    return True

def filter_passive_tasks(tasks: List[Dict[str, Any]], day_of_week: int, task_statuses: Dict[Any, str], min_required: int = 2, allow_off_day_fill: bool = True) -> List[Dict[str, Any]]:
    """
    Filters tasks to select those suitable for passive learning on a given day.
    
    Selects pending tasks with all dependencies met, recommended for passive review or matching passive activity keywords, and planned for the specified day of the week. If fewer than the minimum required tasks are found and off-day fill is allowed, a second pass adds additional suitable passive tasks regardless of planned day.
    
    Args:
        tasks: List of task dictionaries to filter.
        day_of_week: ISO weekday number (1=Monday, 7=Sunday) for which to filter tasks.
        task_statuses: Mapping of task IDs to their current status.
        min_required: Minimum number of tasks to select before considering off-day fill.
        allow_off_day_fill: Whether to include suitable tasks not planned for the target day if needed.
    
    Returns:
        A list of filtered tasks suitable for passive learning on the specified day.
    """
    keywords = ["flashcard_review", "note_review", "summary_writing", "consolidation", "light_reading", "audio_learning"]
    filtered = []
    logger.info(f"[filter_passive_tasks] First pass: strict planned_day_of_week={day_of_week}")
    for task in tasks:
        if task.get("status") != "pending":
            continue
        if not dependencies_met(task, task_statuses):
            continue
        meta = task.get("hpe_learning_meta", {})
        block = meta.get("recommended_block", "")
        activity = meta.get("activity_type", "")
        planned_day = task.get("hpe_scheduling_meta", {}).get("planned_day_of_week")
        if (block == "passive_review" or any(k in activity for k in keywords)):
            if planned_day == day_of_week:
                logger.info(f"  [PASS1] Adding task {task.get('id')} (planned_day_of_week={planned_day})")
                filtered.append(task)
    logger.info(f"[filter_passive_tasks] First pass found {len(filtered)} tasks.")
    if len(filtered) < min_required and allow_off_day_fill:
        logger.info(f"[filter_passive_tasks] Second pass: add all suitable passive tasks not already included.")
        for task in tasks:
            if task.get("status") != "pending":
                continue
            if not dependencies_met(task, task_statuses):
                continue
            meta = task.get("hpe_learning_meta", {})
            block = meta.get("recommended_block", "")
            activity = meta.get("activity_type", "")
            planned_day = task.get("hpe_scheduling_meta", {}).get("planned_day_of_week")
            if (block == "passive_review" or any(k in activity for k in keywords)):
                if task not in filtered:
                    logger.info(f"  [PASS2] Adding task {task.get('id')} (planned_day_of_week={planned_day})")
                    filtered.append(task)
    elif len(filtered) < min_required and not allow_off_day_fill:
        logger.info(f"[filter_passive_tasks] Second pass skipped (allow_off_day_fill=False). Only on-day tasks considered.")
    logger.info(f"[filter_passive_tasks] Final filtered tasks: {[task.get('id') for task in filtered]}")
    return filtered

def prioritize_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sorts tasks by priority for passive learning scheduling.
    
    Tasks are ranked by tier based on tags or activity type, explicit priority label, and estimated minimum effort hours (descending). Returns the sorted list with highest-priority tasks first.
    """
    TIER_SCORES = {1: 400, 2: 300, 3: 200, 4: 100, "default_priority": 0}
    keywords_tiers = {
        1: ["consolidation", "review"],  # tags
        2: ["flashcard_review"],
        3: ["note_review", "summary_writing"],
        4: ["audio_learning", "light_reading"]
    }
    def get_task_tier(task):
        """
        Determines the priority tier score for a task based on its tags and activity type.
        
        Returns the corresponding tier score if any tier keywords are found in the task's tags or activity type; otherwise, returns the default priority score.
        """
        tags = task.get("hpe_scheduling_meta", {}).get("csm_tags", [])
        activity = task.get("hpe_learning_meta", {}).get("activity_type", "")
        for tier, keywords in keywords_tiers.items():
            if any(k in tags for k in keywords) or any(k in activity for k in keywords):
                return TIER_SCORES[tier]
        return TIER_SCORES["default_priority"]
    def priority_value(p):
        """
        Returns a numeric score corresponding to a priority label.
        
        The function maps 'high', 'medium', and 'low' (case-insensitive) to 30, 20, and 10 respectively. Any unrecognized label defaults to 20.
        """
        return {"high": 30, "medium": 20, "low": 10}.get(p.lower(), 20)
    tasks.sort(key=lambda t: (
        get_task_tier(t),
        priority_value(t.get("priority", "medium")),
        -t.get("hpe_learning_meta", {}).get("estimated_effort_hours_min", float('inf'))
    ), reverse=True)
    return tasks

def schedule_tasks(tasks: List[Dict[str, Any]], total_minutes: int = PASSIVE_BLOCK_MINUTES) -> List[Dict[str, Any]]:
    """
    Schedules tasks into a fixed-duration passive learning block based on estimated effort.
    
    Iterates through prioritized tasks, adding each to the schedule if its average estimated effort fits within the remaining block time. If significant time remains after scheduling all tasks, a default passive review task is added to fill the block.
    
    Args:
        tasks: List of task dictionaries to consider for scheduling.
        total_minutes: Total available minutes for the passive learning block.
    
    Returns:
        A list of scheduled task dictionaries, each including planned effort in minutes.
    """
    scheduled = []
    time_left = total_minutes
    logger.info(f"[schedule_tasks] Starting block fill: {len(tasks)} tasks to consider, total_minutes={total_minutes}")
    for task in tasks:
        meta = task.get("hpe_learning_meta", {})
        min_eff_hours = meta.get("estimated_effort_hours_min", 0)
        max_eff_hours = meta.get("estimated_effort_hours_max", min_eff_hours + 0.5)  # Default to min + 30 min
        min_eff_minutes = min_eff_hours * 60
        max_eff_minutes = max_eff_hours * 60
        avg_eff_minutes = (min_eff_minutes + max_eff_minutes) / 2
        logger.info(f"  Considering task {task.get('id')} (min={min_eff_minutes}, max={max_eff_minutes}, avg={avg_eff_minutes}, time_left={time_left})")
        if avg_eff_minutes <= time_left and avg_eff_minutes > 0:
            logger.info(f"    [SCHEDULED] Task {task.get('id')} for {avg_eff_minutes} min")
            scheduled.append({
                "id": task.get("hpe_csm_reference", {}).get("csm_id", task.get("id")),
                "title": task.get("title"),
                "activity": meta.get("activity_type"),
                "effort": meta.get("estimated_effort_hours_raw"),
                "effort_minutes_planned": avg_eff_minutes,
                "notes": task.get("details", "")
            })
            time_left -= avg_eff_minutes
        else:
            logger.info(f"    [SKIPPED] Task {task.get('id')} (avg_eff_minutes={avg_eff_minutes}) does not fit in time_left={time_left}")
        if time_left <= 0:
            logger.info(f"[schedule_tasks] Block filled (time_left={time_left})")
            break
    # Add default task if >15 min left and it fits
    if time_left > 15 and time_left >= DEFAULT_TASK["effort_minutes_planned"]:
        logger.info(f"  [DEFAULT] Adding default passive review task for {min(time_left, DEFAULT_TASK['effort_minutes_planned'])} min")
        default_task = DEFAULT_TASK.copy()
        default_task["effort_minutes_planned"] = min(time_left, DEFAULT_TASK["effort_minutes_planned"])
        scheduled.append(default_task)
    logger.info(f"[schedule_tasks] Final scheduled tasks: {[t['id'] for t in scheduled]}")
    return scheduled

def print_schedule(scheduled: List[Dict[str, Any]], output_md: str = None):
    """
    Prints a formatted summary of the scheduled passive learning block.
    
    If an output file path is provided, also writes the schedule to a markdown file.
    """
    lines = []
    header = "Learning Block (Passive Review & Consolidation) | 23:15 – 00:45 CT (90 minutes)"
    sep = "-" * 80
    lines.append(header)
    lines.append(sep)
    for i, task in enumerate(scheduled, 1):
        lines.append(f"{i}. Task ID: [{task['id']}]")
        lines.append(f"   - Title: {task['title']}")
        lines.append(f"   - Activity: {task['activity']}")
        lines.append(f"   - Est. Effort: {task['effort']} ({task['effort_minutes_planned']:.0f} min planned)")
        lines.append(f"   - Notes: {task['notes']}")
        lines.append("")
    output = "\n".join(lines)
    print(output)
    if output_md:
        try:
            with open(output_md, 'w') as f:
                f.write(output)
            logger.info(f"Markdown output written to {output_md}")
        except Exception as e:
            logger.error(f"Failed to write markdown output: {e}")

def generate_passive_plan(
    all_tasks_data: List[Dict[str, Any]],
    target_date_str: str,
    min_required: int = 2
) -> List[Dict[str, Any]]:
    """
    Generates a scheduled plan of passive learning tasks for a specified date.
    
    Determines the appropriate day of week, filters and prioritizes passive learning tasks based on readiness and scheduling constraints, and fits them into a fixed time block. Returns the list of scheduled tasks for the target date.
     
    Args:
        all_tasks_data: List of all available task dictionaries.
        target_date_str: Target date in 'YYYY-MM-DD' format.
        min_required: Minimum number of passive tasks to schedule (default is 2).
    
    Returns:
        A list of scheduled task dictionaries for the passive learning block.
    """
    day_of_week = get_today_day_of_week(target_date_str)
    task_statuses = build_task_status_map(all_tasks_data)
    passive_tasks = filter_passive_tasks(all_tasks_data, day_of_week, task_statuses, min_required)
    prioritized = prioritize_tasks(passive_tasks)
    scheduled = schedule_tasks(prioritized)
    return scheduled

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Schedule Passive Learning Block Tasks")
    parser.add_argument("--tasks", type=str, default="/Users/tomriddle1/Holistic-Performance-Enhancement/tasks/tasks.json", help="Path to enriched tasks.json")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD), default: today")
    parser.add_argument("--output-md", type=str, default=None, help="If set, write output to markdown file")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    scheduled = generate_passive_plan(tasks, args.date)
    print_schedule(scheduled, args.output_md)
