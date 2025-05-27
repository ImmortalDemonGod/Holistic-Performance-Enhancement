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
    try:
        with open(tasks_json_path, 'r') as f:
            return json.load(f)["tasks"]
    except Exception as e:
        logger.error(f"Error loading tasks.json: {e}")
        return []

def get_today_day_of_week(target_date: str = None) -> int:
    if target_date:
        dt = datetime.strptime(target_date, "%Y-%m-%d")
    else:
        dt = datetime.now()
    return dt.isoweekday()  # Monday=1, Sunday=7

def build_task_status_map(tasks: List[Dict[str, Any]]) -> Dict[Any, str]:
    return {task.get("id"): task.get("status") for task in tasks}

def dependencies_met(task: Dict[str, Any], task_statuses: Dict[Any, str]) -> bool:
    deps = task.get("dependencies", [])
    for dep_id in deps:
        if task_statuses.get(dep_id) != "done":
            logger.debug(f"Task {task.get('id')} dependency {dep_id} not done.")
            return False
    return True

def filter_passive_tasks(tasks: List[Dict[str, Any]], day_of_week: int, task_statuses: Dict[Any, str], min_required: int = 2, allow_off_day_fill: bool = True) -> List[Dict[str, Any]]:
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
    TIER_SCORES = {1: 400, 2: 300, 3: 200, 4: 100, "default_priority": 0}
    keywords_tiers = {
        1: ["consolidation", "review"],  # tags
        2: ["flashcard_review"],
        3: ["note_review", "summary_writing"],
        4: ["audio_learning", "light_reading"]
    }
    def get_task_tier(task):
        tags = task.get("hpe_scheduling_meta", {}).get("csm_tags", [])
        activity = task.get("hpe_learning_meta", {}).get("activity_type", "")
        for tier, keywords in keywords_tiers.items():
            if any(k in tags for k in keywords) or any(k in activity for k in keywords):
                return TIER_SCORES[tier]
        return TIER_SCORES["default_priority"]
    def priority_value(p):
        return {"high": 30, "medium": 20, "low": 10}.get(p.lower(), 20)
    tasks.sort(key=lambda t: (
        get_task_tier(t),
        priority_value(t.get("priority", "medium")),
        -t.get("hpe_learning_meta", {}).get("estimated_effort_hours_min", float('inf'))
    ), reverse=True)
    return tasks

def schedule_tasks(tasks: List[Dict[str, Any]], total_minutes: int = PASSIVE_BLOCK_MINUTES) -> List[Dict[str, Any]]:
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
    Generates the passive learning block plan for a given date and task list.
    Returns a list of scheduled task dicts (not printed output).
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
