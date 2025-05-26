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

def filter_passive_tasks(tasks: List[Dict[str, Any]], day_of_week: int, task_statuses: Dict[Any, str], min_required: int = 2) -> List[Dict[str, Any]]:
    keywords = ["flashcard_review", "note_review", "summary_writing", "consolidation", "light_reading", "audio_learning"]
    # First pass: strict planned_day_of_week
    filtered = []
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
                filtered.append(task)
    # If not enough, add from all suitable passive tasks
    if len(filtered) < min_required:
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
                    filtered.append(task)
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
    for task in tasks:
        meta = task.get("hpe_learning_meta", {})
        min_eff_hours = meta.get("estimated_effort_hours_min", 0)
        max_eff_hours = meta.get("estimated_effort_hours_max", min_eff_hours + 0.5)  # Default to min + 30 min
        min_eff_minutes = min_eff_hours * 60
        max_eff_minutes = max_eff_hours * 60
        avg_eff_minutes = (min_eff_minutes + max_eff_minutes) / 2
        if avg_eff_minutes <= time_left and avg_eff_minutes > 0:
            scheduled.append({
                "id": task.get("hpe_csm_reference", {}).get("csm_id", task.get("id")),
                "title": task.get("title"),
                "activity": meta.get("activity_type"),
                "effort": meta.get("estimated_effort_hours_raw"),
                "effort_minutes_planned": avg_eff_minutes,
                "notes": task.get("details", "")
            })
            time_left -= avg_eff_minutes
        if time_left <= 0:
            break
    # Add default task if >15 min left and it fits
    if time_left > 15 and time_left >= DEFAULT_TASK["effort_minutes_planned"]:
        default_task = DEFAULT_TASK.copy()
        default_task["effort_minutes_planned"] = min(time_left, DEFAULT_TASK["effort_minutes_planned"])
        scheduled.append(default_task)
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Schedule Passive Learning Block Tasks")
    parser.add_argument("--tasks", type=str, default="tasks.json", help="Path to enriched tasks.json")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD), default: today")
    parser.add_argument("--output-md", type=str, default=None, help="If set, write output to markdown file")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    day_of_week = get_today_day_of_week(args.date)
    task_statuses = build_task_status_map(tasks)
    passive_tasks = filter_passive_tasks(tasks, day_of_week, task_statuses)
    prioritized = prioritize_tasks(passive_tasks)
    scheduled = schedule_tasks(prioritized)
    print_schedule(scheduled, args.output_md)
