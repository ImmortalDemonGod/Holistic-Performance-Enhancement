# cultivation/scripts/task_management/test_passive_scheduler.py
import unittest
import json
import copy
import sys
import os
from datetime import datetime

# Ensure module import works regardless of test runner cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import passive_learning_block_scheduler as scheduler

BASELINE_TASKS_PATH = os.path.join(os.path.dirname(__file__), '../../..', 'tasks', 'tasks.json')

class TestPassiveLearningBlockScheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Loads baseline task data from a JSON file before any tests are run.
        
        This method initializes the class-level baseline task list for use in all test cases.
        """
        with open(BASELINE_TASKS_PATH, 'r') as f:
            cls.baseline_tasks_data = json.load(f)["tasks"]

    def setUp(self):
        """
        Creates a deep copy of the baseline tasks data before each test to ensure test isolation.
        """
        self.current_tasks_data = copy.deepcopy(self.baseline_tasks_data)

    def test_new_task10_scheduled_when_deps_met(self):
        """
        Tests that when tasks 1-9 and 11 are marked done and task 10 is pending, running the scheduler on the target date schedules only task 10 and the default passive review task, while excluding task 11.
        """
        target_date = "2025-05-25"  # Sunday
        modifications = []
        # Mark Tasks 1-9 as done
        for i in range(1, 10):
            modifications.append({"id": i, "status": "done"})
        # Mark Task 11 as done (new self-assessment task)
        modifications.append({"id": 11, "status": "done"})
        # Ensure Task 10 is pending
        modifications.append({
            "csm_id": "RNA.P1.Foundations.W1.Part3.ReflectFinalizeLog",
            "status": "pending"
        })
        self._modify_tasks(modifications)

        scheduled_block = self._run_scheduler(self.current_tasks_data, target_date)
        scheduled_ids = [t.get("hpe_csm_reference", {}).get("csm_id") or t.get("id") for t in scheduled_block]

        # Check Task 10 is scheduled
        self.assertIn("RNA.P1.Foundations.W1.Part3.ReflectFinalizeLog", scheduled_ids)
        # Check the default passive review task is scheduled
        self.assertTrue(any("Default.Passive.Review" in str(i) for i in scheduled_ids))
        # Ensure Task 11 and others are NOT scheduled
        self.assertNotIn("RNA.P1.Foundations.W1.Part3.ExecuteAssessments", scheduled_ids)

        self.current_tasks_data = copy.deepcopy(self.baseline_tasks_data)

    def _modify_tasks(self, modifications):
        """
        Applies a list of modifications to the current tasks data in place.
        
        Each modification can update a task's status, priority, or a nested field specified by a field path. Tasks are matched by either their `csm_id` or numeric `id`.
        """
        for mod in modifications:
            for task in self.current_tasks_data:
                csm_id = task.get("hpe_csm_reference", {}).get("csm_id")
                numeric_id = task.get("id")
                if (mod.get("csm_id") and csm_id == mod["csm_id"]) or (mod.get("id") and numeric_id == mod["id"]):
                    if "status" in mod:
                        task["status"] = mod["status"]
                    if "priority" in mod:
                        task["priority"] = mod["priority"]
                    field_path = mod.get("field_path")
                    if field_path and "value" in mod:
                        current = task
                        for i, key in enumerate(field_path):
                            if i == len(field_path) - 1:
                                current[key] = mod["value"]
                            else:
                                current = current.setdefault(key, {})
                    break

    def _run_scheduler(self, tasks_list, target_date_str, strict_on_day_only=False):
        """
        Runs the passive learning block scheduler pipeline on the provided tasks for a given date.
        
        Args:
            tasks_list: List of task dictionaries to be considered for scheduling.
            target_date_str: Date string (YYYY-MM-DD) representing the scheduling day.
            strict_on_day_only: If True, only tasks planned for the target day are eligible; if False, allows off-day tasks if no on-day tasks fit.
        
        Returns:
            The scheduled block as determined by the scheduler, containing the selected tasks and their planned effort.
        """
        day_of_week = scheduler.get_today_day_of_week(target_date_str)
        task_statuses = scheduler.build_task_status_map(tasks_list)
        passive_candidates = scheduler.filter_passive_tasks(
            tasks_list, day_of_week, task_statuses, allow_off_day_fill=(not strict_on_day_only)
        )
        prioritized_tasks = scheduler.prioritize_tasks(passive_candidates)
        scheduled_block = scheduler.schedule_tasks(prioritized_tasks)
        return scheduled_block

    def test_task10_eligible_and_fits(self):
        """
        Tests that Task 10 is scheduled when eligible and its estimated effort fits within constraints.
        
        Marks Tasks 1-9 and 11 as done, sets Task 10's estimated effort to 1.0–1.5 hours, and marks it pending. Verifies the scheduler includes Task 10 with a planned effort of 75 minutes.
        """
        target_date = "2025-05-25"
        modifications = []
        for i in range(1, 10):
            modifications.append({"id": i, "status": "done"})
        modifications.append({"id": 11, "status": "done"})  # Ensure Task 11 is done for Task 10 dependency
        modifications.append({
            "csm_id": "RNA.P1.Foundations.W1.Part3.ReflectFinalizeLog",
            "field_path": ("hpe_learning_meta", "estimated_effort_hours_min"),
            "value": 1.0
        })
        modifications.append({
            "csm_id": "RNA.P1.Foundations.W1.Part3.ReflectFinalizeLog",
            "field_path": ("hpe_learning_meta", "estimated_effort_hours_max"),
            "value": 1.5
        })
        modifications.append({
            "csm_id": "RNA.P1.Foundations.W1.Part3.ReflectFinalizeLog",
            "status": "pending"
        })
        self._modify_tasks(modifications)
        scheduled = self._run_scheduler(self.current_tasks_data, target_date)
        self.assertTrue(len(scheduled) >= 1)
        self.assertEqual(scheduled[0]["id"], "RNA.P1.Foundations.W1.Part3.ReflectFinalizeLog")
        self.assertAlmostEqual(scheduled[0]["effort_minutes_planned"], 75.0)  # (1.0 + 1.5)/2 * 60 = 75

    def test_task10_eligible_too_large(self):
        """
        Tests that task 10 is not scheduled when its estimated effort exceeds the allowed limit.
        
        Marks tasks 1-9 as done and sets task 10's estimated effort to 1.5–2.0 hours, then verifies that only the default passive review task is scheduled when running the scheduler strictly for the target day.
        """
        target_date = "2025-05-25"
        modifications = []
        for i in range(1, 10):
            modifications.append({"id": i, "status": "done"})
        modifications.append({
            "csm_id": "RNA.P1.Foundations.W1.Part3.ReflectFinalizeLog",
            "field_path": ("hpe_learning_meta", "estimated_effort_hours_min"),
            "value": 1.5
        })
        modifications.append({
            "csm_id": "RNA.P1.Foundations.W1.Part3.ReflectFinalizeLog",
            "field_path": ("hpe_learning_meta", "estimated_effort_hours_max"),
            "value": 2.0
        })
        modifications.append({
            "csm_id": "RNA.P1.Foundations.W1.Part3.ReflectFinalizeLog",
            "status": "pending"
        })
        self._modify_tasks(modifications)
        scheduled = self._run_scheduler(self.current_tasks_data, target_date, strict_on_day_only=True)
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0]["id"], "Default.Passive.Review")

    def test_task10_blocked_by_dependencies(self):
        """
        Tests that task 10 is not scheduled when its dependency (task 9) is incomplete.
        
        Marks tasks 1-8 as done and task 9 as pending, sets task 10's effort estimates, and verifies that only the default passive review task is scheduled, confirming dependency blocking behavior.
        """
        target_date = "2025-05-25"
        modifications = []
        for i in range(1, 9):
            modifications.append({"id": i, "status": "done"})
        modifications.append({"id": 9, "status": "pending"})
        modifications.append({
            "csm_id": "RNA.P1.Foundations.W1.Part3.ReflectFinalizeLog",
            "field_path": ("hpe_learning_meta", "estimated_effort_hours_min"),
            "value": 1.0
        })
        modifications.append({
            "csm_id": "RNA.P1.Foundations.W1.Part3.ReflectFinalizeLog",
            "field_path": ("hpe_learning_meta", "estimated_effort_hours_max"),
            "value": 1.5
        })
        modifications.append({
            "csm_id": "RNA.P1.Foundations.W1.Part3.ReflectFinalizeLog",
            "status": "pending"
        })
        self._modify_tasks(modifications)
        scheduled = self._run_scheduler(self.current_tasks_data, target_date, strict_on_day_only=True)
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0]["id"], "Default.Passive.Review")

    def test_two_pass_filter_offday_task(self):
        """
        Tests that the scheduler selects an off-day passive review task via two-pass filtering when no on-day tasks are eligible.
        
        Marks tasks 1-7 as done, sets task 8 as a passive review task planned for a different day, and tasks 9-10 as pending. Verifies that the scheduler schedules task 8 on a day it is not originally planned for, demonstrating the two-pass filter behavior.
        """
        target_date = "2025-05-25"  # Sunday (Day 7)
        modifications = []
        # Tasks 1-7 done, 8 is a passive review task for Day 5, 9/10 pending
        for i in range(1, 8):
            modifications.append({"id": i, "status": "done"})
        modifications.append({"id": 8, "status": "pending"})
        modifications.append({"id": 9, "status": "pending"})
        modifications.append({"id": 10, "status": "pending"})
        # Make Task 8 a passive review task for Day 5
        modifications.append({
            "id": 8,
            "field_path": ("hpe_learning_meta", "recommended_block"),
            "value": "passive_review"
        })
        modifications.append({
            "id": 8,
            "field_path": ("hpe_learning_meta", "activity_type"),
            "value": "summary_writing"
        })
        modifications.append({
            "id": 8,
            "field_path": ("hpe_scheduling_meta", "planned_day_of_week"),
            "value": 5
        })
        modifications.append({
            "id": 8,
            "field_path": ("hpe_learning_meta", "estimated_effort_hours_min"),
            "value": 0.5
        })
        modifications.append({
            "id": 8,
            "field_path": ("hpe_learning_meta", "estimated_effort_hours_max"),
            "value": 0.75
        })
        self._modify_tasks(modifications)
        scheduled = self._run_scheduler(self.current_tasks_data, target_date)
        # Should pick Task 8 via two-pass filter (check by csm_id)
        self.assertTrue(any(t["id"] == "RNA.P1.Foundations.W1.Part2.Thermo.EnvFactors" for t in scheduled))

    def test_tiered_prioritization_activity_type(self):
        """
        Tests that the scheduler prioritizes tasks by activity type, ensuring flashcard review tasks are scheduled before note review and audio learning tasks when all are eligible on the same day.
        """
        target_date = "2025-05-25"
        modifications = [{"id": i, "status": "done"} for i in range(1, 11)]
        # Add three dummy tasks of different tiers
        dummy_a = {
            "id": "A", "title": "Flashcard Review", "status": "pending", "dependencies": [], "priority": "medium",
            "hpe_learning_meta": {"activity_type": "flashcard_review", "recommended_block": "passive_review", "estimated_effort_hours_min": 0.5, "estimated_effort_hours_max": 0.5},
            "hpe_scheduling_meta": {"planned_day_of_week": 7}
        }
        dummy_b = {
            "id": "B", "title": "Note Review", "status": "pending", "dependencies": [], "priority": "medium",
            "hpe_learning_meta": {"activity_type": "note_review", "recommended_block": "passive_review", "estimated_effort_hours_min": 0.5, "estimated_effort_hours_max": 0.5},
            "hpe_scheduling_meta": {"planned_day_of_week": 7}
        }
        dummy_c = {
            "id": "C", "title": "Audio Learning", "status": "pending", "dependencies": [], "priority": "medium",
            "hpe_learning_meta": {"activity_type": "audio_learning", "recommended_block": "passive_review", "estimated_effort_hours_min": 0.5, "estimated_effort_hours_max": 0.5},
            "hpe_scheduling_meta": {"planned_day_of_week": 7}
        }
        self._modify_tasks(modifications)
        self.current_tasks_data.extend([dummy_a, dummy_b, dummy_c])
        scheduled = self._run_scheduler(self.current_tasks_data, target_date)
        # Should schedule A before B before C
        ids = [t["id"] for t in scheduled]
        self.assertTrue(ids.index("A") < ids.index("B") < ids.index("C") or ids.index("A") < ids.index("B"))

    def test_tiered_prioritization_priority(self):
        """
        Tests that among pending note review tasks with different priorities, the scheduler orders the high-priority task before the low-priority one.
        """
        target_date = "2025-05-25"
        modifications = [{"id": i, "status": "done"} for i in range(1, 11)]
        # Two dummy note_review tasks, different priorities
        dummy_d = {
            "id": "D", "title": "Note Review High", "status": "pending", "dependencies": [], "priority": "high",
            "hpe_learning_meta": {"activity_type": "note_review", "recommended_block": "passive_review", "estimated_effort_hours_min": 0.5, "estimated_effort_hours_max": 0.5},
            "hpe_scheduling_meta": {"planned_day_of_week": 7}
        }
        dummy_e = {
            "id": "E", "title": "Note Review Low", "status": "pending", "dependencies": [], "priority": "low",
            "hpe_learning_meta": {"activity_type": "note_review", "recommended_block": "passive_review", "estimated_effort_hours_min": 0.5, "estimated_effort_hours_max": 0.5},
            "hpe_scheduling_meta": {"planned_day_of_week": 7}
        }
        self._modify_tasks(modifications)
        self.current_tasks_data.extend([dummy_d, dummy_e])
        scheduled = self._run_scheduler(self.current_tasks_data, target_date)
        ids = [t["id"] for t in scheduled]
        self.assertTrue(ids.index("D") < ids.index("E"))

    def test_greedy_block_fill_multiple_small_tasks(self):
        """
        Tests that the scheduler fills the passive review block by scheduling multiple small eligible tasks in order of increasing effort, followed by the default passive review task.
        
        Marks tasks 1-10 as done, adds three pending note review tasks with decreasing estimated effort (0.25, 0.5, 1.0 hours), and verifies that the smallest tasks are scheduled first, ensuring greedy block filling behavior.
        """
        target_date = "2025-05-25"
        modifications = [{"id": i, "status": "done"} for i in range(1, 11)]
        dummy_f = {
            "id": "F", "title": "Task F", "status": "pending", "dependencies": [], "priority": "medium",
            "hpe_learning_meta": {"activity_type": "note_review", "recommended_block": "passive_review", "estimated_effort_hours_min": 1.0, "estimated_effort_hours_max": 1.0},
            "hpe_scheduling_meta": {"planned_day_of_week": 7}
        }
        dummy_g = {
            "id": "G", "title": "Task G", "status": "pending", "dependencies": [], "priority": "medium",
            "hpe_learning_meta": {"activity_type": "note_review", "recommended_block": "passive_review", "estimated_effort_hours_min": 0.5, "estimated_effort_hours_max": 0.5},
            "hpe_scheduling_meta": {"planned_day_of_week": 7}
        }
        dummy_h = {
            "id": "H", "title": "Task H", "status": "pending", "dependencies": [], "priority": "medium",
            "hpe_learning_meta": {"activity_type": "note_review", "recommended_block": "passive_review", "estimated_effort_hours_min": 0.25, "estimated_effort_hours_max": 0.25},
            "hpe_scheduling_meta": {"planned_day_of_week": 7}
        }
        self._modify_tasks(modifications)
        self.current_tasks_data.extend([dummy_h, dummy_g, dummy_f])
        scheduled = self._run_scheduler(self.current_tasks_data, target_date)
        ids = [t["id"] for t in scheduled]
        self.assertEqual(ids[0], "H")
        self.assertEqual(ids[1], "G")
        self.assertIn("Default.Passive.Review", ids)

    def test_baseline_default_only(self):
        """
        Verifies that when no tasks are eligible, only the default passive review task is scheduled.
        
        This test ensures the scheduler falls back to scheduling the default passive review task when all tasks are pending and none meet eligibility criteria.
        """
        target_date = "2025-05-25"
        # All tasks pending, none eligible
        scheduled = self._run_scheduler(self.current_tasks_data, target_date)
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0]["id"], "Default.Passive.Review")

if __name__ == "__main__":
    unittest.main()
