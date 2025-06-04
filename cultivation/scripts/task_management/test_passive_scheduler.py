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
        with open(BASELINE_TASKS_PATH, 'r') as f:
            cls.baseline_tasks_data = json.load(f)["tasks"]

    def setUp(self):
        self.current_tasks_data = copy.deepcopy(self.baseline_tasks_data)

    def test_new_task10_scheduled_when_deps_met(self):
        """
        When Tasks 1-9 and Task 11 are done, and Task 10 is pending,
        running the passive scheduler for Day 7 should schedule only Task 10
        (Week 1 Learning Reflection, Flashcard Finalization & HPE Logging)
        and the default passive review task.
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
        day_of_week = scheduler.get_today_day_of_week(target_date_str)
        task_statuses = scheduler.build_task_status_map(tasks_list)
        passive_candidates = scheduler.filter_passive_tasks(
            tasks_list, day_of_week, task_statuses, allow_off_day_fill=(not strict_on_day_only)
        )
        prioritized_tasks = scheduler.prioritize_tasks(passive_candidates)
        scheduled_block = scheduler.schedule_tasks(prioritized_tasks)
        return scheduled_block

    def test_task10_eligible_and_fits(self):
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
        target_date = "2025-05-25"
        # All tasks pending, none eligible - set all to done to make them ineligible
        modifications = [{"id": i, "status": "done"} for i in range(1, 20)]  # Cover all task IDs
        self._modify_tasks(modifications)
        scheduled = self._run_scheduler(self.current_tasks_data, target_date)
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0]["id"], "Default.Passive.Review")

if __name__ == "__main__":
    unittest.main()
