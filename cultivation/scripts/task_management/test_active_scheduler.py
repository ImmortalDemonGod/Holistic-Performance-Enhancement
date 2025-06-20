# cultivation/scripts/task_management/test_active_scheduler.py
import unittest
import json
import copy
import sys
import os
import logging
from typing import List, Dict, Any

from pathlib import Path

# Ensure module import works regardless of test runner cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import active_learning_block_scheduler as scheduler

# Suppress logging for cleaner test output
scheduler.logger.setLevel(logging.CRITICAL)

def get_project_root() -> Path:
    """Traverses up to find the project root, marked by '.git' directory."""
    current_path = Path(__file__).resolve()
    while current_path != current_path.parent:
        if (current_path / '.git').exists():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError("Project root with .git folder not found.")

PROJECT_ROOT = get_project_root()
TEST_DATA_PATH = PROJECT_ROOT / 'tests' / 'test_data' / 'tasks_for_scheduler.json'

class TestActiveLearningBlockScheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load tasks from a baseline JSON file once for all tests."""
        if not TEST_DATA_PATH.is_file():
            raise FileNotFoundError(
                f"FATAL: Test data file 'tasks_for_scheduler.json' not found. "
                f"Attempted path: {TEST_DATA_PATH}. "
                f"Ensure you are running tests from the project root and the file exists."
            )
        with open(TEST_DATA_PATH, 'r') as f:
            cls.baseline_tasks = json.load(f)

    def setUp(self):
        self.current_tasks_data = copy.deepcopy(self.baseline_tasks)
        scheduler.logger.setLevel(logging.INFO)
        for handler in logging.root.handlers:
            handler.setLevel(logging.INFO)

    def _modify_tasks(self, modifications: List[Dict[str, Any]]):

        def recursive_modify(tasks_list: List[Dict[str, Any]], mod: Dict[str, Any]) -> bool:
            found_in_level = False
            for task in tasks_list:
                # Check if current task matches
                csm_id = task.get("hpe_csm_reference", {}).get("csm_id")
                numeric_id = task.get("id")
                match_csm_id = mod.get("csm_id") and csm_id == mod["csm_id"]
                match_numeric_id = "id" in mod and numeric_id == mod["id"]

                if match_csm_id or match_numeric_id:
                    found_in_level = True
                    # Apply modifications
                    if "status" in mod:
                        task["status"] = mod["status"]
                    if "priority" in mod:
                        task["priority"] = mod["priority"]
                    field_path_keys = mod.get("field_path")
                    if field_path_keys and "value" in mod:
                        current_level = task
                        try:
                            for i, key in enumerate(field_path_keys):
                                if i == len(field_path_keys) - 1:
                                    current_level[key] = mod["value"]
                                else:
                                    current_level = current_level.setdefault(key, {})
                        except TypeError:
                            self.fail(f"Failed to traverse field_path {field_path_keys} for task {task.get('id', csm_id)}. Is the path correct and does the structure exist?")

                # Recurse into subtasks even if parent matches, as IDs can be duplicated
                if "subtasks" in task:
                    if recursive_modify(task["subtasks"], mod):
                        found_in_level = True
            
            return found_in_level

        for mod in modifications:
            task_found = recursive_modify(self.current_tasks_data, mod)
            if not task_found and (mod.get("csm_id") or "id" in mod):
                print(f"Warning: Modification target not found in tasks: {mod}")

    def _run_scheduler(self, tasks_list: List[Dict[str, Any]], target_date_str: str, min_focus_tasks: int = 1) -> List[Dict[str, Any]]:
        day_of_week = scheduler.get_day_of_week(target_date_str)
        task_statuses = scheduler.build_task_status_map(tasks_list)
        candidate_tasks = scheduler.filter_active_tasks(
            tasks_list,
            day_of_week,
            task_statuses,
            min_tasks_for_day_focus=min_focus_tasks
        )
        prioritized_tasks = scheduler.prioritize_active_tasks(candidate_tasks, day_of_week)
        scheduled_block = scheduler.schedule_active_tasks_into_block(prioritized_tasks)
        return scheduled_block

    def test_task1_scheduled_on_day1(self):
        target_date = "2025-05-19"
        modifications = [
            {"id": 1, "status": "pending"}
        ]
        self._modify_tasks(modifications)
        scheduled_block = self._run_scheduler(self.current_tasks_data, target_date)
        self.assertTrue(len(scheduled_block) >= 1, "Expected at least one task to be scheduled.")
        scheduled_csm_ids = [t.get("id") for t in scheduled_block]
        self.assertIn("RNA.P1.Foundations.W1.Task0", scheduled_csm_ids, "Task 1 (CSM ID) should be scheduled.")
        task1_scheduled_details = next(t for t in scheduled_block if t.get("id") == "RNA.P1.Foundations.W1.Task0")
        self.assertEqual(task1_scheduled_details.get("effort_minutes_planned"), 1.0 * 60, "Task 1 planned for its min effort.")

    def test_task2_not_scheduled_due_to_effort(self):
        target_date = "2025-05-19"
        modifications = [
            {"id": 1, "status": "done"},
            # Mark all other potentially schedulable tasks as done to isolate the test
            {"id": 3, "status": "done"},
            {"id": 4, "status": "done"},
            {"id": 5, "status": "done"},
            {"id": 7, "status": "done"},
            {"id": 11, "status": "done"},
        ]
        # Also make all subtasks of Task 2 oversized so nothing is scheduled
        for task in self.current_tasks_data:
            if task.get("id") == 2:
                for st in task.get("subtasks", []):
                    st.setdefault("hpe_learning_meta", {})["estimated_effort_hours_min"] = 2.0
        self._modify_tasks(modifications)
        scheduled_block = self._run_scheduler(self.current_tasks_data, target_date)
        scheduled_csm_ids = [t.get("id") for t in scheduled_block]
        self.assertNotIn("RNA.P1.Foundations.W1.Part1.Biochem.NucleotideStructure", scheduled_csm_ids, "Task 2 should not be scheduled as its min effort (1.5hr) exceeds block size (1hr).")
        self.assertEqual(len(scheduled_block), 0, "Expected no tasks to be scheduled if Task 2 and all its subtasks are too large and Task 1 done.")

    def test_task3_blocked_by_dependency(self):
        target_date = "2025-05-20"
        modifications = [
            {"id": 1, "status": "done"}
        ]
        self._modify_tasks(modifications)
        scheduled_block = self._run_scheduler(self.current_tasks_data, target_date)
        scheduled_csm_ids = [t.get("id") for t in scheduled_block]
        self.assertNotIn("RNA.P1.Foundations.W1.Part1.Biochem.BackboneDirectionality", scheduled_csm_ids, "Task 3 should not be scheduled due to unmet dependency.")

    def test_task3_scheduled_when_dependencies_met(self):
        target_date = "2025-05-20"
        modifications = [
            {"id": 1, "status": "done"},
            {"id": 2, "status": "done"},
            # Mark other tasks as done to isolate the test for Task 3
            {"id": 4, "status": "done"},
            {"id": 5, "status": "done"},
            {"id": 7, "status": "done"},
            {"id": 11, "status": "done"},
        ]
        self._modify_tasks(modifications)
        scheduled_block = self._run_scheduler(self.current_tasks_data, target_date)
        self.assertTrue(len(scheduled_block) >= 1, "Expected at least Task 3 to be scheduled.")
        scheduled_csm_ids = [t.get("id") for t in scheduled_block]
        self.assertIn("RNA.P1.Foundations.W1.Part1.Biochem.BackboneDirectionality", scheduled_csm_ids, "Task 3 should be scheduled.")
        task3_scheduled_details = next(t for t in scheduled_block if t.get("id") == "RNA.P1.Foundations.W1.Part1.Biochem.BackboneDirectionality")
        self.assertEqual(task3_scheduled_details.get("effort_minutes_planned"), 1.0 * 60)

    def test_empty_block_if_no_fitting_active_tasks(self):
        target_date = "2025-05-25"
        modifications = [
            {"id": task_id, "status": "done"} for task_id in [1, 3, 4, 7, 11]
        ]
        # Make all subtasks of Task 2 and 5 oversized so no promoted subtask fits
        for task in self.current_tasks_data:
            if task.get("id") in [2, 5]:
                for st in task.get("subtasks", []):
                    st.setdefault("hpe_learning_meta", {})["estimated_effort_hours_min"] = 2.0
        modifications.append({
            "id": 5,
            "field_path": ("hpe_learning_meta", "estimated_effort_hours_min"), "value": 2.0
        })
        self._modify_tasks(modifications)
        scheduled_block = self._run_scheduler(self.current_tasks_data, target_date)
        print("[DEBUG] test_empty_block_if_no_fitting_active_tasks scheduled_block:", scheduled_block)
        print("[DEBUG] test_empty_block_if_no_fitting_active_tasks candidate_tasks:", [t.get("id") for t in scheduler.filter_active_tasks(self.current_tasks_data, scheduler.get_day_of_week(target_date), scheduler.build_task_status_map(self.current_tasks_data))])
        self.assertEqual(len(scheduled_block), 0, "Block should be empty if no fitting tasks.")

    def test_subtask_promotion_parent_too_large_subtask_fits(self):
        """
        Parent task min effort > block, but one subtask fits (should be scheduled)
        """
        target_date = "2025-05-26"
        # Pick Task 2 (RNA.P1.Foundations.W1.Part1.Biochem.NucleotideStructure), min effort 1.5hr, has subtasks
        modifications = [
            # Correctly complete the dependency using its csm_id
            {"csm_id": "RNA.P1.Foundations.W1.Task0", "status": "done"},
            {"id": 2, "status": "pending"}, # Ensure parent task is pending
            {"id": 2, "field_path": ("hpe_learning_meta", "recommended_block"), "value": "active_learning"}, # Force task to be active
            {"id": 2, "field_path": ("hpe_learning_meta", "estimated_effort_hours_min"), "value": 1.5},
            # Mark other tasks as done to isolate subtask promotion
            {"id": 3, "status": "done"},
            {"id": 4, "status": "done"},
            {"id": 5, "status": "done"},
            {"id": 7, "status": "done"},
            {"id": 11, "status": "done"},
        ]
        self._modify_tasks(modifications)

        # This must be done AFTER the main modifications to prevent being overwritten
        for task in self.current_tasks_data:
            if task.get("id") == 2:
                for st in task.get("subtasks", []):
                    if st.get("id") == 1:
                        st["status"] = "pending"
                        st["hpe_learning_meta"] = {"estimated_effort_hours_min": 0.5}

        scheduled_block = self._run_scheduler(self.current_tasks_data, target_date)
        # Should schedule subtask 1 of Task 2
        self.assertTrue(any("_parent_task_id" in t for t in scheduled_block), "A promoted subtask should be scheduled.")
        subtask = next(t for t in scheduled_block if t.get("_parent_task_id") == 2)
        self.assertEqual(subtask.get("id"), 1)
        self.assertEqual(subtask["hpe_learning_meta"]["estimated_effort_hours_min"], 0.5)
        self.assertEqual(subtask["_parent_task_id"], 2)
        self.assertIn("_parent_title", subtask)

    def test_subtask_promotion_all_subtasks_too_large(self):
        """
        Parent and all subtasks have min effort > block (should schedule nothing)
        """
        target_date = "2025-05-27"
        # Pick Task 2, set parent and all subtasks to 2hr
        modifications = [
            {"id": 1, "status": "done"},
            {"id": 2, "field_path": ("hpe_learning_meta", "estimated_effort_hours_min"), "value": 2.0},
            # Mark other tasks as done to isolate test
            {"id": 3, "status": "done"},
            {"id": 4, "status": "done"},
            {"id": 5, "status": "done"},
            {"id": 7, "status": "done"},
            {"id": 11, "status": "done"},
        ]
        for task in self.current_tasks_data:
            if task.get("id") == 2:
                for st in task.get("subtasks", []):
                    st["status"] = "pending"
                    st["hpe_learning_meta"] = {"estimated_effort_hours_min": 2.0}
        self._modify_tasks(modifications)
        scheduled_block = self._run_scheduler(self.current_tasks_data, target_date)
        print("[DEBUG] scheduled_block (all subtasks too large):", scheduled_block)
        self.assertEqual(len(scheduled_block), 0, "Should not schedule any subtasks if all are too large.")

    def test_subtask_promotion_subtask_no_explicit_effort(self):
        """
        Subtask lacks explicit effort, should use parent divided by number of pending subtasks
        """
        target_date = "2025-05-28"
        modifications = [
            {"id": 1, "status": "done"},
            {"id": 2, "field_path": ("hpe_learning_meta", "estimated_effort_hours_min"), "value": 1.2},
            # Mark other tasks as done to isolate test
            {"id": 3, "status": "done"},
            {"id": 4, "status": "done"},
            {"id": 5, "status": "done"},
            {"id": 7, "status": "done"},
            {"id": 11, "status": "done"},
        ]
        self._modify_tasks(modifications)
        for task in self.current_tasks_data:
            if task.get("id") == 2:
                for st in task.get("subtasks", []):
                    st["status"] = "pending"
                    if "hpe_learning_meta" in st:
                        del st["hpe_learning_meta"]
        scheduled_block = self._run_scheduler(self.current_tasks_data, target_date)
        print("[DEBUG] test_subtask_promotion_subtask_no_explicit_effort scheduled_block:", scheduled_block)
        # Should schedule at least one subtask, with fallback effort 1.2/len(subtasks)
        subtasks = [t for t in scheduled_block if t.get("_parent_task_id") == 2]
        self.assertTrue(len(subtasks) > 0, "Should schedule promoted subtasks with fallback effort.")
        for subtask in subtasks:
            # Find the parent task object and count pending subtasks to match scheduler logic
            parent_task_for_assertion = next(t for t in self.current_tasks_data if t.get("id") == 2)
            num_pending_for_assertion = len([st for st in parent_task_for_assertion.get("subtasks", []) if st.get("status") == "pending"])
            if num_pending_for_assertion == 0:
                num_pending_for_assertion = 1
            self.assertAlmostEqual(subtask["hpe_learning_meta"]["estimated_effort_hours_min"], 1.2/num_pending_for_assertion)

    def test_subtask_promotion_reporting_and_labeling(self):
        """
        Scheduled subtask should include parent ID, parent title, and correct labeling.
        """
        target_date = "2025-05-29"
        modifications = [
            {"id": 1, "status": "done"},
            {"id": 2, "field_path": ("hpe_learning_meta", "estimated_effort_hours_min"), "value": 1.5},
            # Mark other tasks as done to isolate test
            {"id": 3, "status": "done"},
            {"id": 4, "status": "done"},
            {"id": 5, "status": "done"},
            {"id": 7, "status": "done"},
            {"id": 11, "status": "done"},
        ]
        self._modify_tasks(modifications)
        for task in self.current_tasks_data:
            if task.get("id") == 2:
                for st in task.get("subtasks", []):
                    st["status"] = "pending"
                    st["hpe_learning_meta"] = {"estimated_effort_hours_min": 0.5}
        scheduled_block = self._run_scheduler(self.current_tasks_data, target_date)
        print("[DEBUG] scheduled_block (reporting/labeling):", scheduled_block)
        subtask = next(t for t in scheduled_block if t.get("_parent_task_id") == 2)
        self.assertIn("_parent_task_id", subtask)
        self.assertIn("_parent_title", subtask)
        self.assertEqual(subtask["_parent_task_id"], 2)
        self.assertIsInstance(subtask["_parent_title"], str)

    def test_prioritization_on_day_vs_off_day(self):
        target_date = "2025-05-23"
        modifications = []
        for i in range(1, 7):
             modifications.append({"id": i, "status": "done"})
        modifications.append({
            "csm_id": "RNA.P1.Foundations.W1.Part2.Thermo.LoopsElectrostatics",
            "field_path": ("hpe_learning_meta", "recommended_block"), "value": "active_learning"
        })
        self._modify_tasks(modifications)
        for task in self.current_tasks_data:
            if task.get("id") == 1:
                task["hpe_learning_meta"]["recommended_block"] = "active_learning"
                task["status"] = "pending"
            if task.get("id") == 7:
                task["status"] = "pending"
        scheduled_block = self._run_scheduler(self.current_tasks_data, target_date, min_focus_tasks=2)
        self.assertTrue(len(scheduled_block) > 0, "Expected tasks to be scheduled.")
        scheduled_csm_ids = [t.get("id") for t in scheduled_block]
        task7_id = "RNA.P1.Foundations.W1.Part2.Thermo.LoopsElectrostatics"
        task1_id = "RNA.P1.Foundations.W1.Task0"
        self.assertIn(task7_id, scheduled_csm_ids)
        if task1_id in scheduled_csm_ids:
            self.assertLess(scheduled_csm_ids.index(task7_id), scheduled_csm_ids.index(task1_id), "Task 7 (on-day) should be before Task 1 (off-day)")
        else:
            self.assertEqual(scheduled_csm_ids[0], task7_id)

    def test_prioritization_by_taskmaster_priority(self):
        target_date = "2025-05-19"
        dummy_high_prio = {
            "id": 101, "title": "High Prio Active Task", "status": "pending", "dependencies": [],
            "priority": "high",
            "hpe_csm_reference": {"csm_id": "DUMMY.HIGH"},
            "hpe_learning_meta": {"recommended_block": "active_learning", "estimated_effort_hours_min": 0.5, "activity_type": "coding_exercise"},
            "hpe_scheduling_meta": {"planned_day_of_week": 1}
        }
        dummy_low_prio = {
            "id": 102, "title": "Low Prio Active Task", "status": "pending", "dependencies": [],
            "priority": "low",
            "hpe_csm_reference": {"csm_id": "DUMMY.LOW"},
            "hpe_learning_meta": {"recommended_block": "active_learning", "estimated_effort_hours_min": 0.5, "activity_type": "coding_exercise"},
            "hpe_scheduling_meta": {"planned_day_of_week": 1}
        }
        modifications = [
            {"id": 1, "priority": "low", "status": "pending"}
        ]
        self._modify_tasks(modifications)
        test_tasks = [dummy_high_prio, dummy_low_prio] + copy.deepcopy(self.current_tasks_data)
        scheduled_block = self._run_scheduler(test_tasks, target_date)
        scheduled_csm_ids = [t.get("id") for t in scheduled_block]
        self.assertTrue(len(scheduled_block) >= 2, "Expected at least two dummy tasks.")
        self.assertIn("DUMMY.HIGH", scheduled_csm_ids)
        self.assertIn("DUMMY.LOW", scheduled_csm_ids)
        self.assertLess(scheduled_csm_ids.index("DUMMY.HIGH"), scheduled_csm_ids.index("DUMMY.LOW"), "High priority task should be scheduled before low priority task.")

    def test_prioritization_by_min_effort(self):
        target_date = "2025-05-19"
        dummy_small_effort = {
            "id": 201, "title": "Small Effort Task", "status": "pending", "dependencies": [],
            "priority": "medium", "hpe_csm_reference": {"csm_id": "DUMMY.SMALL"},
            "hpe_learning_meta": {"recommended_block": "active_learning", "estimated_effort_hours_min": 0.25, "activity_type": "coding_exercise"},
            "hpe_scheduling_meta": {"planned_day_of_week": 1}
        }
        dummy_large_effort = {
            "id": 202, "title": "Large Effort Task", "status": "pending", "dependencies": [],
            "priority": "medium", "hpe_csm_reference": {"csm_id": "DUMMY.LARGE"},
            "hpe_learning_meta": {"recommended_block": "active_learning", "estimated_effort_hours_min": 0.75, "activity_type": "coding_exercise"},
            "hpe_scheduling_meta": {"planned_day_of_week": 1}
        }
        modifications = [{"id": 1, "priority": "low", "status": "pending"}]
        self._modify_tasks(modifications)
        test_tasks = [dummy_large_effort, dummy_small_effort] + self.current_tasks_data
        scheduled_block = self._run_scheduler(test_tasks, target_date)
        scheduled_csm_ids = [t.get("id") for t in scheduled_block]
        self.assertTrue(len(scheduled_block) >= 2, "Expected at least two dummy tasks.")
        self.assertIn("DUMMY.SMALL", scheduled_csm_ids)
        self.assertIn("DUMMY.LARGE", scheduled_csm_ids)
        self.assertLess(scheduled_csm_ids.index("DUMMY.SMALL"), scheduled_csm_ids.index("DUMMY.LARGE"), "Smaller effort task should be scheduled before larger effort task.")
        small_task_details = next(t for t in scheduled_block if t.get("id") == "DUMMY.SMALL")
        large_task_details = next(t for t in scheduled_block if t.get("id") == "DUMMY.LARGE")
        self.assertEqual(small_task_details.get("effort_minutes_planned"), 0.25 * 60)
        self.assertEqual(large_task_details.get("effort_minutes_planned"), 0.75 * 60)
        self.assertEqual(len(scheduled_block), 2, "Only the two dummy tasks should fit perfectly.")

    def test_exclusion_of_passive_review_task(self):
        target_date = "2025-05-23"
        modifications = []
        # Mark all active tasks as done to isolate the passive task
        for i in [1, 2, 3, 4, 5, 7, 11]:
            modifications.append({"id": i, "status": "done"})
        # Patch: Set activity_type to a non-active keyword to ensure robust exclusion
        modifications.append({
            "csm_id": "RNA.P1.Foundations.W1.Part2.Thermo.EnvFactors",
            "status": "pending",
            "field_path": ("hpe_learning_meta", "recommended_block"), "value": "passive_review"
        })
        modifications.append({
            "csm_id": "RNA.P1.Foundations.W1.Part2.Thermo.EnvFactors",
            "field_path": ("hpe_learning_meta", "activity_type"), "value": "summary_writing"
        })
        self._modify_tasks(modifications)
        scheduled_block = self._run_scheduler(self.current_tasks_data, target_date)
        scheduled_csm_ids = [t.get("id") for t in scheduled_block]
        self.assertNotIn("RNA.P1.Foundations.W1.Part2.Thermo.EnvFactors", scheduled_csm_ids, "Task 8 (passive) should not be scheduled by active scheduler.")
        self.assertEqual(len(scheduled_block), 0, "Expected no tasks to be scheduled if only passive tasks are candidates.")

    def test_task_considered_active_by_activity_type_keyword(self):
        target_date = "2025-05-19"
        modifications = [
            {"id": 1, "status": "pending"},
            {
                "csm_id": "RNA.P1.Foundations.W1.Task0",
                "field_path": ("hpe_learning_meta", "recommended_block"), "value": ""
            },
            {
                "csm_id": "RNA.P1.Foundations.W1.Task0",
                "field_path": ("hpe_learning_meta", "activity_type"), "value": "planning_setup_coding_exercise"
            }
        ]
        self._modify_tasks(modifications)
        scheduled_block = self._run_scheduler(self.current_tasks_data, target_date)
        scheduled_csm_ids = [t.get("id") for t in scheduled_block]
        self.assertIn("RNA.P1.Foundations.W1.Task0", scheduled_csm_ids, "Task 1 should be scheduled based on activity_type keyword.")

if __name__ == "__main__":
    unittest.main()
