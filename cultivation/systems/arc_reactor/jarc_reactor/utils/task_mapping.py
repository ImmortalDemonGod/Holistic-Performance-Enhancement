import json
import torch
from torch.utils.data import DataLoader
from typing import Set, Dict
import logging
from pathlib import Path
from jarc_reactor.data.eval_data_prep import prepare_data as prepare_eval_data

class TaskMapper:
    def __init__(self, logger: logging.Logger, config):
        self.logger = logger
        self.config = config
        self.train_task_map: Dict[str, int] = {}
        self.train_int_to_task: Dict[int, str] = {}
        self.eval_task_map: Dict[str, int] = {}
        self.eval_int_to_task: Dict[int, str] = {}
        self.has_training_data: bool = False
        self.has_eval_data: bool = False

    def load_training_task_map(self):
        """Load the training task map from JSON"""
        try:
            self.logger.debug("Loading task_id_map.json...")
            with open("task_id_map.json", "r") as f:
                self.train_task_map = json.load(f)
            self.train_int_to_task = {v: k for k, v in self.train_task_map.items()}
            self.has_training_data = True
            self.logger.info(f"Loaded {len(self.train_task_map)} training tasks")
        except FileNotFoundError:
            self.logger.warning("task_id_map.json not found - skipping training data evaluation")
            self.has_training_data = False
        except Exception as e:
            self.logger.error(f"Error loading training task map: {str(e)}")
            self.has_training_data = False

    def load_evaluation_task_map(self):
        """Load or create the evaluation task map"""
        try:
            self.logger.debug("Loading eval_id_map.json...")
            self.eval_task_map = self.load_eval_map_from_file()
        except FileNotFoundError:
            self.logger.info("eval_id_map.json not found - creating from evaluation data...")
            self.eval_task_map = self.create_eval_map()
        
        if self.eval_task_map:
            self.eval_int_to_task = {v: k for k, v in self.eval_task_map.items()}
            self.has_eval_data = True
            self.logger.debug(
                f"Created reverse mapping with {len(self.eval_int_to_task)} entries"
            )
        else:
            self.logger.error("No evaluation task map created")
            self.has_eval_data = False

    def load_eval_map_from_file(self) -> Dict[str, int]:
        """Load evaluation task map from eval_id_map.json"""
        with open("eval_id_map.json", "r") as f:
            eval_task_map = json.load(f)
        self.logger.info(f"Loaded {len(eval_task_map)} evaluation tasks")
        return eval_task_map

    def create_eval_map(self) -> Dict[str, int]:
        """Create evaluation task map from evaluation data"""
        try:
            _, eval_dataset = prepare_eval_data(
                return_datasets=True,
                config=self.config  # Pass the config instance
            )
            unique_task_ids = self.extract_unique_task_ids(eval_dataset)
            if not unique_task_ids:
                raise ValueError("No task IDs found in evaluation dataset")
            
            eval_task_map = {
                task_id: idx 
                for idx, task_id in enumerate(sorted(unique_task_ids))
            }
            
            self.save_eval_map(eval_task_map)
            return eval_task_map
        except Exception as e:
            self.logger.error(f"Error creating eval_id_map.json: {str(e)}")
            self.has_eval_data = False
            raise

    def extract_unique_task_ids(self, eval_dataset) -> Set[str]:
        """Extract unique task IDs from the evaluation dataset"""
        unique_task_ids = set()
        eval_loader = DataLoader(eval_dataset, batch_size=1)
        for _, _, _, _, task_ids in eval_loader:
            task_id = task_ids[0].item()
            unique_task_ids.add(str(task_id))
        return unique_task_ids

    def save_eval_map(self, eval_task_map: Dict[str, int]):
        """Save the evaluation task map to eval_id_map.json"""
        try:
            with open("eval_id_map.json", "w") as f:
                json.dump(eval_task_map, f, indent=2)
            self.logger.info(
                f"Created and saved eval_id_map.json with {len(eval_task_map)} tasks"
            )
        except Exception as e:
            self.logger.error(f"Error saving eval_id_map.json: {str(e)}")
            # Continue even if save fails - we still have the map in memory

    def validate_task_maps(self):
        """Validate that at least one task map is available"""
        if not (self.has_training_data or self.has_eval_data):
            self.logger.error("Neither training nor evaluation data maps are available")
            raise ValueError("No task maps found or created - cannot perform any evaluation")
