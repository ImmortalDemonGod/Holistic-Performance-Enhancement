# jarc_reactor/utils/submission_handler.py

import json
import logging
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

class SubmissionRecord:
    """Represents a single submission attempt for a task"""
    def __init__(self, grid: List[List[int]], attempt_number: int):
        self.grid = grid
        self.attempt_number = attempt_number

    def to_dict(self) -> Dict[str, List[List[int]]]:
        """Convert to submission format"""
        return {f'attempt_{self.attempt_number}': self.grid}

    @classmethod
    def from_predicted_grid(cls, predicted_grid: List[List[int]], attempt_number: int) -> 'SubmissionRecord':
        """Create SubmissionRecord from a predicted grid, handling padding"""
        formatted_grid = []
        for row in predicted_grid:
            # Filter padding values (10)
            filtered_row = []
            for val in row:
                if val == 10:  # Stop at first padding value
                    break
                filtered_row.append(val)
            if filtered_row:  # Only add non-empty rows
                formatted_grid.append(filtered_row)
        
        return cls(formatted_grid, attempt_number)

class SubmissionBuilder:
    """Handles building and validating submissions"""
    def __init__(self):
        self.submissions: Dict[str, List[SubmissionRecord]] = {}
        self.logger = logging.getLogger(__name__)

    def add_prediction(self, task_id: str, predicted_grid: List[List[int]]):
        """Add a new prediction for a task"""
        if task_id not in self.submissions:
            self.submissions[task_id] = []
            
        attempt_number = len(self.submissions[task_id]) + 1
        submission_record = SubmissionRecord.from_predicted_grid(
            predicted_grid, 
            attempt_number
        )
        self.submissions[task_id].append(submission_record)

    def build(self) -> Dict[str, List[Dict[str, List[List[int]]]]]:
        """Create final submission dictionary"""
        submission = {}
        for task_id, records in self.submissions.items():
            submission[task_id] = [
                record.to_dict() for record in records
            ]
        return submission

class SubmissionValidator:
    """Validates submission format and content"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_grid(self, grid: List[List[int]], task_id: str, attempt_key: str) -> bool:
        """Validate a single grid's format and values"""
        try:
            # Check grid is non-empty
            if not grid:
                raise ValueError("Empty grid")

            # Check grid dimensions
            if len(grid) > 30 or any(len(row) > 30 for row in grid):
                raise ValueError("Grid dimensions exceed 30x30")

            # Check grid values
            if not all(all(isinstance(val, int) and 0 <= val <= 9 for val in row) for row in grid):
                raise ValueError("Invalid grid values")

            # Check rectangular shape
            row_lengths = {len(row) for row in grid}
            if len(row_lengths) > 1:
                raise ValueError("Grid is not rectangular")

            return True

        except ValueError as e:
            self.logger.error(f"Grid validation failed for task {task_id}, {attempt_key}: {str(e)}")
            return False

    def validate_submission(self, submission: Dict[str, List[Dict[str, List[List[int]]]]]) -> bool:
        """Validate entire submission structure and content"""
        try:
            for task_id, predictions in tqdm(submission.items(), desc="Validating submission"):
                # Check predictions is a list
                if not isinstance(predictions, list):
                    raise ValueError(f"Task {task_id}: predictions must be a list")

                # Check each prediction
                for pred_dict in predictions:
                    # Validate attempt keys
                    attempt_keys = [key for key in pred_dict.keys() if key.startswith('attempt_')]
                    if not attempt_keys:
                        raise ValueError(f"Task {task_id}: missing attempt key")

                    # Validate each attempt's grid
                    for attempt_key in attempt_keys:
                        grid = pred_dict[attempt_key]
                        if not self.validate_grid(grid, task_id, attempt_key):
                            return False

            self.logger.info("Submission validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Submission validation failed: {str(e)}")
            return False

class SubmissionManager:
    """Manages the entire submission process"""
    def __init__(self, output_dir: str = "submissions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        self.builder = SubmissionBuilder()
        self.validator = SubmissionValidator()

    def create_from_evaluation(self, evaluation_results: Dict[str, Any]) -> Dict[str, List[Dict[str, List[List[int]]]]]:
        """Create submission from evaluation results"""
        try:
            task_summaries = evaluation_results.get('task_summaries', {})
            
            for task_id, task_data in tqdm(task_summaries.items(), desc="Processing tasks"):
                predictions = task_data.get('predictions', [])
                
                for prediction in predictions:
                    predicted_grid = prediction.get('predicted_grid', [])
                    self.builder.add_prediction(task_id, predicted_grid)
            
            submission = self.builder.build()
            
            # Validate submission
            if not self.validator.validate_submission(submission):
                raise ValueError("Generated submission failed validation")
            
            return submission
            
        except Exception as e:
            self.logger.error(f"Error creating submission: {str(e)}")
            raise

    def save_submission(self, submission: Dict[str, List[Dict[str, List[List[int]]]]]) -> Path:
        """Save submission to file with timestamp"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.output_dir / f'submission_{timestamp}.json'
            
            with open(filepath, 'w') as f:
                json.dump(submission, f)
            
            self.logger.info(f"Saved submission to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving submission: {str(e)}")
            raise

def create_submission_from_evaluation(evaluation_results: Dict[str, Any], output_dir: str = "submissions") -> Path:
    """Convenience function to create and save submission from evaluation results"""
    manager = SubmissionManager(output_dir)
    submission = manager.create_from_evaluation(evaluation_results)
    return manager.save_submission(submission)