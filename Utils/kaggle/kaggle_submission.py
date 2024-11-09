# kaggle_submission.py

import json
import torch
from pathlib import Path
from typing import Dict, List, Any
import logging
from tqdm import tqdm
import kagglehub
import os

from Utils.padding_utils import pad_to_fixed_size
from Utils.context_data import ContextPair
from config import Config
from train import TransformerTrainer
from Utils.model_factory import create_transformer_trainer

class KaggleSubmissionHandler:
    def __init__(self, model, config, output_dir: str = "/kaggle/working"):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup logging
        self.logger = logging.getLogger('kaggle_submission')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.output_dir / 'submission.log')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

    def load_challenges(self, challenge_path: str) -> Dict[str, Any]:
        """Load challenge data from JSON file."""
        try:
            with open(challenge_path, 'r') as f:
                challenges = json.load(f)
            self.logger.info(f"Loaded {len(challenges)} challenges from {challenge_path}")
            return challenges
        except Exception as e:
            self.logger.error(f"Error loading challenges: {str(e)}")
            raise

    def prepare_task_data(self, task_data: Dict[str, Any]) -> tuple:
        """Prepare input tensors for a single task."""
        try:
            # Get context from first training example
            context_example = task_data['train'][0]
            context_input = pad_to_fixed_size(
                torch.tensor(context_example['input'], dtype=torch.float32),
                target_shape=(30, 30)
            )
            context_output = pad_to_fixed_size(
                torch.tensor(context_example['output'], dtype=torch.float32),
                target_shape=(30, 30)
            )
            context_pair = ContextPair(context_input, context_output)
            
            # Prepare test input
            test_inputs = []
            for test_case in task_data['test']:
                test_input = pad_to_fixed_size(
                    torch.tensor(test_case['input'], dtype=torch.float32),
                    target_shape=(30, 30)
                )
                test_inputs.append(test_input)
            
            return context_pair, test_inputs
            
        except Exception as e:
            self.logger.error(f"Error preparing task data: {str(e)}")
            raise

    def predict_output(self, src, ctx_input, ctx_output) -> List[List[int]]:
        """Generate prediction for a single test case."""
        try:
            with torch.no_grad():
                # Move tensors to device
                src = src.unsqueeze(0).to(self.device)
                ctx_input = ctx_input.unsqueeze(0).to(self.device)
                ctx_output = ctx_output.unsqueeze(0).to(self.device)
                
                # Create initial target (will be updated during inference)
                tgt = torch.zeros_like(src)
                
                # Get model prediction
                output = self.model(src, tgt, ctx_input, ctx_output)
                
                # Convert to grid format
                prediction = output.argmax(dim=-1).squeeze().cpu()
                
                # Convert to list format and remove padding
                grid = prediction.tolist()
                
                # Find actual grid dimensions (exclude padding)
                height = len(grid)
                width = len(grid[0]) if height > 0 else 0
                
                # Trim padding (values of 10)
                while height > 0 and all(cell == 10 for cell in grid[height-1]):
                    height -= 1
                while width > 0 and all(row[width-1] == 10 for row in grid[:height]):
                    width -= 1
                
                return [row[:width] for row in grid[:height]]
                
        except Exception as e:
            self.logger.error(f"Error generating prediction: {str(e)}")
            raise

    def create_submission(self, challenge_path: str) -> Dict[str, List[Dict[str, List[List[int]]]]]:
        """Create submission dictionary for all challenges."""
        try:
            self.logger.info("Starting submission creation...")
            challenges = self.load_challenges(challenge_path)
            submission = {}
            
            for task_id, task_data in tqdm(challenges.items(), desc="Processing tasks"):
                try:
                    # Prepare task data
                    context_pair, test_inputs = self.prepare_task_data(task_data)
                    
                    # Generate predictions for each test case
                    task_predictions = []
                    for test_input in test_inputs:
                        prediction = self.predict_output(
                            test_input,
                            context_pair.context_input,
                            context_pair.context_output
                        )
                        task_predictions.append({"output": prediction})
                    
                    submission[task_id] = task_predictions
                    
                except Exception as e:
                    self.logger.error(f"Error processing task {task_id}: {str(e)}")
                    # Use placeholder for failed predictions
                    submission[task_id] = [{"output": [[0]]} for _ in range(len(task_data['test']))]
            
            return submission
            
        except Exception as e:
            self.logger.error(f"Error creating submission: {str(e)}")
            raise

    def save_submission(self, submission: Dict[str, List[Dict[str, List[List[int]]]]], filename: str = "submission.json"):
        """Save submission dictionary to JSON file."""
        try:
            output_path = self.output_dir / filename
            with open(output_path, 'w') as f:
                json.dump(submission, f)
            self.logger.info(f"Saved submission to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving submission: {str(e)}")
            raise

    def run_submission_pipeline(self, challenge_path: str, submission_filename: str = "submission.json"):
        """Run complete submission pipeline."""
        try:
            self.logger.info("Starting submission pipeline...")
            
            # Create submission
            submission = self.create_submission(challenge_path)
            
            # Validate submission format
            self._validate_submission(submission)
            
            # Save submission
            self.save_submission(submission, submission_filename)
            
            self.logger.info("Submission pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Submission pipeline failed: {str(e)}")
            raise

    def _validate_submission(self, submission: Dict[str, List[Dict[str, List[List[int]]]]]):
        """Validate submission format and content."""
        try:
            for task_id, predictions in submission.items():
                # Check that predictions is a list
                if not isinstance(predictions, list):
                    raise ValueError(f"Task {task_id}: predictions must be a list")
                
                # Validate each prediction
                for i, pred in enumerate(predictions):
                    # Check prediction structure
                    if not isinstance(pred, dict) or 'output' not in pred:
                        raise ValueError(f"Task {task_id}, prediction {i}: invalid format")
                    
                    # Validate output grid
                    grid = pred['output']
                    if not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
                        raise ValueError(f"Task {task_id}, prediction {i}: invalid grid format")
                    
                    # Check grid values
                    if not all(all(isinstance(cell, int) and 0 <= cell <= 9 for cell in row) for row in grid):
                        raise ValueError(f"Task {task_id}, prediction {i}: invalid grid values")
                    
                    # Check grid dimensions
                    height = len(grid)
                    width = len(grid[0]) if height > 0 else 0
                    if height < 1 or height > 30 or width < 1 or width > 30:
                        raise ValueError(f"Task {task_id}, prediction {i}: invalid grid dimensions")
            
            self.logger.info("Submission validation passed")
            
        except Exception as e:
            self.logger.error(f"Submission validation failed: {str(e)}")
            raise


def load_model_and_create_submission():
    try:
        # Initialize configuration
        config = Config()
        
        # Load model from Kaggle Hub
        MODEL_SLUG = "arc_checkpoint_test"
        VARIATION_SLUG = "default"
        
        # Create local directory for model
        model_dir = Path("/kaggle/working/model")
        model_dir.mkdir(exist_ok=True)
        
        # Download model checkpoint
        checkpoint_path = kagglehub.model_download(
            f"miguelingram/{MODEL_SLUG}/keras/{VARIATION_SLUG}"
        )
        
        print(f"Downloaded checkpoint to: {checkpoint_path}")
        
        # Update config with checkpoint path
        config.model.checkpoint_path = checkpoint_path
        
        # Create model using factory
        model = create_transformer_trainer(
            config=config,
            checkpoint_path=checkpoint_path
        )
        
        print("Model loaded successfully")
        
        # Initialize submission handler
        submission_handler = KaggleSubmissionHandler(
            model=model,
            config=config,
            output_dir="/kaggle/working"
        )
        
        # Run submission pipeline
        submission_handler.run_submission_pipeline(
            challenge_path="/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json",
            submission_filename="submission.json"
        )
        
        print("Submission created successfully")
        
        # Verify submission file exists
        submission_path = Path("/kaggle/working/submission.json")
        if submission_path.exists():
            print(f"Submission file created at {submission_path}")
            # Print first few entries of submission
            with open(submission_path, 'r') as f:
                submission = json.load(f)
                print("\nFirst submission entry:")
                first_key = next(iter(submission))
                print(f"Task {first_key}:")
                print(json.dumps(submission[first_key], indent=2))
        else:
            print("Error: Submission file not created")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    # Login to Kaggle
    kagglehub.login()
    
    # Load model and create submission
    load_model_and_create_submission()