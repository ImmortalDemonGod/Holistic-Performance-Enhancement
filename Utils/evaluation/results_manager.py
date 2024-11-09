from pathlib import Path
import json
from datetime import datetime
import torch
import logging

class EvaluationResults:
    def __init__(self, config, debug_logger):
        self.config = config
        self.debug = debug_logger
        self.output_dir = Path(config.evaluation.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'training_train': {},
            'training_validation': {},
            'evaluation': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'model': vars(config.model),
                    'training': vars(config.training),
                    'evaluation': vars(config.evaluation)
                }
            }
        }
    
    def add_result(self, mode: str, task_id: str, metrics: dict):
        """Add evaluation results for a task"""
        try:
            # Verify metrics values are valid
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    metrics[k] = v.item()
                if isinstance(v, (float, int)):
                    if not torch.isfinite(torch.tensor(v)):
                        self.debug.logger.warning(
                            f"Non-finite value in metrics for task {task_id}, key {k}: {v}"
                        )
            
            self.results[mode][task_id] = metrics
            self.debug.logger.debug(
                f"Added result for mode={mode}, task={task_id}: {metrics}"
            )
            
        except Exception as e:
            self.debug.logger.error(
                f"Error adding result for task {task_id}: {str(e)}"
            )
    
    def save(self):
        """Save results to JSON with error handling"""
        try:
            output_file = self.output_dir / 'evaluation_results.json'
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            self.debug.logger.debug(f"Saved results to {output_file}")
            
            # Save a timestamped copy for history
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            history_file = self.output_dir / f'results_{timestamp}.json'
            with open(history_file, 'w') as f:
                json.dump(self.results, f, indent=2)
                
        except Exception as e:
            self.debug.logger.error(f"Error saving results: {str(e)}")