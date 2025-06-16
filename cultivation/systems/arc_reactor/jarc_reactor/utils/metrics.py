import torch
import logging
from typing import Dict, List, Union, Any
import numpy as np

logger = logging.getLogger(__name__)

def compute_standard_accuracy(predictions, targets, pad_idx=10):
    """
    Compute accuracy with shape and type validation.
    
    Args:
        predictions: torch.Tensor - Model predictions
        targets: torch.Tensor - Ground truth targets
        pad_idx: int - Value used for padding
        
    Returns:
        tuple: (accuracy: float, details: dict)
    """
    # Convert types for consistent comparison
    predictions = predictions.to(torch.long)
    targets = targets.to(torch.long)
    
    # Ensure tensors are on same device
    if predictions.device != targets.device:
        predictions = predictions.to(targets.device)
    
    # Create mask for valid positions
    valid_mask = targets != pad_idx
    valid_count = valid_mask.sum().item()
    
    if valid_count == 0:
        logger.warning("No valid positions found (all padding)")
        return 0.0, {
            'valid_count': 0,
            'correct_count': 0,
            'predictions': predictions.cpu().tolist(),
            'targets': targets.cpu().tolist(),
            'valid_mask': valid_mask.cpu().tolist()
        }
    
    # Calculate accuracy
    matches = (predictions == targets) & valid_mask
    correct = matches.sum().item()
    accuracy = correct / valid_count
    
    details = {
        'valid_count': valid_count,
        'correct_count': correct,
        'predictions': predictions.cpu().tolist(),
        'targets': targets.cpu().tolist(),
        'valid_mask': valid_mask.cpu().tolist(),
        'matches': matches.cpu().tolist()
    }
    
    logger.debug(f"Correct predictions: {correct}/{valid_count} = {accuracy:.4f}")
    return accuracy, details

def compute_differential_accuracy(inputs, targets, predictions, pad_idx=10):
    """
    Compute accuracy for changing positions only.
    
    Args:
        inputs: torch.Tensor - Input grid
        targets: torch.Tensor - Ground truth targets
        predictions: torch.Tensor - Model predictions
        pad_idx: int - Value used for padding
        
    Returns:
        tuple: (accuracy: float, details: dict)
    """
    # Convert types for consistent comparison
    inputs = inputs.to(torch.long)
    targets = targets.to(torch.long)
    predictions = predictions.to(torch.long)
    
    # Create masks
    valid_mask = targets != pad_idx
    diff_mask = (inputs != targets) & valid_mask
    changing_count = diff_mask.sum().item()
    
    if changing_count == 0:
        logger.warning("No changing positions found")
        return 0.0, {
            'changing_count': 0,
            'correct_count': 0,
            'predictions': predictions.cpu().tolist(),
            'targets': targets.cpu().tolist(),
            'inputs': inputs.cpu().tolist(),
            'diff_mask': diff_mask.cpu().tolist()
        }
    
    # Calculate accuracy for changing positions
    correct_changes = ((predictions == targets) & diff_mask).sum().item()
    accuracy = correct_changes / changing_count
    
    details = {
        'changing_count': changing_count,
        'correct_count': correct_changes,
        'predictions': predictions.cpu().tolist(),
        'targets': targets.cpu().tolist(),
        'inputs': inputs.cpu().tolist(),
        'diff_mask': diff_mask.cpu().tolist()
    }
    
    logger.debug(f"Correct predictions of changes: {correct_changes}/{changing_count} = {accuracy:.4f}")
    return accuracy, details

class PredictionRecord:
    """Store prediction details for a single example"""
    def __init__(self, 
                 input_grid: List[List[int]],
                 target_grid: List[List[int]],
                 predicted_grid: List[List[int]],
                 raw_logits: List[List[List[float]]],
                 position_metrics: Dict[str, float] = None):
        self.input_grid = input_grid
        self.target_grid = target_grid
        self.predicted_grid = predicted_grid
        self.raw_logits = raw_logits
        self.position_metrics = position_metrics or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'input_grid': self.input_grid,
            'target_grid': self.target_grid,
            'predicted_grid': self.predicted_grid,
            'raw_logits': self.raw_logits,
            'position_metrics': self.position_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionRecord':
        return cls(
            input_grid=data['input_grid'],
            target_grid=data['target_grid'],
            predicted_grid=data['predicted_grid'],
            raw_logits=data['raw_logits'],
            position_metrics=data.get('position_metrics', {})
        )

class TaskMetricsCollector:
    """Collect and aggregate metrics and predictions for multiple tasks."""
    
    def __init__(self):
        self.task_metrics = {}  # Format: {task_id: {'metrics': {}, 'predictions': []}}
        self.logger = logging.getLogger(__name__)
    
    def add_result(self, task_id: str, metrics_dict: Dict[str, Any], 
                  prediction_record: PredictionRecord = None):
        """
        Add a new result for a task.
        
        Args:
            task_id: Task identifier
            metrics_dict: Metrics and standard accuracy/differential accuracy details
            prediction_record: Optional PredictionRecord containing detailed predictions
        """
        # Initialize task if not seen before
        if task_id not in self.task_metrics:
            self.task_metrics[task_id] = {
                'metrics': {},
                'predictions': []
            }
        
        # Store metrics
        for metric_name, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                # Handle numeric metrics
                if metric_name not in self.task_metrics[task_id]['metrics']:
                    self.task_metrics[task_id]['metrics'][metric_name] = []
                self.task_metrics[task_id]['metrics'][metric_name].append(value)
            else:
                # Store additional metric details with the prediction
                # prediction_details = value  # TODO: Decide how to handle non-numeric metric details
                pass  # Or log a warning, or decide how to store 'value'
        
        # Store prediction record if provided
        if prediction_record:
            self.task_metrics[task_id]['predictions'].append(
                prediction_record.to_dict()
            )
        
        self.logger.debug(
            f"Added result for task {task_id}: metrics={metrics_dict}, "
            f"prediction_stored={'yes' if prediction_record else 'no'}"
        )
    
    def get_task_summary(self) -> Dict[str, Dict[str, Union[float, List[Dict[str, Any]]]]]:
        """
        Returns aggregated metrics and predictions per task.
        """
        summary = {}
        
        for task_id, data in self.task_metrics.items():
            summary[task_id] = {
                'metrics': {},
                'predictions': data['predictions']  # Store all predictions
            }
            
            # Average numeric metrics
            for metric_name, values in data['metrics'].items():
                if values:  # Check if we have any values
                    summary[task_id]['metrics'][metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)) if len(values) > 1 else 0.0,
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'num_samples': len(values)
                    }
            
            # Add prediction statistics
            if data['predictions']:
                correct_predictions = sum(
                    1 for pred in data['predictions']
                    if pred['target_grid'] == pred['predicted_grid']
                )
                summary[task_id]['metrics']['perfect_matches'] = {
                    'count': correct_predictions,
                    'percentage': (correct_predictions / len(data['predictions'])) * 100
                }
        
        return summary
    
    def get_overall_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate overall metrics across all tasks."""
        all_values = {}
        
        # Collect all values for each metric
        for task_data in self.task_metrics.values():
            for metric_name, values in task_data['metrics'].items():
                if metric_name not in all_values:
                    all_values[metric_name] = []
                all_values[metric_name].extend(values)
        
        # Calculate statistics
        overall_metrics = {}
        for metric_name, values in all_values.items():
            if values:  # Check if we have any values
                overall_metrics[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)) if len(values) > 1 else 0.0,
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'num_samples': len(values)
                }
        
        return overall_metrics
    
    def save_detailed_predictions(self, task_id: str, filename: str):
        """
        Save detailed predictions for a specific task to a file.
        
        Args:
            task_id: Task to save predictions for
            filename: Where to save the predictions
        """
        import json
        
        if task_id not in self.task_metrics:
            logger.warning(f"No predictions found for task {task_id}")
            return
        
        predictions = self.task_metrics[task_id]['predictions']
        
        with open(filename, 'w') as f:
            json.dump({
                'task_id': task_id,
                'predictions': predictions,
                'metrics': self.task_metrics[task_id]['metrics']
            }, f, indent=2)
        
        logger.info(f"Saved {len(predictions)} predictions for task {task_id} to {filename}")