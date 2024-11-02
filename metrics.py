import torch

def compute_standard_accuracy(predictions, targets, pad_idx=10):
    """
    Args:
        predictions: torch.Tensor(B, H, W) - Model predictions
        targets: torch.Tensor(B, H, W) - Ground truth
        pad_idx: int - Index used for padding
    Returns:
        float: Accuracy excluding padding tokens
    """
    valid_mask = targets != pad_idx
    correct = ((predictions == targets) & valid_mask).sum()
    total = valid_mask.sum()
    return (correct / total).item() if total > 0 else 1.0

def compute_differential_accuracy(inputs, targets, predictions, pad_idx=10):
    """
    Args:
        inputs: torch.Tensor(B, 1, H, W) - Input grids
        targets: torch.Tensor(B, H, W) - Ground truth
        predictions: torch.Tensor(B, H, W) - Model predictions
        pad_idx: int - Index used for padding
    Returns:
        float: Accuracy on pixels that differ between input and target
    """
    # Exclude padding
    valid_mask = targets != pad_idx
    
    # Find pixels that change
    diff_mask = (inputs.squeeze(1) != targets) & valid_mask
    
    # Count correct predictions of changing pixels
    correct_diffs = (predictions == targets) & diff_mask
    
    total_diffs = diff_mask.sum()
    total_correct = correct_diffs.sum()
    
    return (total_correct / total_diffs).item() if total_diffs > 0 else 1.0

class TaskMetricsCollector:
    def __init__(self):
        self.task_metrics = {}  # Format: {task_id: {metric_name: [values]}}
        
    def add_result(self, task_id, metrics_dict):
        """
        Args:
            task_id: str - Task identifier
            metrics_dict: dict - Metrics for this task instance
                Format: {
                    'standard_accuracy': float,
                    'differential_accuracy': float
                }
        """
        if task_id not in self.task_metrics:
            self.task_metrics[task_id] = {
                'standard_accuracy': [],
                'differential_accuracy': []
            }
            
        for metric_name, value in metrics_dict.items():
            self.task_metrics[task_id][metric_name].append(value)
            
    def get_task_summary(self):
        """Returns aggregated metrics per task"""
        summary = {}
        for task_id, metrics in self.task_metrics.items():
            summary[task_id] = {
                name: sum(values) / len(values)
                for name, values in metrics.items()
            }
        return summary
