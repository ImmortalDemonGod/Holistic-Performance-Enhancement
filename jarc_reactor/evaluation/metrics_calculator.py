# jarc_reactor/evaluation/metrics_calculator.py
import torch
from jarc_reactor.utils.metrics import compute_standard_accuracy, compute_differential_accuracy

class MetricsCalculator:
    def __init__(self, logger):
        self.logger = logger

    def calculate_metrics(self, task_input, task_target, task_pred, outputs, idx, analysis, confidence):
        """
        Calculate standard and differential accuracy metrics for a task.

        Args:
            task_input (torch.Tensor): Input tensor for the task.
            task_target (torch.Tensor): Ground truth tensor for the task.
            task_pred (torch.Tensor): Predicted tensor for the task.
            outputs (torch.Tensor): Raw model outputs.
            idx (int): Index of the task in the batch.
            analysis (str): Distribution analysis placeholder.
            confidence (torch.Tensor): Confidence scores.

        Returns:
            dict: A dictionary containing calculated metrics and debug information.
        """
        std_acc, std_details = compute_standard_accuracy(
            task_pred.flatten(),
            task_target.flatten(),
        )
        
        diff_acc, diff_details = compute_differential_accuracy(
            task_input.squeeze(0),
            task_target.squeeze(0),
            task_pred.squeeze(0),
        )
        
        metrics = {
            'standard_accuracy': std_acc,
            'differential_accuracy': diff_acc,
            'std_details': std_details,
            'diff_details': diff_details,
            'debug_info': {
                'target_shape': list(task_target.shape),
                'pred_shape': list(task_pred.shape),
                'target_unique': torch.unique(task_target).tolist(),
                'pred_unique': torch.unique(task_pred).tolist(),
                'output_range': [float(outputs[idx].min()), float(outputs[idx].max())],
                'logits_stats': {
                    'mean': float(outputs[idx].mean()),
                    'std': float(outputs[idx].std()),
                    'min': float(outputs[idx].min()),
                    'max': float(outputs[idx].max())
                },
                'confidence_stats': {
                    'mean': float(confidence.mean()),
                    'std': float(confidence.std()),
                    'min': float(confidence.min()),
                    'max': float(confidence.max())
                },
                'distribution_analysis': analysis
            }
        }
        
        return metrics