# jarc_reactor/evaluation/batch_processor.py
import torch
from dataclasses import dataclass
from jarc_reactor.evaluation.metrics_calculator import MetricsCalculator
from jarc_reactor.utils.metrics import compute_standard_accuracy, compute_differential_accuracy, PredictionRecord

@dataclass
class BatchData:
    src: torch.Tensor
    tgt: torch.Tensor
    ctx_input: torch.Tensor
    ctx_output: torch.Tensor
    predictions: torch.Tensor
    outputs: torch.Tensor
    task_ids: torch.Tensor

@dataclass
class TaskContext:
    idx: int
    task_id_int: torch.Tensor
    mode: str
    int_to_task: dict

class BatchProcessor:
    def __init__(self, model, device, logger, metrics_calculator):
        self.model = model
        self.device = device
        self.logger = logger
        self.metrics_calculator = MetricsCalculator

    def process_batch(self, batch, mode, int_to_task):
        """Process a single batch with detailed logging and prediction storage."""
        try:
            src, tgt, ctx_input, ctx_output, task_ids = self._move_batch_to_device(batch)
            self._log_initial_tensor_info(src, tgt)
            
            with torch.no_grad():
                outputs = self.model(src, tgt, ctx_input, ctx_output)
            
            self._log_model_outputs(outputs)
            self._analyze_logits_distribution(outputs)
            
            predictions = outputs.argmax(dim=-1)  # Shape: [batch, seq_len]
            
            # Encapsulate batch data
            batch_data = BatchData(
                src=src,
                tgt=tgt,
                ctx_input=ctx_input,
                ctx_output=ctx_output,
                predictions=predictions,
                outputs=outputs,
                task_ids=task_ids
            )
            
            task_results = []
            for idx, task_id_int in enumerate(task_ids):
                # Encapsulate task context
                task_context = TaskContext(
                    idx=idx,
                    task_id_int=task_id_int,
                    mode=mode,
                    int_to_task=int_to_task
                )
                task_result = self._process_single_task(batch_data, task_context)
                task_results.append(task_result)
                    
            return task_results  # Return list of processed task results
                
        except Exception as e:
            self._handle_batch_processing_error(e, batch)
            raise

    def _move_batch_to_device(self, batch):
        """Move batch tensors to the specified device."""
        return [t.to(self.device) for t in batch]

    def _log_initial_tensor_info(self, src, tgt):
        """Log initial tensor information for source and target."""
        self.logger.debug("\nInitial tensor information:")
        self.logger.debug(f"src type: {src.dtype}, range: [{src.min():.1f}, {src.max():.1f}]")
        self.logger.debug(f"tgt type: {tgt.dtype}, range: [{tgt.min():.1f}, {tgt.max():.1f}]")
        self.logger.debug(f"src shape: {src.shape}")
        self.logger.debug(f"tgt shape: {tgt.shape}")

    def _log_model_outputs(self, outputs):
        """Log the shape, type, and range of model outputs."""
        self.logger.debug("\nModel outputs:")
        self.logger.debug(f"shape: {outputs.shape}")
        self.logger.debug(f"type: {outputs.dtype}")
        self.logger.debug(f"range: [{outputs.min():.1f}, {outputs.max():.1f}]")

    def _analyze_logits_distribution(self, outputs):
        """Analyze and log the distribution of logits for each class."""
        self.logger.debug("\nLogits Distribution:")
        logits_flat = outputs.view(-1, outputs.size(-1))
        for i in range(outputs.size(-1)):  # For each class
            class_logits = logits_flat[:, i]
            self.logger.debug(
                f"Class {i} logits - "
                f"mean: {class_logits.mean():.3f}, "
                f"std: {class_logits.std():.3f}, "
                f"range: [{class_logits.min():.3f}, {class_logits.max():.3f}]"
            )

    def _process_single_task(self, batch_data: BatchData, task_context: TaskContext):
        """Process an individual task within the batch."""
        idx = task_context.idx
        task_id = task_context.int_to_task[task_context.task_id_int.item()]
        
        # Extract tensors for the task
        task_input, task_target, task_pred, task_raw_outputs = self._extract_task_tensors(idx, batch_data)
        self._log_task_tensor_shapes(task_input, task_target, task_pred, task_raw_outputs)
        
        task_target = task_target.to(torch.long)
        task_pred = task_pred.to(torch.long)
        
        analysis = self.analyze_outputs_distribution(
            task_raw_outputs,
            task_pred,
            task_target,
            task_id
        )
        
        probs, confidence = self._calculate_probabilities_confidence(task_raw_outputs)
        pred_record = self._create_prediction_record(
            task_input, task_target, task_pred, task_raw_outputs, probs, confidence, analysis
        )
        metrics_dict = self.metrics_calculator.calculate_metrics(
            task_input, task_target, task_pred, batch_data.outputs, idx, analysis, confidence
        )
        
        return {
            'task_id': task_id,
            'metrics': metrics_dict,
            'prediction_record': pred_record
        }

    def _extract_task_tensors(self, idx, batch_data: BatchData):
        """Extract individual tensors for a specific task."""
        task_input = batch_data.src[idx:idx + 1]  # [1, H, W]
        task_target = batch_data.tgt[idx:idx + 1]  # [1, H, W]
        task_pred = batch_data.predictions[idx:idx + 1]  # [1, H, W]
        task_raw_outputs = batch_data.outputs[idx]  # [H, W, num_classes]
        return task_input, task_target, task_pred, task_raw_outputs

    def _log_task_tensor_shapes(self, task_input, task_target, task_pred, task_raw_outputs):
        """Log the shapes of task-specific tensors."""
        self.logger.debug("\nTask tensor shapes:")
        self.logger.debug(f"task_input: {task_input.shape}")
        self.logger.debug(f"task_target: {task_target.shape}")
        self.logger.debug(f"task_pred: {task_pred.shape}")
        self.logger.debug(f"task_raw_outputs: {task_raw_outputs.shape}")

    def analyze_outputs_distribution(self, outputs, predictions, targets, task_id):
        """
        Analyze distribution of outputs, predictions, and targets
        
        Args:
            outputs: [batch, H, W, num_classes] or [H, W, num_classes]
            predictions: [batch, H, W] or [H, W]
            targets: [batch, H, W] or [H, W]
            task_id: Task identifier
        """
        with torch.no_grad():
            # Debug original shapes
            self.logger.debug("\nOriginal shapes before processing:")
            self.logger.debug(f"outputs: {outputs.shape}")
            self.logger.debug(f"predictions: {predictions.shape}")
            self.logger.debug(f"targets: {targets.shape}")
            
            # Handle batch dimension properly for all tensors
            if outputs.dim() == 4:
                outputs = outputs[0]  # [H, W, num_classes]
            if predictions.dim() == 3:
                predictions = predictions[0]  # [H, W]
                
            # Special handling for targets to ensure grid structure
            if targets.dim() == 3:
                targets = targets[0]  # Remove batch dimension if present
            if targets.dim() == 1:
                # Reshape to match grid structure using known dimensions
                H = W = int(math.sqrt(targets.size(0)))  # Should be 30
                targets = targets.view(H, W)
                
            # Extra validation
            if targets.dim() != 2:
                raise ValueError(f"Expected 2D targets after processing, got {targets.dim()}D with shape {targets.shape}")
                
            # Verify shapes after processing
            self.logger.debug("\nShapes after processing:")
            self.logger.debug(f"outputs: {outputs.shape}")
            self.logger.debug(f"predictions: {predictions.shape}")
            self.logger.debug(f"targets: {targets.shape}")
            
            # Assert correct dimensions
            assert outputs.dim() == 3, f"Expected outputs to be 3D after processing, got {outputs.dim()}D"
            assert predictions.dim() == 2, f"Expected predictions to be 2D after processing, got {predictions.dim()}D"
            assert targets.dim() == 2, f"Expected targets to be 2D after processing, got {targets.dim()}D"
            assert predictions.shape == targets.shape, (
                f"Predictions shape {predictions.shape} doesn't match targets shape {targets.shape}. "
                f"Original target shape was {targets.shape}"
            )
        
        return "Distribution analysis placeholder"  # Replace with actual analysis if needed

    def _calculate_probabilities_confidence(self, task_raw_outputs):
        """Calculate softmax probabilities and confidence scores."""
        probs = torch.softmax(task_raw_outputs, dim=-1)
        confidence, _ = probs.max(dim=-1)
        return probs, confidence

    def _create_prediction_record(self, task_input, task_target, task_pred, task_raw_outputs, probs, confidence, analysis):
        """Create a PredictionRecord with detailed information."""
        return PredictionRecord(
            input_grid=task_input.squeeze(0).cpu().tolist(),
            target_grid=task_target.squeeze(0).cpu().tolist(),
            predicted_grid=task_pred.squeeze(0).cpu().tolist(),
            raw_logits=task_raw_outputs.cpu().tolist(),
            position_metrics={
                'output_probabilities': probs.cpu().tolist(),
                'output_classes': task_pred.squeeze(0).cpu().tolist(),
                'output_confidences': confidence.cpu().tolist(),
                'distribution_analysis': analysis
            }
        )

    def _handle_batch_processing_error(self, exception, batch):
        """Handle exceptions during batch processing with detailed logging."""
        self.logger.error(f"Error processing batch: {str(exception)}", exc_info=True)
        self.logger.error("Batch details at failure:")
        self.logger.error(f"Batch content: {batch}")
        """Handle exceptions during batch processing with detailed logging."""
        self.logger.error(f"Error processing batch: {str(exception)}", exc_info=True)
        self.logger.error("Batch details at failure:")
        self.logger.error(f"Batch content: {batch}")