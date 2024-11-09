import torch
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from Utils.metrics import TaskMetricsCollector

class ModelEvaluator:
    def __init__(self, model, config, data_manager, results_manager, debug_logger):
        self.model = model
        self.config = config
        self.data_manager = data_manager
        self.results = results_manager
        self.debug = debug_logger
        self.metrics_collector = TaskMetricsCollector()
        
        # Move model to correct device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = self.model.to(self.device)
        self.debug.logger.debug(f"Model moved to device: {self.device}")
    
    def _process_batch(self, batch, task_map):
        """Process a single batch with detailed logging"""
        try:
            src, tgt, ctx_input, ctx_output, task_ids = batch
            
            # Log input tensors
            self.debug.log_tensor('source', src)
            self.debug.log_tensor('target', tgt)
            self.debug.log_tensor('context_input', ctx_input)
            self.debug.log_tensor('context_output', ctx_output)
            self.debug.log_tensor('task_ids', task_ids)
            
            # Move tensors to device
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            ctx_input = ctx_input.to(self.device)
            ctx_output = ctx_output.to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                output = self.model(src, tgt, ctx_input, ctx_output)
            self.debug.log_tensor('model_output', output)
            
            # Process each example in batch
            for i in range(len(task_ids)):
                task_id_idx = task_ids[i].item()
                task_id = task_map[task_id_idx]
                
                # Get individual tensors
                task_input = src[i:i+1]
                task_target = tgt[i:i+1]
                task_pred = output[i:i+1]
                
                self.debug.logger.debug(
                    f"Processing task {task_id} "
                    f"(shape: in={task_input.shape}, "
                    f"target={task_target.shape}, "
                    f"pred={task_pred.shape})"
                )
                
                # Calculate metrics
                metrics = self._calculate_metrics(
                    task_input, task_target, task_pred
                )
                
                return task_id, metrics
                
        except Exception as e:
            self.debug.logger.error(f"Error processing batch: {str(e)}")
            raise
    
    def _calculate_metrics(self, task_input, task_target, task_pred):
        """Calculate evaluation metrics with validation"""
        from Utils.metrics import compute_standard_accuracy, compute_differential_accuracy
        
        try:
            # Ensure shapes match
            self.debug.logger.debug(
                f"Calculating metrics for shapes: "
                f"input={task_input.shape}, "
                f"target={task_target.shape}, "
                f"pred={task_pred.shape}"
            )
            
            # Calculate metrics
            std_acc = compute_standard_accuracy(task_pred, task_target)
            diff_acc = compute_differential_accuracy(
                task_input, task_target, task_pred
            )
            
            self.debug.logger.debug(
                f"Calculated metrics: "
                f"standard_accuracy={std_acc}, "
                f"differential_accuracy={diff_acc}"
            )
            
            return {
                "standard_accuracy": std_acc,
                "differential_accuracy": diff_acc
            }
            
        except Exception as e:
            self.debug.logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def evaluate_dataset(self, dataset, mode: str, task_map: dict):
        """Evaluate a complete dataset"""
        self.debug.logger.debug(f"\nStarting evaluation for mode: {mode}")
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        for batch in tqdm(loader, desc=f"Evaluating {mode}"):
            try:
                task_id, metrics = self._process_batch(batch, task_map)
                self.results.add_result(mode, task_id, metrics)
            except Exception as e:
                self.debug.logger.error(
                    f"Error evaluating batch in {mode}: {str(e)}"
                )
                continue
    
    def run_evaluation(self):
        """Run evaluation based on config mode"""
        mode = self.config.evaluation.mode
        self.debug.logger.debug(f"Starting evaluation in mode: {mode}")
        
        try:
            if mode in ['training-train', 'all']:
                train_dataset, _ = self.data_manager.get_training_data()
                self.evaluate_dataset(
                    train_dataset, 
                    'training_train',
                    self.data_manager.train_task_map
                )
            
            if mode in ['training-validation', 'all']:
                _, val_dataset = self.data_manager.get_training_data()
                self.evaluate_dataset(
                    val_dataset,
                    'training_validation',
                    self.data_manager.train_task_map
                )
            
            if mode in ['evaluation-only', 'all']:
                eval_dataset = self.data_manager.get_evaluation_data()
                self.evaluate_dataset(
                    eval_dataset,
                    'evaluation',
                    self.data_manager.eval_task_map
                )
            
            # Save final results
            self.results.save()
            
        except Exception as e:
            self.debug.logger.error(f"Evaluation failed: {str(e)}")
            raise