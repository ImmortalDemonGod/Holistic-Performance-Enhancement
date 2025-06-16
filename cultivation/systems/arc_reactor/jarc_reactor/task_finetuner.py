import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Third-party imports
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

# Local application imports
# The following sys.path manipulation is to ensure local imports work correctly.
# Determine the current directory and the parent directory
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

# from cultivation.systems.arc_reactor.jarc_reactor.config import Config # Removed old config
from cultivation.systems.arc_reactor.jarc_reactor.utils.metrics import TaskMetricsCollector  # noqa: E402
from cultivation.systems.arc_reactor.jarc_reactor.utils.model_factory import create_transformer_trainer  # noqa: E402
from cultivation.systems.arc_reactor.jarc_reactor.data.data_preparation import prepare_data  # noqa: E402
from cultivation.systems.arc_reactor.jarc_reactor.utils.train import TransformerTrainer  # noqa: E402
from cultivation.utils.logging_config import setup_logging  # noqa: E402
                                                                                                        
class TaskFineTuner:
    def __init__(self, base_model: TransformerTrainer, cfg: DictConfig):
        """Initialize fine-tuner with base model and configuration."""
        # Set up directories using config
        self.log_dir = Path(cfg.logging.log_dir)
        self.save_dir = Path(cfg.finetuning.save_dir)  # Add this line
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)  # Add this line

        # Logging is configured in the main entry point.
        # This logger will propagate messages to the root logger.
        self.logger = logging.getLogger("task_finetuner")

        # Store configuration
        self.base_model = base_model
        self.cfg = cfg
        self.device = cfg.training.device_choice
        self.max_epochs = cfg.finetuning.max_epochs
        self.learning_rate = cfg.finetuning.learning_rate
        self.patience = cfg.finetuning.patience

        # Initialize metrics collector
        self.metrics_collector = TaskMetricsCollector()

        # Store results
        self.results: Dict[str, Any] = {}
        
        # Log initialization
        self.logger.info("TaskFineTuner initialized:")
        self.logger.info(f"Save directory: {self.save_dir}")
        self.logger.info(f"Log directory: {self.log_dir}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Max epochs: {self.max_epochs}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Patience: {self.patience}")

    def prepare_task_data(self, train_loader, val_loader, task_id, task_id_map):
        """Extract task-specific data from loaders."""
        # Map the task_id string to its corresponding integer index
        task_id_idx = task_id_map[task_id]
        task_id_tensor = torch.tensor(task_id_idx)

        def filter_task_data(loader, purpose="training"):
            inputs, outputs, ctx_inputs, ctx_outputs = [], [], [], []

            for batch in loader:
                src, tgt, ctx_input, ctx_output, task_ids = batch
                # Create a mask for the current task_id
                mask = (task_ids == task_id_tensor)
                if mask.any():
                    inputs.append(src[mask])
                    outputs.append(tgt[mask])
                    ctx_inputs.append(ctx_input[mask])
                    ctx_outputs.append(ctx_output[mask])

            if not inputs:
                raise ValueError(f"No {purpose} data found for task {task_id}")

            return (
                torch.cat(inputs),
                torch.cat(outputs),
                torch.cat(ctx_inputs),
                torch.cat(ctx_outputs),
                torch.full((len(torch.cat(inputs)),), task_id_idx, dtype=torch.long)  # Add task_ids
            )

        # Get task-specific data
        train_data = filter_task_data(train_loader, "training")
        val_data = filter_task_data(val_loader, "validation")

        # Create task-specific datasets
        train_dataset = self._create_dataset(*train_data)
        val_dataset = self._create_dataset(*val_data)

        # Create dataloaders
        task_train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        task_val_loader = DataLoader(val_dataset, batch_size=4)

        return task_train_loader, task_val_loader

    def _create_dataset(self, src, tgt, ctx_input, ctx_output, task_ids):
        """Helper to create dataset with consistent formatting."""
        return TensorDataset(src, tgt, ctx_input, ctx_output, task_ids)  # Now includes task_ids

    def finetune_task(self, task_id: str, train_loader, val_loader, test_example):
        """Fine-tune model for specific task and evaluate with enhanced debugging."""
        try:
            self.logger.info(f"\n{'='*50}\nFine-tuning task {task_id}\n{'='*50}")
            
            # Debug memory usage at start
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
                self.logger.debug(f"Initial GPU memory: {initial_memory / 1e6:.2f} MB")
            
            # Unpack and validate test example
            try:
                src, tgt, ctx_input, ctx_output, _ = test_example
                self.logger.debug(f"Test example shapes - src: {src.shape}, tgt: {tgt.shape}")
                self.logger.debug(f"Context shapes - input: {ctx_input.shape}, output: {ctx_output.shape}")
            except Exception as e:
                self.logger.error(f"Failed to unpack test example: {str(e)}")
                raise ValueError("Invalid test example format") from e
            
            # Base model evaluation with error catching
            try:
                base_metrics = self._evaluate_base_model(
                    src, tgt, ctx_input, ctx_output
                )
                self.logger.info(f"Base model accuracy: {base_metrics['accuracy']:.4f}")
            except Exception as e:
                self.logger.error(f"Base model evaluation failed: {str(e)}")
                raise
            
            # Create task-specific model with validation
            try:
                task_model = self._create_task_model()
                self.logger.debug("Task model created successfully")
            except Exception as e:
                self.logger.error(f"Failed to create task model: {str(e)}")
                raise
            
            # Setup training with proper error handling
            try:
                trainer, callbacks = self._setup_training(task_id)
                self.logger.debug("Training setup complete")
            except Exception as e:
                self.logger.error(f"Failed to setup training: {str(e)}")
                raise
            
            # Training with enhanced monitoring
            try:
                self.logger.info("Starting model training")
                trainer.fit(task_model, train_loader, val_loader)
                self.logger.info(f"Training completed at epoch {trainer.current_epoch}")
            except Exception as e:
                self.logger.error(f"Training failed: {str(e)}")
                raise
            
            # Final evaluation with detailed metrics
            try:
                metrics = self._evaluate_finetuned_model(
                    task_model, src, tgt, ctx_input, ctx_output,
                    base_metrics, trainer, task_id  # Add task_id here
                )
                self.logger.info("\nFinal Results:")
                self.logger.info(f"Base Accuracy: {metrics['base_accuracy']:.4f}")
                self.logger.info(f"Final Accuracy: {metrics['final_accuracy']:.4f}")
                self.logger.info(f"Improvement: {metrics['improvement']:.4f}")
            except Exception as e:
                self.logger.error(f"Final evaluation failed: {str(e)}")
                raise
                    
           # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated()
                self.logger.debug(f"Final GPU memory: {final_memory / 1e6:.2f} MB")
            
            return metrics
                
        except Exception as e:
            self.logger.error(f"Task {task_id} fine-tuning failed: {str(e)}")
            # Attempt cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
 

    def _evaluate_base_model(self, src, tgt, ctx_input, ctx_output):
        """Evaluate base model with consistent device handling."""
        self.base_model.eval()
        with torch.no_grad():
            try:
                # First make sure all input tensors are on CPU
                src = src.cpu()
                tgt = tgt.cpu()
                ctx_input = ctx_input.cpu()
                ctx_output = ctx_output.cpu()
                
                # Then move everything to the target device together
                src_b = src.unsqueeze(0).to(self.device)
                tgt_b = tgt.unsqueeze(0).to(self.device)
                ctx_input_b = ctx_input.unsqueeze(0).to(self.device)
                ctx_output_b = ctx_output.unsqueeze(0).to(self.device)
                
                # Get prediction
                base_prediction = self.base_model(src_b, tgt_b, ctx_input_b, ctx_output_b)
                if torch.isnan(base_prediction).any():
                    raise ValueError("NaN values in base model prediction")
                    
                # Move prediction to CPU for comparison
                base_prediction = base_prediction.cpu()
                base_prediction = base_prediction.argmax(dim=-1)
                
                # Calculate accuracy on CPU
                base_acc = (base_prediction == tgt.long()).float().mean().item()
                
                return {
                    'accuracy': base_acc,
                    'prediction': base_prediction,
                    'prediction_distribution': base_prediction.unique(return_counts=True)
                }
                
            except Exception as e:
                self.logger.error(f"Base model evaluation error: {str(e)}")
                raise
    
    def _evaluate_finetuned_model(self, task_model, src, tgt, ctx_input, ctx_output, base_metrics, trainer, task_id):
        """Evaluate fine-tuned model with consistent device handling."""
        try:
            # Explicitly move model and ensure it's in eval mode
            task_model = task_model.to(self.device)
            task_model.eval()

            with torch.no_grad():
                # Move all inputs to the correct device first
                src_b = src.unsqueeze(0).to(self.device)
                tgt_b = tgt.unsqueeze(0).to(self.device)
                ctx_input_b = ctx_input.unsqueeze(0).to(self.device)
                ctx_output_b = ctx_output.unsqueeze(0).to(self.device)
                
                # Get prediction and ensure it's on the correct device
                final_prediction = task_model(src_b, tgt_b, ctx_input_b, ctx_output_b)
                final_prediction = final_prediction.to(self.device)
                
                # Move everything to CPU for metrics calculation
                final_prediction = final_prediction.cpu()
                final_prediction = final_prediction.argmax(dim=-1)
                tgt_cpu = tgt.cpu()
                
                # Calculate accuracy on CPU
                final_acc = (final_prediction == tgt_cpu.long()).float().mean().item()
                
                # Get validation loss (ensure it's on CPU)
                val_loss = trainer.callback_metrics.get('val_loss_epoch', 0.0)
                if torch.is_tensor(val_loss):
                    val_loss = val_loss.cpu().item()
                else:
                    val_loss = float(val_loss)
                
                metrics = {
                    'base_accuracy': base_metrics['accuracy'],
                    'final_accuracy': final_acc,
                    'improvement': final_acc - base_metrics['accuracy'],
                    'val_loss': val_loss,
                    'converged_epoch': trainer.current_epoch,
                    'target': tgt_cpu.tolist(),
                    'base_prediction': base_metrics['prediction'].tolist(),
                    'final_prediction': final_prediction.tolist()
                }
                
                # Store the results
                self.metrics_collector.add_result(task_id, metrics)
                self.results[task_id] = metrics
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Fine-tuned model evaluation error: {str(e)}")
            raise
    
    def _create_task_model(self) -> TransformerTrainer:
        """Create a new model instance for task-specific fine-tuning."""
        return create_transformer_trainer(config=self.cfg, checkpoint_path=None)

    def _setup_training(self, task_id: str):
        """Set up trainer and callbacks for fine-tuning."""
        # Callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss", 
            patience=self.cfg.finetuning.patience, 
            verbose=True, 
            mode="min"
        )
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.save_dir / task_id,  # Save to task-specific subdirectory
            filename="{epoch}-{val_loss:.2f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )
        callbacks = [early_stopping, checkpoint_callback]

        # Trainer
        trainer = pl.Trainer(
            max_epochs=self.cfg.finetuning.max_epochs,
            callbacks=callbacks,
            logger=False,  # Disable default logger to avoid conflicts
            enable_checkpointing=True,
            accelerator=self.cfg.training.device_choice if self.cfg.training.device_choice != 'auto' else ('gpu' if torch.cuda.is_available() else 'cpu'),
            devices=1 if self.cfg.training.device_choice != 'auto' else None,
            precision=self.cfg.training.precision,
            gradient_clip_val=self.cfg.training.gradient_clip_val,
        )
        return trainer, callbacks

    def run_all_tasks(self, train_loader, val_loader, task_id_map, selected_task_ids=None):
        """Fine-tune and evaluate specified tasks. If no tasks are specified, fine-tune all tasks."""
        self.logger.info("Starting fine-tuning for all tasks")

        # Create a reverse mapping from index to task_id
        idx_to_task_id = {idx: task_id for task_id, idx in task_id_map.items()}

        # Get test examples for each task
        test_examples = {}
        for batch in val_loader:
            src, tgt, ctx_input, ctx_output, task_ids = batch
            # Iterate through the batch to find a test example for each task_id
            for i, task_id_idx in enumerate(task_ids):
                task_id = idx_to_task_id[task_id_idx.item()]
                if selected_task_ids and task_id not in selected_task_ids:
                    continue  # Skip tasks not selected
                if task_id not in test_examples:
                    test_examples[task_id] = (
                        src[i], 
                        tgt[i], 
                        ctx_input[i], 
                        ctx_output[i], 
                        task_ids[i]
                    )
            # Optionally, break early if all selected tasks have been assigned
            if selected_task_ids and len(test_examples) >= len(selected_task_ids):
                break

        if not test_examples:
            self.logger.error("No test examples found for any tasks.")
            raise ValueError("No test examples found for any tasks.")

        # Determine the list of tasks to process
        if selected_task_ids:
            tasks_to_process = selected_task_ids
        else:
            tasks_to_process = list(task_id_map.keys())

        # Process each task with a progress bar
        for task_id in tqdm(tasks_to_process, desc="Fine-tuning Tasks", unit="task"):
            try:
                if task_id not in test_examples:
                    self.logger.error(f"No test example found for task {task_id}. Skipping.")
                    self.results[task_id] = {'error': 'No test example found.'}
                    self.metrics_collector.add_result(task_id, {'error': 'No test example found.'})
                    continue

                # Prepare task data
                task_train_loader, task_val_loader = self.prepare_task_data(
                    train_loader, val_loader, task_id, task_id_map
                )

                # Fine-tune and evaluate
                metrics = self.finetune_task(
                    task_id,
                    task_train_loader,
                    task_val_loader,
                    test_examples[task_id]
                )

                self.logger.info(f"Completed task {task_id} - Val Loss: {metrics['val_loss']:.4f}")

            except Exception as e:
                self.logger.error(f"Failed processing task {task_id}: {str(e)}")
                self.results[task_id] = {'error': str(e)}
                self.metrics_collector.add_result(task_id, {'error': str(e)})

        # **Update the JSON Structure before saving**
        for task_id in list(self.results.keys()):
            if 'error' not in self.results[task_id]:
                task_metrics = self.results[task_id]
                formatted_results = {
                    'test_details': {
                        'target': task_metrics['target'],
                        'input_shape': list(test_examples[task_id][0].shape)
                    },
                    'base_model': {
                        'prediction': task_metrics['base_prediction'],
                        'accuracy': task_metrics['base_accuracy']
                    },
                    'fine_tuned_model': {
                        'prediction': task_metrics['final_prediction'],
                        'accuracy': task_metrics['final_accuracy'],
                        'val_loss': task_metrics['val_loss'],
                        'converged_epoch': task_metrics['converged_epoch']
                    },
                    'improvement': task_metrics['improvement']  # Use improvement directly
                }
                self.results[task_id] = formatted_results

        # Then save as before:
        results_file = self.save_dir / 'final_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"Saved final results to {results_file}")

        return self.results

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main function to run the fine-tuning pipeline, configured by Hydra."""
    logger = logging.getLogger(__name__)
    try:
        # Setup logging based on config
        setup_logging(config=cfg.logging)  # Pass logging sub-config
        logger.info("Starting fine-tuning process...")

        # Load base model from checkpoint if specified, otherwise train from scratch
        base_model_checkpoint = cfg.model.get('checkpoint_path') # Use cfg
        if base_model_checkpoint and Path(base_model_checkpoint).exists():
            logger.info(f"Loading base model from checkpoint: {base_model_checkpoint}")
            base_model = create_transformer_trainer(
                config=cfg, # Use cfg
                checkpoint_path=base_model_checkpoint
            )
        else:
            logger.info("No valid checkpoint path provided or file doesn't exist. Training base model from scratch or using uninitialized model.")
            base_model = create_transformer_trainer(config=cfg) # Use cfg

        # Prepare data
        train_dataset, val_dataset, task_id_map = prepare_data(
            cfg=cfg, # Use cfg
            return_datasets=True, 
            return_task_id_map=True
        )
        train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True) # Use cfg
        val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size) # Use cfg

        if task_id_map is None: # This block should ideally not be needed if prepare_data is robust
            task_map_path = Path(cfg.data.data_dir) / "task_id_map.json" # Use cfg
            if not task_map_path.exists():
                logger.error(f"Task ID map file not found at {task_map_path}. This is required.")
                raise FileNotFoundError(f"Task ID map not found: {task_map_path}")
            with open(task_map_path, 'r') as f:
                task_id_map = json.load(f)
            logger.info(f"Loaded task map from {task_map_path} with {len(task_id_map)} tasks")

        # Initialize fine-tuner
        finetuner = TaskFineTuner(base_model, cfg=cfg) # Use cfg and pass as cfg

        # Select tasks for fine-tuning based on config mode
        if cfg.finetuning.mode == "all":
            tasks_to_finetune = list(task_id_map.keys())
        elif cfg.finetuning.mode == "random":
            import random
            num_tasks = min(cfg.finetuning.num_random_tasks, len(task_id_map))
            tasks_to_finetune = random.sample(list(task_id_map.keys()), num_tasks)
        elif cfg.finetuning.mode == "specific":
            tasks_to_finetune = cfg.finetuning.specific_tasks
            if not tasks_to_finetune:
                 logger.warning("Finetuning mode is 'specific' but no specific_tasks provided. Defaulting to first task.")
                 tasks_to_finetune = [list(task_id_map.keys())[0]]
        else: # Fallback or error for unknown mode
            logger.warning(f"Unknown fine-tuning mode: {cfg.finetuning.mode}. Defaulting to first task.")
            tasks_to_finetune = [list(task_id_map.keys())[0]]

        logger.info(f"Selected tasks for fine-tuning: {tasks_to_finetune}")

        # Run fine-tuning for selected tasks
        # Ensure run_all_tasks can handle a list of selected_task_ids
        finetuner.run_all_tasks(train_loader, val_loader, task_id_map, selected_tasks=tasks_to_finetune)

        logger.info("Fine-tuning complete")
        # The results are saved within run_all_tasks or a similar method in TaskFineTuner
        # logger.info(f"Results saved to {finetuner.save_dir}/final_results.json") # This might be redundant if saved inside

    except Exception as e:
        logger.error(f"Fine-tuning script failed: {str(e)}", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise # Re-raise the exception after logging and cleanup

if __name__ == "__main__":
    main()
