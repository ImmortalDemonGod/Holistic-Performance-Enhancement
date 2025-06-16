import sys
from pathlib import Path

# Determine the current directory and the parent directory
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

from cultivation.systems.arc_reactor.jarc_reactor.config import Config
import random
import pytorch_lightning as pl
import torch                                                                                           
import logging                                                                                         
from pathlib import Path                                                                               
from typing import Dict, Any                                                                           
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from cultivation.systems.arc_reactor.jarc_reactor.utils.metrics import TaskMetricsCollector
from cultivation.systems.arc_reactor.jarc_reactor.utils.model_factory import create_transformer_trainer
from copy import deepcopy
from cultivation.systems.arc_reactor.jarc_reactor.utils.train import TransformerTrainer
from pytorch_lightning.callbacks import EarlyStopping                                                  
from cultivation.systems.arc_reactor.jarc_reactor.data.data_preparation import prepare_data                                                        
from cultivation.systems.arc_reactor.jarc_reactor.utils.metrics import TaskMetricsCollector
                                                                                                        
class TaskFineTuner:
    def __init__(self, base_model: TransformerTrainer, config: Config):
        """Initialize fine-tuner with base model and configuration."""
        from pathlib import Path
        # Set up directories using config
        self.log_dir = Path(config.logging.log_dir)
        self.save_dir = Path(config.finetuning.save_dir)  # Add this line
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)  # Add this line

        # Setup logging
        self.logger = logging.getLogger("task_finetuner")
        fh = logging.FileHandler(self.log_dir / "finetuning.log")
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)
        self.logger.setLevel(logging.INFO)

        # Store configuration
        self.base_model = base_model
        self.config = config
        self.device = config.training.device_choice
        self.max_epochs = config.finetuning.max_epochs
        self.learning_rate = config.finetuning.learning_rate
        self.patience = config.finetuning.patience

        # Initialize metrics collector
        self.metrics_collector = TaskMetricsCollector()

        # Store results
        self.results: Dict[str, Any] = {}
        
        # Log initialization
        self.logger.info(f"TaskFineTuner initialized:")
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
    
    def _create_task_model(self):
        """Create task-specific model with validation."""
        try:
            # Create new model instance
            task_model = create_transformer_trainer(
                config=self.config,
                checkpoint_path=None
            )
            
            # Load state dict with validation
            state_dict = self.base_model.state_dict()
            missing_keys, unexpected_keys = task_model.load_state_dict(state_dict)
            
            if missing_keys:
                self.logger.warning(f"Missing keys in state dict: {missing_keys}")
            if unexpected_keys:
                self.logger.warning(f"Unexpected keys in state dict: {unexpected_keys}")
            
            # Move to device
            task_model = task_model.to(self.device)
            
            return task_model
            
        except Exception as e:
            self.logger.error(f"Task model creation failed: {str(e)}")
            raise

    def _setup_training(self, task_id):
        """Setup training with proper configuration validation."""
        try:
            # Validate training parameters
            if self.max_epochs <= 0:
                raise ValueError(f"Invalid max_epochs: {self.max_epochs}")
            if self.patience <= 0:
                raise ValueError(f"Invalid patience: {self.patience}")
            
            # Create callbacks with validation
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    mode='min'
                )
            ]
            
            # Configure trainer with detailed logging
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                callbacks=callbacks,
                accelerator=self.device,
                devices=1,
                enable_progress_bar=True,
                enable_checkpointing=False,
                default_root_dir=self.save_dir / f"task_{task_id}",
                logger=True
            )
            
            return trainer, callbacks
            
        except Exception as e:
            self.logger.error(f"Training setup failed: {str(e)}")
            raise

    def _validate_and_move_tensor(self, tensor, name):
        """Validate tensor and move to device with error checking."""
        try:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if torch.isnan(tensor).any():
                raise ValueError(f"NaN values in {name}")
            if torch.isinf(tensor).any():
                raise ValueError(f"Inf values in {name}")
                
            # First ensure tensor is on CPU
            tensor = tensor.cpu()
            # Then move to target device
            return tensor.to(self.device)
                
        except Exception as e:
            self.logger.error(f"Tensor validation failed for {name}: {str(e)}")
            raise


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

def main(config):
    """Main entry point for fine-tuning process."""
    from pathlib import Path

    # Ensure the logs directory exists
    Path(config.logging.log_dir).mkdir(parents=True, exist_ok=True)

    # Configure logging with both file and console output
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(config.logging.log_dir) / "finetuning_debug.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("finetuning_main")
    
    try:
        from pathlib import Path

        # Use the checkpoint path from config
        checkpoint_file = Path(config.model.checkpoint_path)
        logger.info(f"Looking for checkpoint at: {config.model.checkpoint_path}")

        if not checkpoint_file.is_file():
            logger.error(f"Pretrained model checkpoint not found at {config.model.checkpoint_path}")
            return

        logger.info(f"Loading pretrained model from {config.model.checkpoint_path}")
        
        # Set device to cuda if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config.training.device_choice = device
        logger.info(f"Using device: {device}")

        # Load the base model
        base_model = create_transformer_trainer(
            config=config,
            checkpoint_path=config.model.checkpoint_path  # Changed from model_path
        )
        base_model = base_model.to(device)
        logger.info("Base model loaded and moved to device")

        # Prepare data
        logger.info("Preparing data...")
        train_loader, val_loader = prepare_data()
        logger.info("Data preparation complete")

        # Load task mapping
        task_map_path = 'task_id_map.json'
        if not Path(task_map_path).is_file():
            logger.error(f"Task ID map file not found at {task_map_path}.")
            return

        with open(task_map_path, 'r') as f:
            task_id_map = json.load(f)
            logger.info(f"Loaded task map with {len(task_id_map)} tasks")

        # Initialize fine-tuner
        finetuner = TaskFineTuner(base_model, config=config)

        # Select just one task for testing
        selected_tasks = [list(task_id_map.keys())[0]]  # Get the first task
        logger.info(f"Testing with single task: {selected_tasks[0]}")

        # Run fine-tuning
        results = finetuner.run_all_tasks(train_loader, val_loader, task_id_map, selected_tasks)

        logger.info("Fine-tuning complete")
        logger.info(f"Results saved to {finetuner.save_dir}/final_results.json")

    except Exception as e:
        logger.error(f"Fine-tuning failed: {str(e)}", exc_info=True)
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
if __name__ == "__main__":
    config = Config()
    main(config)
