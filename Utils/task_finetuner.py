import sys
from pathlib import Path

# Determine the current directory and the parent directory
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

from config import Config
import random
import pytorch_lightning as pl
import torch                                                                                           
import logging                                                                                         
from pathlib import Path                                                                               
from typing import Dict, Any                                                                           
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from train import TransformerTrainer
from pytorch_lightning.callbacks import EarlyStopping                                                  
from Utils.data_preparation import prepare_data                                                        
from Utils.metrics import TaskMetricsCollector                                                         
                                                                                                        
class TaskFineTuner:
    def __init__(self, base_model: TransformerTrainer, config: Config):
        """Initialize fine-tuner with base model and configuration."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger("task_finetuner")
        fh = logging.FileHandler(self.save_dir / "finetuning.log")
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)
        self.logger.setLevel(logging.INFO)

        # Store configuration
        self.base_model = base_model
        self.device = device
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.base_model = base_model
        self.device = device
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.patience = patience

        # Initialize metrics collector
        self.metrics_collector = TaskMetricsCollector()

        # Store results
        self.results: Dict[str, Any] = {}

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
        """Fine-tune model for specific task and evaluate."""
        self.logger.info(f"\n{'='*50}\nFine-tuning task {task_id}\n{'='*50}")
        
        # Unpack test example
        src, tgt, ctx_input, ctx_output, _ = test_example
        
        # Log test example details
        self.logger.info("\nTest Example:")
        self.logger.info(f"Input shape: {src.shape}")
        self.logger.info(f"Target shape: {tgt.shape}")
        self.logger.info(f"Target values (first 20): {tgt.flatten()[:20].tolist()}")  # Show first 20 values
        
        # Get base model prediction before fine-tuning
        self.base_model.eval()
        with torch.no_grad():
            src_b = src.unsqueeze(0).to(self.device)
            tgt_b = tgt.unsqueeze(0).to(self.device)
            ctx_input_b = ctx_input.unsqueeze(0).to(self.device)
            ctx_output_b = ctx_output.unsqueeze(0).to(self.device)
            
            base_prediction = self.base_model(src_b, tgt_b, ctx_input_b, ctx_output_b)
            base_prediction = base_prediction.argmax(dim=-1)
            
            # Calculate base accuracy
            base_acc = (base_prediction.cpu() == tgt.long()).float().mean().item()
            self.logger.info(f"\nBase Model Accuracy: {base_acc:.4f}")
            self.logger.info(f"Base Model Prediction (first 20): {base_prediction.cpu().flatten()[:20].tolist()}")

        # Proceed with fine-tuning as usual
        # Initialize task-specific model with overridden learning_rate
        hparams = self.base_model.hparams.copy()
        hparams.pop('learning_rate', None)  # Remove existing learning_rate
        hparams.pop('device_choice', None)  # Remove 'device_choice' if present

        task_model = TransformerTrainer(
            **hparams,  # Unpack hyperparameters without learning_rate and device_choice
            learning_rate=self.learning_rate  # Add the desired learning_rate
        )
        task_model.load_state_dict(self.base_model.state_dict())
        task_model.to(self.device)

        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                mode='min'
            )
        ]

        # Configure trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            accelerator=self.device,
            devices=1,
            enable_progress_bar=True,
            enable_checkpointing=False,
            default_root_dir=self.save_dir / f"task_{task_id}"
        )

        # Train model
        try:
            trainer.fit(task_model, train_loader, val_loader)

            # Evaluate on test example
            task_model.eval()
            with torch.no_grad():
                src, tgt, ctx_input, ctx_output, _ = test_example
                src = src.unsqueeze(0).to(self.device)
                tgt = tgt.unsqueeze(0).to(self.device)
                ctx_input = ctx_input.unsqueeze(0).to(self.device)
                ctx_output = ctx_output.unsqueeze(0).to(self.device)

                prediction = task_model(src, tgt, ctx_input, ctx_output)
                prediction = prediction.argmax(dim=-1)

            # Calculate metrics
            final_acc = (prediction.cpu() == tgt.long()).float().mean().item()
            accuracy_improvement = final_acc - base_acc

            metrics = {
                'val_loss': trainer.callback_metrics.get('val_loss', float('inf')).item(),
                'base_accuracy': base_acc,
                'base_prediction': base_prediction.cpu().flatten().tolist(),
                'target': tgt.cpu().flatten().tolist(),
                'final_prediction': prediction.cpu().flatten().tolist(),
                'final_accuracy': final_acc,
                'accuracy_improvement': accuracy_improvement,
                'converged_epoch': trainer.current_epoch
            }

            self.logger.info(f"\nResults for task {task_id}:")
            self.logger.info(f"Base Accuracy: {metrics['base_accuracy']:.4f}")
            self.logger.info(f"Final Accuracy: {metrics['final_accuracy']:.4f}")
            self.logger.info(f"Improvement: {metrics['accuracy_improvement']:.4f}")
            self.logger.info(f"Final Prediction (first 20): {metrics['final_prediction'][:20]}")

            # Save task results
            self.results[task_id] = metrics
            self.metrics_collector.add_result(task_id, metrics)
            return metrics

        except Exception as e:
            self.logger.error(f"Failed fine-tuning task {task_id}: {str(e)}")
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
                formatted_results = {
                    'test_details': {
                        'target': self.results[task_id]['target'],
                        'input_shape': list(test_examples[task_id][0].shape)
                    },
                    'base_model': {
                        'prediction': self.results[task_id]['base_prediction'],
                        'accuracy': self.results[task_id]['base_accuracy']
                    },
                    'fine_tuned_model': {
                        'prediction': self.results[task_id]['final_prediction'],
                        'accuracy': self.results[task_id]['final_accuracy'],
                        'val_loss': self.results[task_id]['val_loss'],
                        'converged_epoch': self.results[task_id]['converged_epoch']
                    },
                    'improvement': self.results[task_id]['accuracy_improvement']
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

    # Configure logging based on LoggingConfig
    logging.basicConfig(level=getattr(logging, config.logging.level.upper(), logging.INFO))
    logger = logging.getLogger("finetuning_main")
    
    if config.logging.debug_mode:
        logger.setLevel(logging.DEBUG)
        logging.debug("Debug mode is enabled in LoggingConfig.")

    try:
        model_path = config.model.checkpoint_path
        checkpoint_file = Path(model_path)
        if not checkpoint_file.is_file():
            logger.error(f"Pretrained model checkpoint not found at {model_path}.")
            return

        logger.info(f"Loading pretrained model from {model_path}")
        # Define the device based on configuration
        device = config.training.device_choice
        logger.info(f"Using device: {device}")

        # Load the base model from checkpoint using config
        model_path = config.model.checkpoint_path
        base_model = TransformerTrainer.load_from_checkpoint(
            model_path,
            include_synthetic_training_data=config.training.include_synthetic_training_data
        )
        
        # Move the base model to the specified device
        base_model.to(device)  # Ensure model is on the correct device

        # Prepare data
        train_loader, val_loader = prepare_data()

        # Load task mapping
        task_map_path = 'task_id_map.json'
        if not Path(task_map_path).is_file():
            logger.error(f"Task ID map file not found at {task_map_path}.")
            return

        with open(task_map_path, 'r') as f:
            task_id_map = json.load(f)

        # Initialize fine-tuner
        finetuner = TaskFineTuner(base_model, config=config, device=device)

        # Determine fine-tuning mode from Config
        mode = config.finetuning.mode
        if mode == 'random':
            num_tasks = config.finetuning.num_random_tasks
            selected_tasks = random.sample(list(task_id_map.keys()), k=num_tasks)
            logger.info(f"Fine-tuning will be performed on {num_tasks} randomly selected tasks: {selected_tasks}")
        elif mode == 'all':
            selected_tasks = list(task_id_map.keys())
            logger.info(f"Fine-tuning will be performed on all {len(selected_tasks)} tasks.")

        # Run fine-tuning
        results = finetuner.run_all_tasks(train_loader, val_loader, task_id_map, selected_tasks)

        logger.info("Fine-tuning complete")
        logger.info(f"Results saved to {finetuner.save_dir}/final_results.json")

    except Exception as e:
        logger.error(f"Fine-tuning failed: {str(e)}")
        raise

if __name__ == "__main__":
    config = Config()  # Initialize Config instance
    main(config)  # Pass Config instance to main
