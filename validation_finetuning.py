import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import logging
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import json
from Utils.model_factory import create_transformer_trainer
from pathlib import Path
import sys
import traceback

def debug_print_tensor(name, tensor):
    """Helper function to print tensor information"""
    try:
        print(f"\nDEBUG - {name}:")
        print(f"Shape: {tensor.shape}")
        print(f"Type: {tensor.dtype}")
        print(f"Device: {tensor.device}")
        print(f"First few values: {tensor.flatten()[:5]}")
        print(f"Min/Max: {tensor.min():.2f}/{tensor.max():.2f}")
    except Exception as e:
        print(f"Error printing tensor {name}: {str(e)}")

def log_exception(e, context=""):
    """Helper function to log detailed exception information"""
    print(f"\nERROR in {context}")
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {str(e)}")
    print("\nTraceback:")
    traceback.print_exc()
    print("\n")

class ValidationFineTuner:
    def __init__(self, base_model, checkpoint_path, device='cpu', patience=5, max_epochs=100):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('finetuning.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Store parameters
        self.base_model = base_model
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.patience = patience
        self.max_epochs = max_epochs
        
        # Create checkpoint directory
        try:
            self.checkpoint_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created/verified checkpoint directory at {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint directory: {str(e)}")
            raise
        
        # Verify model device
        try:
            self.logger.info(f"Model device: {next(base_model.parameters()).device}")
            self.logger.info(f"Target device: {device}")
        except StopIteration:
            self.logger.error("The base_model has no parameters.")
            raise

    def prepare_task_data(self, val_loader, task_id):
        """Prepare validation data for a specific task"""
        self.logger.info(f"Preparing data for task_id: {task_id}")
        
        task_inputs, task_targets = [], []
        task_id_tensor = torch.tensor(task_id)

        try:
            # Extract data for specific task
            for batch_idx, batch in enumerate(val_loader):
                inputs, targets, batch_task_ids, _, _ = batch  # Adjusted unpacking to match TensorDataset
              
                # Debug print for first batch
                if batch_idx == 0:
                    debug_print_tensor("First batch inputs", inputs)
                    debug_print_tensor("First batch targets", targets)
                    debug_print_tensor("First batch task_ids", batch_task_ids)
                
                # Find examples for this task
                mask = batch_task_ids == task_id_tensor
                matching_count = mask.sum().item()
                
                self.logger.info(f"Batch {batch_idx}: Found {matching_count} examples for task {task_id}")
                
                if matching_count > 0:
                    task_inputs.append(inputs[mask])
                    task_targets.append(targets[mask])

            if not task_inputs:
                raise ValueError(f"No validation data found for task {task_id}")

            # Combine all examples
            task_inputs = torch.cat(task_inputs)
            task_targets = torch.cat(task_targets)
            
            debug_print_tensor("Combined task inputs", task_inputs)
            debug_print_tensor("Combined task targets", task_targets)
            
            # Create dataset and loader
            task_dataset = TensorDataset(task_inputs, task_targets)
            task_loader = DataLoader(
                task_dataset,
                batch_size=4,  # Smaller batch size for fine-tuning
                shuffle=True
            )
            
            self.logger.info(f"Created dataloader with {len(task_dataset)} examples")
            return task_loader
            
        except Exception as e:
            log_exception(e, f"prepare_task_data for task {task_id}")
            raise

    def fine_tune_task(self, task_id, task_loader, test_example):
        """Fine-tune model for a specific task"""
        self.logger.info(f"\n{'='*50}\nStarting fine-tuning for task {task_id}\n{'='*50}")
        
        try:
            # **Create a task-specific model using the factory function**
            task_model = create_transformer_trainer(
                config=task_config,
                checkpoint_path=None  # Instantiate without loading from checkpoint
            )
            task_model.load_state_dict(self.base_model.state_dict())
            
            # Debug model state
            self.logger.info(f"Model device after moving: {next(task_model.parameters()).device}")
            
            # Setup training
            optimizer = torch.optim.Adam(task_model.parameters(), lr=1e-5)
            criterion = nn.CrossEntropyLoss()
            best_loss = float('inf')
            best_prediction = None
            
            # Convert test example
            test_input = test_example[0].unsqueeze(0).to(self.device)
            debug_print_tensor("Test input", test_input)
            
            # Training loop
            for epoch in range(self.max_epochs):
                epoch_loss = 0
                batches = 0
                
                # Process batches
                for batch_idx, batch in enumerate(task_loader):
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    if batch_idx == 0 and epoch == 0:
                        debug_print_tensor(f"First batch inputs epoch {epoch}", inputs)
                        debug_print_tensor(f"First batch targets epoch {epoch}", targets)
                    
                    # Forward pass
                    try:
                        outputs = task_model(inputs, targets)
                        if batch_idx == 0:
                            debug_print_tensor("Model outputs", outputs)
                        
                        loss = criterion(outputs.view(-1, self.base_model.model.output_fc.out_features), targets.view(-1).long())
                        
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        batches += 1
                        
                    except Exception as e:
                        log_exception(e, f"Training batch {batch_idx} in epoch {epoch}")
                        raise
            
                avg_loss = epoch_loss / batches
                self.logger.info(f"Epoch {epoch}: Average loss = {avg_loss:.4f}")
                
                # Evaluate on test example
                task_model.eval()
                with torch.no_grad():
                    current_prediction = task_model(test_input, test_input).argmax(dim=-1)
                    debug_print_tensor("Current prediction", current_prediction)
                task_model.train()
                
                # Update best model if improved
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_prediction = current_prediction.cpu()
                    
                    # Save checkpoint
                    checkpoint_file = self.checkpoint_path / f'task_{task_id}_best.pt'
                    torch.save({
                        'task_id': task_id,
                        'model_state_dict': task_model.state_dict(),
                        'loss': best_loss,
                        'epoch': epoch
                    }, checkpoint_file)
                    self.logger.info(f"Saved new best model with loss {best_loss:.4f}")
            
            return best_prediction, best_loss
            
        except Exception as e:
            log_exception(e, f"fine_tune_task for task {task_id}")
            raise

    def evaluate_all_tasks(self, val_loader, test_loader, task_id_map):
        """Run fine-tuning and evaluation for all tasks"""
        self.logger.info("Starting evaluation of all tasks")
        
        results = {}
        idx_to_task_id = {v: k for k, v in task_id_map.items()}
        
        try:
            # Process each test example
            for batch_idx, batch in enumerate(test_loader):
                test_inputs, test_targets, task_indices, _, _ = batch  # Adjusted unpacking to match TensorDataset
                
                debug_print_tensor(f"Test batch {batch_idx} inputs", test_inputs)
                debug_print_tensor(f"Test batch {batch_idx} targets", test_targets)
                
                # Process each task in the batch
                for i, task_idx in enumerate(task_indices):
                    task_id = idx_to_task_id.get(task_idx.item(), None)
                    if task_id is None:
                        self.logger.error(f"Invalid task index {task_idx.item()} encountered.")
                        continue
                    
                    self.logger.info(f"\nProcessing task {task_id} ({i+1}/{len(task_indices)})")
                    
                    try:
                        # Prepare validation data
                        task_loader = self.prepare_task_data(val_loader, task_idx.item())
                        
                        # Get test example
                        test_example = (test_inputs[i], test_targets[i])
                        
                        # Fine-tune and predict
                        prediction, loss = self.fine_tune_task(task_id, task_loader, test_example)
                        
                        results[task_id] = {
                            'prediction': prediction.numpy().tolist(),
                            'loss': float(loss),
                            'processed_at': batch_idx
                        }
                        
                    except Exception as e:
                        self.logger.error(f"Failed processing task {task_id}")
                        log_exception(e, f"Task processing {task_id}")
                        results[task_id] = {'error': str(e)}
            
            # Save results
            results_file = self.checkpoint_path / 'fine_tuning_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Saved results to {results_file}")
            
            return results
            
        except Exception as e:
            log_exception(e, "evaluate_all_tasks")
            raise
