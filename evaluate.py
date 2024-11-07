import logging
import config  # Imported config module
import config
import os
import torch
from Utils.model_factory import create_transformer_trainer
from Utils.data_preparation import prepare_data
from pytorch_lightning import Trainer

from Utils.metrics import compute_standard_accuracy, compute_differential_accuracy, TaskMetricsCollector
import config


import json
import argparse
from config import Config

# Add argument parser for command-line arguments
parser = argparse.ArgumentParser(description='Evaluate the Transformer model with a specified checkpoint.')
# parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file.')

args = parser.parse_args()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the task_id_map
logger.info("Loading task_id_map.json...")
with open("task_id_map.json", "r") as f:
    task_id_map = json.load(f)

logger.info("Loaded task_id_map successfully.")
int_to_task_id = {v: k for k, v in task_id_map.items()}

# Prepare data loaders
logger.info("Preparing data loaders...")
train_loader, val_loader = prepare_data()
logger.info("Data loaders prepared.")
test_loader = val_loader  # Replace with a separate test loader if available
logger.info(f"Test loader batch size: {test_loader.batch_size}")
logger.info(f"Number of batches: {len(test_loader)}")
train_loader, val_loader = prepare_data()
logger.info("Data loaders prepared.")
test_loader = val_loader  # Replace with a separate test loader if available

# Instantiate the Config class
cfg = Config()

# Validate checkpoint path using the config
checkpoint_path = cfg.model.checkpoint_path

if checkpoint_path and not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(
        f"Checkpoint file not found at {checkpoint_path}. Please provide a valid path."
    )

model = create_transformer_trainer(config=cfg, checkpoint_path=checkpoint_path)
logger.info("Loaded model from checkpoint.")

model.eval()  # Set model to evaluation mode
logger.info("Set model to evaluation mode.")

# Set up the Trainer for evaluation
cfg = Config()
accelerator = 'cuda' if cfg.training.device_choice == 'gpu' else cfg.training.device_choice
logger.info("Trainer initialized for evaluation.")
trainer = Trainer(
    devices=1,              # Use a single device
    accelerator=accelerator,  # Use device_choice from config
    logger=False,           # Disable loggers
    enable_progress_bar=True  # Enable progress bar for visibility

)

# Initialize the metrics collector
metrics_collector = TaskMetricsCollector()

# Perform evaluation
logger.info("Starting evaluation...")
for batch_idx, batch in enumerate(test_loader):
    src, tgt, ctx_input, ctx_output, task_ids = batch
    logger.debug(f"Batch {batch_idx} - src shape: {src.shape}, tgt shape: {tgt.shape}")
    with torch.no_grad():
        outputs = model(src, tgt, ctx_input, ctx_output)  # Pass context data
        predictions = outputs.argmax(dim=-1)  # Assuming output is logits

    logger.debug(f"Predictions shape: {predictions.shape}")
    if (batch_idx + 1) % 10 == 0:
        logger.info(f"Processed {batch_idx + 1} batches...")
    for idx, task_id_int in enumerate(task_ids):
        # Convert integer task_id back to string
        task_id = int_to_task_id[task_id_int.item()]
        # Get individual task tensors
        task_input = src[idx : idx + 1]
        task_target = tgt[idx : idx + 1]
        task_pred = predictions[idx : idx + 1]

        # Ensure task_pred and task_target have compatible shapes
        # Ensure task_pred and task_target have compatible shapes
        task_pred = task_pred.view(-1, 30)  # Adjusted reshape to match actual data structure
        task_target = task_target.view(-1, 30)  # Adjusted reshape to match actual data structure

        # Calculate metrics
        std_acc = compute_standard_accuracy(task_pred, task_target)
        diff_acc = compute_differential_accuracy(task_input, task_target, task_pred)

        # Store results
        metrics_collector.add_result(
            task_id, {"standard_accuracy": std_acc, "differential_accuracy": diff_acc}
        )

logger.info("Evaluation completed.")
task_summaries = metrics_collector.get_task_summary()

# Calculate overall metrics
all_std_accs = []
all_diff_accs = []

for task_metrics in task_summaries.values():
    all_std_accs.append(task_metrics["standard_accuracy"])
    all_diff_accs.append(task_metrics["differential_accuracy"])

overall_standard_accuracy = (
    sum(all_std_accs) / len(all_std_accs) if all_std_accs else 0.0
)
overall_differential_accuracy = (
    sum(all_diff_accs) / len(all_diff_accs) if all_diff_accs else 0.0
)

# Print the aggregated metrics
logger.info("All metrics have been computed and displayed.")
print(f"Overall Standard Accuracy: {overall_standard_accuracy:.4f}")
print(f"Overall Differential Accuracy: {overall_differential_accuracy:.4f}")
print("Per-Task Metrics:")
for task_id, metrics in task_summaries.items():
    std_acc = metrics["standard_accuracy"]
    diff_acc = metrics["differential_accuracy"]
    print(
        f"Task {task_id}: Standard Accuracy = {std_acc:.4f}, "
        f"Differential Accuracy = {diff_acc:.4f}"
    )

    # Check if the task was completely solved
    if std_acc >= 0.999:
        print(
            f"--> Task {task_id} was completely solved with {std_acc*100:.2f}% accuracy.\n"
        )

# No main function is defined, so we remove the call to it
