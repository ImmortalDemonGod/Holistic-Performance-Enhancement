import logging
import config
import os
import torch
from train import TransformerTrainer, prepare_data
from pytorch_lightning import Trainer
from metrics import compute_standard_accuracy, compute_differential_accuracy, TaskMetricsCollector

import json

# Load the task_id_map
logger.info("Loading task_id_map.json...")
with open('task_id_map.json', 'r') as f:
    task_id_map = json.load(f)

logger.info("Loaded task_id_map successfully.")
int_to_task_id = {v: k for k, v in task_id_map.items()}
# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prepare data loaders
logger.info("Preparing data loaders...")
train_loader, val_loader = prepare_data()
logger.info("Data loaders prepared.")
test_loader = val_loader  # Replace with a separate test loader if available

# Instantiate the model and load the checkpoint
# Validate checkpoint path
if config.CHECKPOINT_PATH and not os.path.isfile(config.CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint file not found at {config.CHECKPOINT_PATH}. Please update config.py with a valid path.")

model = TransformerTrainer.load_from_checkpoint(config.CHECKPOINT_PATH, strict=True)
logger.info("Loaded model from checkpoint.")

model.eval()  # Set model to evaluation mode
logger.info("Set model to evaluation mode.")

# Set up the Trainer for evaluation
logger.info("Trainer initialized for evaluation.")
trainer = Trainer(
    devices=1,              # Use a single device (CPU)
    accelerator='cpu',     # Set explicitly to CPU
    logger=False,          # Disable loggers
    enable_progress_bar=True  # Enable progress bar for visibility
)

# Initialize the metrics collector
metrics_collector = TaskMetricsCollector()

# Perform evaluation
logger.info("Starting evaluation...")
for batch_idx, batch in enumerate(test_loader):
    src, tgt, task_ids = batch  # Ensure task_ids are included
    with torch.no_grad():
        outputs = model(src, tgt)
        predictions = outputs.argmax(dim=1)  # Assuming output is logits

    if (batch_idx + 1) % 10 == 0:
        logger.info(f"Processed {batch_idx + 1} batches...")
    for idx, task_id_int in enumerate(task_ids):
        # Convert integer task_id back to string
        task_id = int_to_task_id[task_id_int.item()]
        # Get individual task tensors
        task_input = src[idx:idx+1]
        task_target = tgt[idx:idx+1]
        task_pred = predictions[idx:idx+1]

        # Calculate metrics
        std_acc = compute_standard_accuracy(task_pred, task_target)
        diff_acc = compute_differential_accuracy(
            task_input, task_target, task_pred
        )

        # Store results
        metrics_collector.add_result(task_id, {
            'standard_accuracy': std_acc,
            'differential_accuracy': diff_acc
        })

logger.info("Evaluation completed.")
task_summaries = metrics_collector.get_task_summary()

# Calculate overall metrics
all_std_accs = []
all_diff_accs = []

for task_metrics in task_summaries.values():
    all_std_accs.append(task_metrics['standard_accuracy'])
    all_diff_accs.append(task_metrics['differential_accuracy'])

overall_standard_accuracy = sum(all_std_accs) / len(all_std_accs) if all_std_accs else 0.0
overall_differential_accuracy = sum(all_diff_accs) / len(all_diff_accs) if all_diff_accs else 0.0

# Print the aggregated metrics
logger.info("All metrics have been computed and displayed.")
print(f"Overall Standard Accuracy: {overall_standard_accuracy:.4f}")
print(f"Overall Differential Accuracy: {overall_differential_accuracy:.4f}")
print("Per-Task Metrics:")
for task_id, metrics in task_summaries.items():
    std_acc = metrics['standard_accuracy']
    diff_acc = metrics['differential_accuracy']
    print(f"Task {task_id}: Standard Accuracy = {std_acc:.4f}, "
          f"Differential Accuracy = {diff_acc:.4f}")
    
    # Check if the task was completely solved
    if std_acc >= 0.999:
        print(f"--> Task {task_id} was completely solved with {std_acc*100:.2f}% accuracy.\n")

# No main function is defined, so we remove the call to it
