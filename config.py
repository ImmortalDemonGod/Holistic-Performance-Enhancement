# Configuration and parameters for the transformer model
# Precision setting for PyTorch Lightning Trainer
# Options:
# - 16: Mixed precision (FP16) for reduced memory usage and faster training on supported GPUs
# - 32: Full precision (FP32), the default for most training scenarios
# - 64: Double precision (FP64), rarely used due to high computational cost
# - 'bf16': Bfloat16 precision, used for specific hardware that supports it
import torch

precision = 32  # Set to 16 for mixed precision, 32 for full precision, etc.
# These parameters should be treated as modifiable from the main run_model.py script

device_choice = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-select device
calculate_parameters = True  # Whether to calculate and print the total parameter size before training
run_for_100_epochs = True  # Whether to only run for 100 epochs and estimate time for 20,000 epochs
num_epochs = 100  # Number of training epochs
input_dim = 30  # Number of features per input row
d_model = 128  # Transformer model dimension
encoder_layers = 2  # Number of encoder layers
decoder_layers = 2   # Number of decoder layers
heads = 8  # Reduced number of attention heads for efficiency
d_ff = 256  # Feedforward network dimension
output_dim = 30  # Number of features per output row
learning_rate = 0.00007  # Learning rate
batch_size = 50  # Batch size for DataLoader
dropout_rate = 0.35  # Dropout rate for the model
include_sythtraining_data = True  # Set to True to include sythtraining data
synthetic_dir = 'sythtraining'
CHECKPOINT_PATH = 'path_to_pretrained_checkpoint.ckpt'  # Specify the actual checkpoint path
FAST_DEV_RUN = False  # Set to True to enable fast development run

# Context Encoder Configuration
context_encoder_d_model = 128  # Transformer model dimension for Context Encoder
context_encoder_heads = 8       # Number of attention heads for Context Encoder
# Fine-Tuning Configurations
finetuning_patience = 5
finetuning_max_epochs = 100
