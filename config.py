# Configuration and parameters for the transformer model
# Precision setting for PyTorch Lightning Trainer
# Options:
# - 16: Mixed precision (FP16) for reduced memory usage and faster training on supported GPUs
# - 32: Full precision (FP32), the default for most training scenarios
# - 64: Double precision (FP64), rarely used due to high computational cost
# - 'bf16': Bfloat16 precision, used for specific hardware that supports it
precision = 64  # Set to 16 for mixed precision, 32 for full precision, etc.
# These parameters should be treated as modifiable from the main run_model.py script

device_choice = "cpu"  # Set to "cpu" for CPU
calculate_parameters = True  # Whether to calculate and print the total parameter size before training
run_for_100_epochs = True  # Whether to only run for 100 epochs and estimate time for 20,000 epochs
num_epochs = 30  # Number of training epochs
input_dim = 30  # Number of features per input row
d_model = 1128  # Transformer model dimension
encoder_layers = 0  # Number of encoder layers
decoder_layers = 7   # Number of decoder layers
heads = 8  # Reduced number of attention heads for efficiency
d_ff = 256  # Feedforward network dimension
output_dim = 30  # Number of features per output row
learning_rate = 0.0001  # Learning rate
batch_size = 26  # Batch size for DataLoader
dropout_rate = 0.35  # Dropout rate for the model
include_sythtraining_data = False  # Set to True to include sythtraining data
