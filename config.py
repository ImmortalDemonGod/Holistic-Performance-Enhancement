
# Configuration and parameters for the transformer model
# These parameters should be treated as modifiable from the main run_model.py script

device_choice = "cpu"  # Set to "cpu" for CPU
calculate_parameters = True  # Whether to calculate and print the total parameter size before training
run_for_100_epochs = True  # Whether to only run for 100 epochs and estimate time for 20,000 epochs
num_epochs = 700  # Number of training epochs
input_dim = 30  # Input feature dimension
d_model = 64  # Transformer model dimension
N = 33  # Number of encoder layers
heads = 16  # Reduced number of attention heads for efficiency
d_ff = 256  # Feedforward network dimension
output_dim = 30  # Output feature dimension
learning_rate = 0.00004  # Learning rate
batch_size = 45  # Batch size for DataLoader
dropout_rate = 0.06  # Dropout rate for regularization
