
# Configuration and parameters for the transformer model
# These parameters should be treated as modifiable from the main run_model.py script

device_choice = "cpu"  # Set to "cpu" for CPU
calculate_parameters = True  # Whether to calculate and print the total parameter size before training
run_for_100_epochs = True  # Whether to only run for 100 epochs and estimate time for 20,000 epochs
num_epochs = 30  # Number of training epochs
input_dim = 30  # Number of features per input row
d_model = 128  # Transformer model dimension
N = 15  # Number of encoder layers
heads = 16  # Reduced number of attention heads for efficiency
d_ff = 1024  # Feedforward network dimension
output_dim = 30  # Number of features per output row
learning_rate = 0.00007  # Learning rate
batch_size = 100  # Batch size for DataLoader
