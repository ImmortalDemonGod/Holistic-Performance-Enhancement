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
FAST_DEV_RUN = True  # Set to True to enable fast development run
TRAIN_FROM_CHECKPOINT = False  # Set to True to resume training from a checkpoint

# Context Encoder Configuration
context_encoder_d_model = 128  # Transformer model dimension for Context Encoder
context_encoder_heads = 8       # Number of attention heads for Context Encoder
# Fine-Tuning Configurations
finetuning_patience = 5
finetuning_max_epochs = 100

class OptunaConfig:
    def __init__(self):
        # Debug: Print when config is created
        print("DEBUG: Creating OptunaConfig")
        
        self.n_trials = 100
        self.study_name = "jarc_optimization"
        self.storage_url = "sqlite:///jarc_optuna.db"
        
        # Hyperparameter ranges
        self.param_ranges = {
            "learning_rate": (1e-5, 1e-2),
            "batch_size": (16, 128),
            "d_model": (64, 256),
            "decoder_layers": (4, 32),
            "heads": (2, 16),
            "d_ff": (128, 512),
            "dropout": (0.1, 0.5)
        }
        
        print("DEBUG: OptunaConfig created with ranges:", self.param_ranges)

class Config:
    def __init__(self):
        self.model = self.ModelConfig()
        self.training = self.TrainingConfig()
        self.optuna = OptunaConfig()
    
    class ModelConfig:
        def __init__(self):
            self.input_dim = input_dim
            self.d_model = d_model
            self.encoder_layers = encoder_layers
            self.decoder_layers = decoder_layers
            self.heads = heads
            self.d_ff = d_ff
            self.output_dim = output_dim
            self.dropout = dropout_rate
    
    class TrainingConfig:
        def __init__(self):
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.include_sythtraining_data = include_sythtraining_data
