# Configuration and parameters for the transformer model
# Precision setting for PyTorch Lightning Trainer

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
        self.context_encoder_d_model = context_encoder_d_model
        self.context_encoder_heads = context_encoder_heads
        self.checkpoint_path = CHECKPOINT_PATH  # Path to checkpoint file for resuming training

class TrainingConfig:
    def __init__(self, batch_size, learning_rate, include_synthetic_training_data, num_epochs, device_choice='cpu', precision=None, fast_dev_run=None, train_from_checkpoint=None):
        if precision is None:
            precision = 32  # Default precision
        if fast_dev_run is None:
            fast_dev_run = True  # Default fast_dev_run
        if train_from_checkpoint is None:
            train_from_checkpoint = False  # Default train_from_checkpoint
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = num_epochs
        self.device_choice = device_choice
        self.precision = precision  # Moved here
        self.train_from_checkpoint = train_from_checkpoint
        self.include_synthetic_training_data = include_synthetic_training_data
        assert self.device_choice in ['cpu', 'gpu'], "device_choice must be 'cpu' or 'gpu'"
        assert self.precision in [16, 32, 64, 'bf16'], "Invalid precision value"
        self.gradient_clip_val = 1.0
        self.FAST_DEV_RUN = fast_dev_run
# Options:
# - 16: Mixed precision (FP16) for reduced memory usage and faster training on supported GPUs
# - 32: Full precision (FP32), the default for most training scenarios
# - 64: Double precision (FP64), rarely used due to high computational cost
# - 'bf16': Bfloat16 precision, used for specific hardware that supports it
import torch
import logging

precision = 32  # Set to 16 for mixed precision, 32 for full precision, etc.
TRAIN_FROM_CHECKPOINT = False  # Set to True to resume training from a checkpoint
# These parameters should be treated as modifiable from the main run_model.py script

device_choice = 'gpu' if torch.cuda.is_available() else 'cpu'  # Auto-select device
calculate_parameters = True  # Whether to calculate and print the total parameter size before training
run_for_100_epochs = True  # Whether to only run for 100 epochs and estimate time for 20,000 epochs
num_epochs = 45  # Number of training epochs
input_dim = 30  # Number of features per input row
d_model = 128  # Transformer model dimension
encoder_layers = 4  # Number of encoder layers
decoder_layers = 6   # Number of decoder layers
heads = 8  # Reduced number of attention heads for efficiency
d_ff = 256  # Feedforward network dimension
output_dim = 30  # Number of features per output row
learning_rate = 0.00003  # Learning rate
batch_size = 50  # Batch size for DataLoader
dropout_rate = 0.15  # Dropout rate for the model
synthetic_dir = 'sythtraining'
include_synthetic_training_data = True  # Set to True to include synthetic data
CHECKPOINT_PATH = ''  # Correct path
FAST_DEV_RUN = False  # Set to True to enable fast development run

# Context Encoder Configuration
context_encoder_d_model = 128  # Transformer model dimension for Context Encoder
context_encoder_heads = 8       # Number of attention heads for Context Encoder
# Fine-Tuning Configurations
finetuning_patience = 5
finetuning_max_epochs = 100

class OptunaConfig:
    def __init__(self):
        print("DEBUG: Creating OptunaConfig")
        
        self.n_trials = 1
        self.study_name = "jarc_optimization_v3"  # Updated study name
        self.storage_url = "sqlite:///jarc_optuna.db"
        
        # Expanded hyperparameter ranges
        self.param_ranges = {
            # Model Architecture
            # Model Architecture Parameters
            "d_model": (32, 2048, 16),      # Expanded from 32 to 512 with step 16
            "heads": [2, 4, 8, 16, 32, 64],    # Fixed list of choices
            "encoder_layers": (1, 12),     # Expanded upper bound to 12 layers
            "decoder_layers": (1, 12),     # Expanded upper bound to 12 layers
            "d_ff": (64, 2048, 64),        # Expanded from 64 to 1024 with step 64
            "dropout": (0.01, 0.7),        # Expanded dropout rate range
            
            # Context Encoder Parameters (actually used in model)
            "context_encoder_d_model": (32, 512, 32),  # Expanded from 32 to 512 with step 32
            "context_encoder_heads": [2, 4, 8],   # Fixed list of choices
            
            # Training Parameters
            "batch_size": (8, 512),                   # Expanded batch size range from 8 to 256
            "learning_rate": (1e-6, 1e-1),            # Expanded learning rate range from 1e-6 to 1e-1
            
            # **New:** Add max_epochs range
            "max_epochs": (4, 5, 1),       # Updated range
            "gradient_clip_val": (0.0, 5.0),  # Add gradient_clip_val range
        }
        
        # Pruning Configuration
        self.pruning = {
            "n_warmup_steps": 5,           # Number of trials before pruning starts
            "n_startup_trials": 10,        # Number of trials before using pruning
            "patience": 2,                 # Number of epochs without improvement before pruning
            "pruning_percentile": 25,      # Percentile for pruning
        }
        
        print("DEBUG: OptunaConfig created with ranges:", self.param_ranges)

class LoggingConfig:
    def __init__(self):
        self.level = "INFO"  # Default logging level
        self.debug_mode = False
        logging.debug("Initializing LoggingConfig")  # Debugging statement

class FineTuningConfig:
    def __init__(self):
        self.mode = "all"  # "all" or "random"
        self.num_random_tasks = 1
        self.save_dir = "finetuning_results"
        self.max_epochs = 100
        self.learning_rate = 1e-5
        self.patience = 5


class SchedulerConfig:
    def __init__(self):
        self.use_cosine_annealing = True  # Set to True to enable CosineAnnealingWarmRestarts
        self.T_0 = 10  # Number of epochs for the first restart
        self.T_mult = 2  # Multiplicative factor to increase T_0 after each restart
        self.eta_min = 6e-7  # Minimum learning rate during annealing
class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig(
            batch_size=batch_size,
            learning_rate=learning_rate,
            include_synthetic_training_data=include_synthetic_training_data,
            num_epochs=num_epochs,
            device_choice=device_choice,
            precision=precision,
            fast_dev_run=FAST_DEV_RUN,
            train_from_checkpoint=TRAIN_FROM_CHECKPOINT
        )
        self.logging = LoggingConfig()
        self.finetuning = FineTuningConfig()
        self.optuna = OptunaConfig()
        self.scheduler = SchedulerConfig()
        self.use_best_params = False  # Whether to load and use best parameters from Optuna study

    def validate_config(self):
        """Validate configuration values"""
        assert self.training.batch_size > 0, "Batch size must be positive"
        assert self.training.learning_rate > 0, "Learning rate must be positive"
        assert self.training.device_choice in ['cpu', 'gpu'], "device_choice must be 'cpu' or 'gpu'"
        assert self.training.precision in [16, 32, 64, 'bf16'], "Invalid precision value"

    def __init__(self, model=None, training=None, device_choice=None):
        self.model = model if model is not None else ModelConfig()
        
        # Update device_choice to use 'gpu' or 'cpu'
        if device_choice is None:
            device_choice = 'gpu' if torch.cuda.is_available() else 'cpu'
        
        if training is not None:
            self.training = training
        else:
            self.training = TrainingConfig(
                batch_size=batch_size,
                learning_rate=learning_rate,
                include_synthetic_training_data=include_synthetic_training_data,
                num_epochs=num_epochs,
                device_choice=device_choice,
                precision=precision,
                fast_dev_run=FAST_DEV_RUN,
                train_from_checkpoint=TRAIN_FROM_CHECKPOINT
            )
        
        self.logging = LoggingConfig()
        self.finetuning = FineTuningConfig()
        self.validate_config()
        self.optuna = OptunaConfig()
        self.scheduler = SchedulerConfig()  # Add scheduler configuration
        
        # Control for using best parameters
        self.use_best_params = False  # Whether to load and use best parameters from Optuna study
    
        logging.debug("Config initialized with:")
        logging.debug(f"  - Logging level: {self.logging.level}")
        logging.debug(f"  - Fine-tuning mode: {self.finetuning.mode}")
        logging.debug(f"  - Device: {device_choice}")
        
        # Initialize model-specific attributes from ModelConfig
        self.input_dim = self.model.input_dim
        self.d_model = self.model.d_model
        self.encoder_layers = self.model.encoder_layers
        self.decoder_layers = self.model.decoder_layers
        self.heads = self.model.heads
        self.d_ff = self.model.d_ff
        self.output_dim = self.model.output_dim
        self.dropout = self.model.dropout
        self.context_encoder_d_model = self.model.context_encoder_d_model
        self.context_encoder_heads = self.model.context_encoder_heads
        self.checkpoint_path = self.model.checkpoint_path  # Path to checkpoint file for resuming training
    def validate_config(self):
        """Validate configuration values"""
        assert self.training.batch_size > 0, "Batch size must be positive"
        assert self.training.learning_rate > 0, "Learning rate must be positive"
        assert self.training.device_choice in ['cpu', 'gpu'], "device_choice must be 'cpu' or 'gpu'"
        assert self.training.precision in [16, 32, 64, 'bf16'], "Invalid precision value"
