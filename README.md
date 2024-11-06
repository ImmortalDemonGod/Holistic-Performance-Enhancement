# Transformer Model with Custom Sigmoid Activation

This project implements a Transformer model with a custom sigmoid activation function for processing grid-based input data. The model is designed to handle input and output grids ranging from 1x1 to 30x30, with values between 0 and 9.

## Project Structure

- **`config.py`**: Central configuration for model parameters, training settings, and features
- **`transformer_model.py`**: Transformer model implementation with context encoding
- **`train.py`**: Training module and data preparation
- **`run_model.py`**: Training process initialization
- **`evaluate.py`**: Model evaluation tools
- **`validation_finetuning.py`**: Fine-tuning capabilities for specific tasks

## Installation

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

## Configuration

All settings are managed through the `config.py` file. The configuration is divided into several sections:

### Model Configuration
```python
class ModelConfig:
    input_dim = 30
    d_model = 128
    encoder_layers = 2
    decoder_layers = 2
    heads = 8
    d_ff = 256
    dropout_rate = 0.35
    context_encoder_d_model = 128
    context_encoder_heads = 8
```

### Training Configuration
```python
class TrainingConfig:
    batch_size = 50
    learning_rate = 0.00007
    include_synthetic_training_data = False
    max_epochs = 100
    device_choice = 'gpu'  # or 'cpu'
    precision = 32  # 16, 32, 64, or 'bf16'
    gradient_clip_val = 1.0
```

### Fine-tuning Configuration
```python
class FineTuningConfig:
    mode = "all"  # "all" or "random"
    num_random_tasks = 1  # Only used if mode is "random"
```

### Logging Configuration
```python
class LoggingConfig:
    level = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    debug_mode = False
```

## Running the Model

1. Training:
```python
from config import Config
cfg = Config()

# Customize configuration
cfg.training.batch_size = 64
cfg.training.learning_rate = 0.0001
cfg.model.dropout_rate = 0.4

# Run training
python run_model.py
```

2. Fine-tuning:
```python
# Configure fine-tuning
cfg.finetuning.mode = "random"  # Fine-tune random tasks
cfg.finetuning.num_random_tasks = 5  # Number of tasks to fine-tune

# Run fine-tuning
python -m Utils.task_finetuner
```

3. Evaluation:
```python
# Set evaluation checkpoint
cfg.model.checkpoint_path = "path/to/your/checkpoint.ckpt"

# Run evaluation
python evaluate.py
```

## Features

### Hyperparameter Optimization
The project includes Optuna-based hyperparameter optimization:

```python
# Enable best parameters from previous optimization
cfg.use_best_params = True

# Configure optimization
cfg.optuna.n_trials = 100
cfg.optuna.study_name = "your_study_name"
```

### Context Encoding
The model supports context-aware learning through a dedicated context encoder:

```python
# Configure context encoder
cfg.model.context_encoder_d_model = 256
cfg.model.context_encoder_heads = 8
```

### Synthetic Data Support
Enable synthetic data training:

```python
cfg.training.include_synthetic_training_data = True
cfg.training.synthetic_dir = 'synthetic_training'
```

## Advanced Usage

### Custom Training Resume
```python
cfg.training.train_from_checkpoint = True
cfg.model.checkpoint_path = "path/to/checkpoint.ckpt"
```

### Device Configuration
```python
# Auto-select GPU if available
cfg.training.device_choice = 'gpu' if torch.cuda.is_available() else 'cpu'

# Configure precision
cfg.training.precision = 16  # Use mixed precision training
```

### Debugging
```python
# Enable debug mode
cfg.logging.level = "DEBUG"
cfg.logging.debug_mode = True

# Enable fast development run
cfg.training.FAST_DEV_RUN = True
```

## Troubleshooting

- If encountering memory issues:
  ```python
  cfg.training.batch_size = 32  # Reduce batch size
  cfg.training.gradient_clip_val = 0.5  # Add gradient clipping
  ```

- For debugging model behavior:
  ```python
  cfg.logging.level = "DEBUG"
  cfg.training.FAST_DEV_RUN = True
  ```

- For unstable training:
  ```python
  cfg.model.dropout_rate = 0.5  # Increase dropout
  cfg.training.learning_rate = 1e-4  # Adjust learning rate
  ```

## Notes

- All data is padded to a 30x30 grid using padding token of -1
- Support for mixed precision training (16-bit, 32-bit, or 64-bit)
- Automatic device selection between CPU and GPU
- Built-in logging and metrics collection

## License


This project is licensed under the MIT License. See the `LICENSE` file for details.

