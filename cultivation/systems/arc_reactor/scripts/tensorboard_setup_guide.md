# TensorBoard Setup and Usage Guide

## Overview

This guide explains how to set up and use TensorBoard for viewing PyTorch Lightning training logs in a dedicated Python environment. This setup was created to avoid Python 3.13 compatibility issues with TensorBoard.

## Problem Solved

- **Issue**: TensorBoard has compatibility issues with Python 3.13 (missing `imghdr` module)
- **Solution**: Dedicated Python 3.11 environment with TensorBoard and TensorFlow
- **Benefit**: Clean, isolated environment that works reliably

## Files Created

### 1. `setup_tensorboard_env.sh`
**Purpose**: One-time setup script to create the TensorBoard environment

**What it does**:
- Detects available Python versions (3.10, 3.11, or 3.12)
- Creates a virtual environment `.tensorboard_env/`
- Installs TensorBoard and TensorFlow
- Provides usage instructions

### 2. `run_tensorboard.sh`
**Purpose**: Quick launcher script for TensorBoard

**What it does**:
- Activates the TensorBoard environment
- Launches TensorBoard on http://localhost:6006
- Points to the ARC Reactor training logs

### 3. `.tensorboard_env/` Directory
**Purpose**: Isolated Python 3.11 environment

**Contents**:
- Python 3.11 interpreter
- TensorBoard 2.19.0
- TensorFlow 2.19.0
- All required dependencies

## Quick Start

### First Time Setup
```bash
# Make scripts executable
chmod +x setup_tensorboard_env.sh run_tensorboard.sh

# Run setup (only needed once)
./setup_tensorboard_env.sh
```

### Launch TensorBoard
```bash
# Start TensorBoard
./run_tensorboard.sh

# Open browser to http://localhost:6006
```

### Stop TensorBoard
```bash
# Press Ctrl+C in the terminal, or
pkill -f tensorboard
```

## Manual Usage

If you prefer manual control:

```bash
# Activate environment
source .tensorboard_env/bin/activate

# Start TensorBoard
tensorboard --logdir=cultivation/systems/arc_reactor/logs/training/lightning_logs --port=6006

# Deactivate when done
deactivate
```

## Understanding Your Training Logs

### Log Structure
```
cultivation/systems/arc_reactor/logs/training/lightning_logs/
├── version_0/          # First training run
├── version_1/          # Second training run
└── version_2/          # Third training run
```

Each version contains:
- `events.out.tfevents.*` - TensorBoard event files
- `hparams.yaml` - Hyperparameters (if logged)
- `metrics.csv` - CSV export of metrics (if available)

### Key Metrics to Monitor

#### 1. **Loss Metrics**
- `train_loss` - Training loss (should decrease)
- `val_loss_epoch` - Validation loss per epoch
- `val_loss_step` - Validation loss per step

#### 2. **Accuracy Metrics**
- `train_cell_accuracy` - Training cell-level accuracy
- `train_grid_accuracy` - Training grid-level accuracy
- `val_cell_accuracy` - Validation cell-level accuracy
- `val_grid_accuracy` - Validation grid-level accuracy
- `test_cell_accuracy` - Test cell-level accuracy
- `test_grid_accuracy` - Test grid-level accuracy

#### 3. **Training Progress**
- `epoch` - Current epoch number

### Interpreting Results

#### Good Training Signs:
- ✅ Training loss decreases steadily
- ✅ Validation loss decreases (may plateau)
- ✅ Gap between train/val accuracy is reasonable
- ✅ Test accuracy is close to validation accuracy

#### Warning Signs:
- ⚠️ Training loss decreases but validation loss increases (overfitting)
- ⚠️ Large gap between training and validation accuracy
- ⚠️ Training accuracy = 100% but test accuracy is much lower

#### Your ARC Reactor Results Summary:
- **Version 1**: Best performance (83.5% test cell accuracy, 33.7% grid accuracy)
- **Version 2**: Lower performance (68.2% test cell accuracy, 15.9% grid accuracy)
- **Version 0**: Minimal training (likely early stop or test run)

## TensorBoard Interface Guide

### Main Tabs

#### 1. **Scalars Tab** (Most Important)
- View all numeric metrics over time
- Compare multiple training runs
- Smooth curves with the smoothing slider
- Download plots as images

#### 2. **Images Tab**
- View image data (if logged)
- Useful for seeing model predictions vs ground truth

#### 3. **Graphs Tab**
- Model architecture visualization
- Computational graph structure

#### 4. **Distributions/Histograms**
- Weight and gradient distributions
- Useful for debugging training dynamics

### Useful Features

#### Smoothing
- Use the smoothing slider (0-0.99) to reduce noise in curves
- Higher values = smoother curves
- Good for seeing overall trends

#### Run Selection
- Toggle different training runs on/off
- Compare performance across experiments
- Use different colors for easy identification

#### Time Range Selection
- Zoom into specific training periods
- Drag to select time ranges
- Double-click to reset zoom

#### Download Options
- Download individual plots as PNG/SVG
- Export data as CSV
- Share specific views with team members

## Troubleshooting

### TensorBoard Won't Start
```bash
# Check if environment exists
ls -la .tensorboard_env/

# If missing, run setup again
./setup_tensorboard_env.sh

# Check for port conflicts
lsof -i :6006
```

### No Data Visible
```bash
# Verify log directory exists
ls -la cultivation/systems/arc_reactor/logs/training/lightning_logs/

# Check for event files
find cultivation/systems/arc_reactor/logs/training/lightning_logs/ -name "events.out.tfevents.*"
```

### Environment Issues
```bash
# Recreate environment
rm -rf .tensorboard_env/
./setup_tensorboard_env.sh
```

### Port Already in Use
```bash
# Use different port
source .tensorboard_env/bin/activate
tensorboard --logdir=cultivation/systems/arc_reactor/logs/training/lightning_logs --port=6007

# Or kill existing TensorBoard
pkill -f tensorboard
```

## Alternative Log Viewing

If TensorBoard isn't working, you can use the Python scripts:

### 1. Basic Log Viewer
```bash
python view_training_logs.py
```
Shows basic information about training runs.

### 2. Metrics Extractor
```bash
python extract_training_metrics.py
```
Extracts metrics to CSV and creates matplotlib plots.

## Advanced Usage

### Custom Log Directory
```bash
source .tensorboard_env/bin/activate
tensorboard --logdir=/path/to/your/logs --port=6006
```

### Multiple Experiments
```bash
# Compare different experiments
tensorboard --logdir_spec=exp1:/path/to/exp1,exp2:/path/to/exp2
```

### Remote Access
```bash
# Allow access from other machines
tensorboard --logdir=logs --host=0.0.0.0 --port=6006
```

## Maintenance

### Updating TensorBoard
```bash
source .tensorboard_env/bin/activate
pip install --upgrade tensorboard tensorflow
```

### Cleaning Old Logs
```bash
# Remove old training runs (be careful!)
rm -rf cultivation/systems/arc_reactor/logs/training/lightning_logs/version_*
```

### Environment Cleanup
```bash
# Remove environment completely
rm -rf .tensorboard_env/
```

## Integration with Training Scripts

To ensure your PyTorch Lightning training logs work with this setup:

### 1. Use TensorBoard Logger
```python
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(
    save_dir="cultivation/systems/arc_reactor/logs/training",
    name="lightning_logs"
)

trainer = Trainer(logger=logger)
```

### 2. Log Important Metrics
```python
# In your LightningModule
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    self.log('train_loss', loss)
    return loss

def validation_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    accuracy = self.compute_accuracy(batch)
    self.log('val_loss', loss)
    self.log('val_accuracy', accuracy)
```

### 3. Log Hyperparameters
```python
# Save hyperparameters for comparison
self.save_hyperparameters()
```

## Best Practices

1. **Regular Monitoring**: Check TensorBoard during training to catch issues early
2. **Experiment Naming**: Use descriptive names for different training runs
3. **Metric Selection**: Focus on the most important metrics for your task
4. **Comparison**: Always compare new experiments with previous baselines
5. **Documentation**: Keep notes about what changes were made between experiments

## Security Notes

- TensorBoard runs on localhost by default (secure)
- Don't expose TensorBoard to public networks without authentication
- Be careful when sharing TensorBoard URLs (they may contain sensitive data)

## Resources

- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [PyTorch Lightning TensorBoard Integration](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#tensorboard)
- [TensorBoard Scalars Dashboard](https://www.tensorflow.org/tensorboard/scalars_and_keras)

---

**Created**: June 2025  
**Last Updated**: June 2025  
**Environment**: Python 3.11, TensorBoard 2.19.0, macOS ARM64
