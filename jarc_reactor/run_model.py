import logging
import sys
from pytorch_lightning.callbacks import ModelCheckpoint

# Initialize logging before importing other modules
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create console handler for stdout
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create file handler for logging to a file
file_handler = logging.FileHandler('training.log')
file_handler.setLevel(logging.INFO)

# Define logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the root logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Redirect stdout and stderr to the logger
class StreamToLogger(object):
    """
    Redirect writes to a logger.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)
from jarc_reactor.optuna.best_params_manager import BestParamsManager
import signal
import sys

def signal_handler(sig, frame):
    print("KeyboardInterrupt received. Exiting immediately.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
import torch
import logging
import os
import torch
from jarc_reactor.data.data_module import MyDataModule
from jarc_reactor.utils.train import TransformerTrainer
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pathlib import Path
from jarc_reactor.utils.model_factory import create_transformer_trainer
from jarc_reactor.config import Config
import torch.quantization
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

cfg = Config()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a file handler
file_handler = logging.FileHandler('training.log')
file_handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def setup_model_training(cfg):
    """Setup model and training configuration."""
    if cfg.use_best_params:
        logger.info("Loading best parameters from Optuna study...")
        params_manager = BestParamsManager(
            storage_url=cfg.optuna.storage_url,
            study_name=cfg.optuna.study_name
        )
        if params_manager.update_config(cfg):
            logger.info("Successfully applied best parameters")
        else:
            logger.warning("Failed to load best parameters, using existing configuration")

    # Print final configuration after attempting to load best parameters
    logger.info("Final configuration after attempting to load best parameters:")
    logger.info(f"Model parameters: {cfg.model.__dict__}")
    logger.info(f"Training parameters: {cfg.training.__dict__}")
    logger.info("Final configuration:")
    logger.info(f"Model parameters: {cfg.model.__dict__}")
    logger.info(f"Training parameters: {cfg.training.__dict__}")

    try:
        # Log the checkpoint path and attempt to load the model
        logger.info(f"Attempting to load model from checkpoint: {cfg.model.checkpoint_path}")
        if cfg.training.train_from_checkpoint and cfg.model.checkpoint_path:
            try:
                model = TransformerTrainer.load_from_checkpoint(cfg.model.checkpoint_path, config=cfg)
                logger.info("Model loaded successfully from checkpoint.")
            except Exception as e:
                logger.error(f"Failed to load model from checkpoint: {str(e)}")
                model = TransformerTrainer(config=cfg)  # Fallback to initializing a new model
        else:
            model = TransformerTrainer(config=cfg)

        # Verify checkpoint content
        if cfg.training.train_from_checkpoint and cfg.model.checkpoint_path:
            try:
                checkpoint = torch.load(cfg.model.checkpoint_path)
                logger.info(f"Checkpoint keys: {checkpoint.keys()}")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
        if cfg.training.train_from_checkpoint and cfg.model.checkpoint_path:
            try:
                checkpoint = torch.load(cfg.model.checkpoint_path)
                logger.info(f"Checkpoint keys: {checkpoint.keys()}")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")

        # Log device information
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Selected device: {cfg.training.device_choice}")

        return model

    except Exception as e:
        logger.error(f"Error in setup_model_training: {str(e)}")
        raise

"""
import os

# Check if the checkpoint file exists
checkpoint_path = '/Users/larryf/Desktop/Jarc_Cur/epoch=97-step=42826.ckpt'
print(f"Checking for checkpoint at: {checkpoint_path}")
if os.path.isfile(checkpoint_path):
    print("Checkpoint file exists.")
else:
    print("Checkpoint file does not exist.")
    # List the contents of the checkpoint directory
    checkpoint_dir = '/absolute/path/to/checkpoints/'
    print("Files in checkpoint directory:")
    for file in os.listdir(checkpoint_dir):
        print(file)

"""
# Ensure the checkpoint path is correct
import os

# Define the checkpoint directory
checkpoint_dir = 'checkpoints'

# Check if the checkpoint directory exists
if not os.path.exists(checkpoint_dir):
    try:
        os.makedirs(checkpoint_dir)
        logger.info(f"Created checkpoint directory: {checkpoint_dir}")
    except Exception as e:
        logger.error(f"Failed to create checkpoint directory: {str(e)}")

# Verify that the checkpoint directory is writable
if not os.access(checkpoint_dir, os.W_OK):
    logger.error(f"Checkpoint directory '{checkpoint_dir}' is not writable")

cfg = Config()  # Load config
model = setup_model_training(cfg)  # Use the setup function
data_module = MyDataModule(batch_size=cfg.training.batch_size)

# Set up the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath='lightning_logs/checkpoints/',  # Updated directory path
    filename='model-step={step}-val_loss={val_loss:.4f}',
    save_top_k=1,                        # Keep only the 2 best models
    monitor='val_loss',
    mode='min',
    save_weights_only=False,              # Only save weights to reduce file size and save time
    every_n_epochs=1,                    # Save once per epoch instead of every step
    save_last=True,                     # Don't save the last checkpoint
    verbose=True
)

# Add debug logging for checkpoint configuration
logger.info(f"Checkpoint callback configured to save to: {checkpoint_callback.dirpath}")

# Initialize the Trainer
trainer = Trainer(
    max_epochs=cfg.training.max_epochs,
    enable_progress_bar=True,  # Added to maintain progress visualization
    callbacks=[checkpoint_callback, EarlyStopping(monitor='val_loss', patience=35, mode='min')],
    precision=cfg.training.precision,
    log_every_n_steps=50,
    detect_anomaly=False,
    enable_checkpointing=True,
    default_root_dir="lightning_logs"
)

# Start training and testing
try:
    trainer.fit(model, data_module)
    
    # Debug logging for checkpoints
    if trainer.checkpoint_callback.best_model_path:
        logger.info(f"Best model saved to: {trainer.checkpoint_callback.best_model_path}")
    else:
        logger.warning("No checkpoint was saved during training")
        logger.info(f"Current val_loss: {trainer.callback_metrics.get('val_loss', None)}")
    
    trainer.test(model, data_module)
except KeyboardInterrupt:
    print("Training interrupted by user. Exiting gracefully.")
