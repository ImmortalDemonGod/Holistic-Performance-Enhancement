from Utils.optuna.best_params_manager import BestParamsManager
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
from Utils.model_factory import create_transformer_trainer
from data_module import MyDataModule
from train import TransformerTrainer
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pathlib import Path
from Utils.model_factory import create_transformer_trainer
from config import Config
import torch.quantization
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Create a file handler
file_handler = logging.FileHandler('training.log')
file_handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)
cfg = Config()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
cfg = Config()  # Load config
model = setup_model_training(cfg)  # Use the setup function
data_module = MyDataModule(batch_size=cfg.training.batch_size)

# Initialize the Trainer
trainer = Trainer(
    max_epochs=cfg.training.max_epochs,
    #gpus=1 if cfg.training.device_choice == 'gpu' else 0,
    callbacks=[
        ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min'),
        EarlyStopping(monitor='val_loss', patience=35, mode='min')
    ],
    precision=cfg.training.precision,
    log_every_n_steps=50,
    detect_anomaly=False,
)

# Start training and testing
try:
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
except KeyboardInterrupt:
    print("Training interrupted by user. Exiting gracefully.")
