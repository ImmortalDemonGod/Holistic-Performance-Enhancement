from Utils.optuna.best_params_manager import BestParamsManager
import argparse
import logging
import os
import torch
from Utils.model_factory import create_transformer_trainer
from Utils.data_preparation import prepare_data
from pathlib import Path
from Utils.model_factory import create_transformer_trainer
from config import Config
import torch.quantization
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Instantiate the Config class
cfg = Config()
logging.basicConfig(level=logging.INFO)
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

    # Print final configuration
    logger.info("Final configuration:")
    logger.info(f"Model parameters: {cfg.model.__dict__}")
    logger.info(f"Training parameters: {cfg.training.__dict__}")

    try:
        # Initialize model
        if cfg.training.train_from_checkpoint and cfg.model.checkpoint_path:
            logger.debug(f"Attempting to load model from checkpoint: {cfg.model.checkpoint_path}")
            checkpoint_path = cfg.model.checkpoint_path
            checkpoint_file = Path(checkpoint_path)
            if checkpoint_file.is_file():
                logger.info(f"Resuming from checkpoint: {checkpoint_path}")
                model = create_transformer_trainer(
                    config=cfg,
                    checkpoint_path=checkpoint_path
                )
            else:
                logger.info("Initializing new model")
                model = create_transformer_trainer(
                    config=cfg,
                    checkpoint_path=None
                )
        else:
            logger.info("Initializing new model")
            model = create_transformer_trainer(
                config=cfg,
                checkpoint_path=None
            )

        # Log device information
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Selected device: {cfg.training.device_choice}")

        return model

    except Exception as e:
        logger.error(f"Error in setup_model_training: {str(e)}")
        raise

    # Setup dynamic quantization
    #model.model = torch.quantization.quantize_dynamic(
    #    model.model,
    #    {torch.nn.Linear, torch.nn.Conv2d},  # Layers to apply dynamic quantization
    #    dtype=torch.qint8  # Quantization data type
    #)
    
    # Only apply scripting if not training from checkpoint
    if not cfg.training.train_from_checkpoint:
        try:
            #model.model = torch.jit.script(model.model)
            logger.info("Successfully applied TorchScript optimization")
        except Exception as e:
            logger.warning(f"Failed to apply TorchScript: {str(e)}")

    return model

if __name__ == '__main__':
    cfg = Config()  # Load config
    model = setup_model_training(cfg)
    
    # Prepare data
    train_loader, val_loader = prepare_data(batch_size=cfg.training.batch_size)

    # Calculate logging frequency
    num_training_batches = len(train_loader)
    log_every_n_steps = min(50, num_training_batches) if num_training_batches > 0 else 1

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            filename="epoch={epoch}-val_loss={val_loss:.4f}",
            save_last=True
        ),
        EarlyStopping(
            monitor="val_loss", 
            patience=10, 
            mode="min"
        )
    ]

    # Configure trainer with resume_from_checkpoint
    trainer_kwargs = {
        "max_epochs": cfg.training.max_epochs,
        "callbacks": callbacks,
        "devices": 1,
        "accelerator": cfg.training.device_choice,
        "precision": cfg.training.precision,
        "fast_dev_run": cfg.training.FAST_DEV_RUN,
        "log_every_n_steps": log_every_n_steps,
        "gradient_clip_val": cfg.training.gradient_clip_val
    }


    trainer = Trainer(**trainer_kwargs)
    
    # Add device memory logging before training
    if torch.cuda.is_available():
        logger.info("Starting training on GPU...")
        logger.info(f"GPU Memory before training: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    else:
        logger.info("Starting training on CPU...")

    # Train model
    trainer.fit(
        model, 
        train_loader, 
        val_loader
    )

    # Add device memory logging after training
    if torch.cuda.is_available():
        logger.info(f"GPU Memory after training: {torch.cuda.memory_allocated()/1e9:.2f} GB")
