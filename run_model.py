from best_params_manager import BestParamsManager
import argparse
from config import Config
import logging
import os
import torch
from train import TransformerTrainer
from config import TRAIN_FROM_CHECKPOINT
from Utils.data_preparation import prepare_data
import config
from torch.quantization import get_default_qconfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Instantiate the Config class
cfg = Config()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments for optional overrides."""
    parser = argparse.ArgumentParser(description='Run the transformer model')
    parser.add_argument('--override-epochs', type=int, help='Override number of epochs in config')
    parser.add_argument('--override-best-params', action='store_true', help='Override use_best_params in config')
    return parser.parse_args()

def setup_model_training(cfg, args=None):
    """Setup model and training configuration."""
    # Check for command line overrides
    if args and args.override_epochs is not None:
        logger.info(f"Overriding epochs from command line: {args.override_epochs}")
        cfg.training.max_epochs = args.override_epochs
    
    if args and args.override_best_params:
        logger.info("Overriding use_best_params from command line")
        cfg.use_best_params = True

    # Apply best parameters if configured
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

    # Initialize model
    if cfg.model.checkpoint_path and os.path.isfile(cfg.model.checkpoint_path):
        logger.info(f"Resuming from checkpoint: {cfg.model.checkpoint_path}")
        model = TransformerTrainer.load_from_checkpoint(cfg.model.checkpoint_path)
    else:
        logger.info("Initializing new model")
        model = TransformerTrainer(
            input_dim=cfg.model.input_dim,
            d_model=cfg.model.d_model,
            encoder_layers=cfg.model.encoder_layers,
            decoder_layers=cfg.model.decoder_layers,
            heads=cfg.model.heads,
            d_ff=cfg.model.d_ff,
            output_dim=cfg.model.output_dim,
            learning_rate=cfg.training.learning_rate,
            include_sythtraining_data=cfg.training.include_sythtraining_data
        )

    return model

    args = parse_arguments()
    
    # Initialize configuration
    cfg = Config()
    
    # Setup model with configuration
    model = setup_model_training(cfg, args)
    
    # Prepare data
    train_loader, val_loader = prepare_data(batch_size=cfg.training.batch_size)

    # Calculate the number of training batches
    num_training_batches = len(train_loader)

    # Dynamically set log_every_n_steps to the smaller of 50 or the number of training batches
    log_every_n_steps = min(50, num_training_batches) if num_training_batches > 0 else 1
    torch.backends.quantized.engine = 'qnnpack'  # Use 'fbgemm' for x86 platforms if needed

    # Validate checkpoint path only if TRAIN_FROM_CHECKPOINT is True
    if TRAIN_FROM_CHECKPOINT:
        if cfg.model.CHECKPOINT_PATH and not os.path.isfile(cfg.model.CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"Checkpoint file not found at {cfg.model.CHECKPOINT_PATH}. Please update config.py with a valid path."
            )

    # Initialize the model
    if TRAIN_FROM_CHECKPOINT and config.CHECKPOINT_PATH:
        logger.info(f"Resuming training from checkpoint: {config.CHECKPOINT_PATH}")
        model = TransformerTrainer.load_from_checkpoint(cfg.model.CHECKPOINT_PATH)
        model = TransformerTrainer.load_from_checkpoint(config.CHECKPOINT_PATH)
    else:
        if TRAIN_FROM_CHECKPOINT and not config.CHECKPOINT_PATH:
            logger.warning("TRAIN_FROM_CHECKPOINT is True but CHECKPOINT_PATH is not set. Starting training from scratch.")
        else:
            logger.info("Starting training from scratch.")
        model = TransformerTrainer(
            input_dim=cfg.model.input_dim,
            d_model=cfg.model.d_model,
            encoder_layers=cfg.model.encoder_layers,
            decoder_layers=cfg.model.decoder_layers,
            heads=cfg.model.heads,
            d_ff=cfg.model.d_ff,
            output_dim=cfg.model.output_dim,
            learning_rate=cfg.training.learning_rate,
            include_sythtraining_data=cfg.training.include_sythtraining_data
        )
    qconfig = get_default_qconfig('qnnpack')  # Use 'fbgemm' if on x86 platforms
    model.model.qconfig = qconfig


    #torch.quantization.prepare(model.model, inplace=True)

    # Commented out TorchScript scripting during training
    # torch.quantization.convert(model.model, inplace=True)
    # model.model = torch.jit.script(model.model)  # Script the model for optimized deployment

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

    # Set up logging statements for clarity
    if TRAIN_FROM_CHECKPOINT and config.CHECKPOINT_PATH:
        logger.info(f"Resuming training from checkpoint: {config.CHECKPOINT_PATH}")
    else:
        if TRAIN_FROM_CHECKPOINT and not config.CHECKPOINT_PATH:
            logger.warning("TRAIN_FROM_CHECKPOINT is True but CHECKPOINT_PATH is not set. Starting training from scratch.")
        else:
            logger.info("Starting training from scratch.")

    # Configure trainer
    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        devices=1,
        accelerator='gpu' if cfg.training.device_choice == 'cuda' else 'cpu',
        precision=cfg.training.precision,
        fast_dev_run=cfg.training.FAST_DEV_RUN,
        log_every_n_steps=log_every_n_steps,
        gradient_clip_val=cfg.training.gradient_clip_val
    )
    
    # Train model
    trainer.fit(
        model, 
        train_loader, 
        val_loader, 
        ckpt_path=cfg.model.checkpoint_path if cfg.model.checkpoint_path else None
    )
