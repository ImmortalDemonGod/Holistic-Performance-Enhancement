import config
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

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    # Prepare data loaders for training and validation
    train_loader, val_loader = prepare_data()

    # Calculate the number of training batches
    num_training_batches = len(train_loader)

    # Dynamically set log_every_n_steps to the smaller of 50 or the number of training batches
    log_every_n_steps = min(50, num_training_batches) if num_training_batches > 0 else 1
    torch.backends.quantized.engine = 'qnnpack'  # Use 'fbgemm' for x86 platforms if needed

    # Validate checkpoint path only if TRAIN_FROM_CHECKPOINT is True
    if TRAIN_FROM_CHECKPOINT:
        if config.CHECKPOINT_PATH and not os.path.isfile(config.CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"Checkpoint file not found at {config.CHECKPOINT_PATH}. Please update config.py with a valid path."
            )

    # Initialize the model
    if TRAIN_FROM_CHECKPOINT and config.CHECKPOINT_PATH:
        logger.info(f"Resuming training from checkpoint: {config.CHECKPOINT_PATH}")
        model = TransformerTrainer.load_from_checkpoint(config.CHECKPOINT_PATH)
        model = TransformerTrainer.load_from_checkpoint(config.CHECKPOINT_PATH)
    else:
        if TRAIN_FROM_CHECKPOINT and not config.CHECKPOINT_PATH:
            logger.warning("TRAIN_FROM_CHECKPOINT is True but CHECKPOINT_PATH is not set. Starting training from scratch.")
        else:
            logger.info("Starting training from scratch.")
        model = TransformerTrainer(
            input_dim=config.input_dim,
            d_model=config.d_model,
            encoder_layers=config.encoder_layers,
            decoder_layers=config.decoder_layers,
            heads=config.heads,
            d_ff=config.d_ff,
            output_dim=config.output_dim,
            learning_rate=config.learning_rate,
            include_sythtraining_data=config.include_sythtraining_data
        )
    qconfig = get_default_qconfig('qnnpack')  # Use 'fbgemm' if on x86 platforms
    model.model.qconfig = qconfig


    #torch.quantization.prepare(model.model, inplace=True)

    # Convert the model to a quantized version
    torch.quantization.convert(model.model, inplace=True)
    model.model = torch.jit.script(model.model)  # Script the model for optimized deployment

    # Set up model checkpointing and early stopping callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="epoch={epoch}-val_loss={val_loss:.4f}",
        save_last=True  # Saves the last checkpoint with the suffix 'last'
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        patience=10, 
        mode="min"
    )

    # Set up logging statements for clarity
    if TRAIN_FROM_CHECKPOINT and config.CHECKPOINT_PATH:
        logger.info(f"Resuming training from checkpoint: {config.CHECKPOINT_PATH}")
    else:
        if TRAIN_FROM_CHECKPOINT and not config.CHECKPOINT_PATH:
            logger.warning("TRAIN_FROM_CHECKPOINT is True but CHECKPOINT_PATH is not set. Starting training from scratch.")
        else:
            logger.info("Starting training from scratch.")

    # Configure the Trainer with conditional checkpoint resumption
    trainer_kwargs = {
        "max_epochs": config.training.max_epochs,  # **Modified:** Use max_epochs from TrainingConfig
        "callbacks": [checkpoint_callback, early_stop_callback],
        "devices": 1,  # Use a single device
        "accelerator": 'gpu' if config.device_choice == 'cuda' else 'cpu',
        "precision": config.precision,
        "fast_dev_run": config.FAST_DEV_RUN,
        "log_every_n_steps": log_every_n_steps,
    }

    if TRAIN_FROM_CHECKPOINT and config.CHECKPOINT_PATH:
        trainer_kwargs["resume_from_checkpoint"] = config.CHECKPOINT_PATH

    trainer = Trainer(**trainer_kwargs)

    # Set ckpt_path based on config.CHECKPOINT_PATH
    ckpt_path = config.CHECKPOINT_PATH if TRAIN_FROM_CHECKPOINT and config.CHECKPOINT_PATH else None
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
