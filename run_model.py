
import torch
import torch.multiprocessing as mp
import logging
from train import TransformerTrainer, prepare_data
from config import *
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def setup_logger(rank):
    logger = logging.getLogger(__name__)
    if rank == 0:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)
    return logger

def train(rank, world_size):
    logger = setup_logger(rank)
    # Prepare data
    train_loader, val_loader = prepare_data()
    
    # Initialize the model
    model = TransformerTrainer()
    
    # Print model summary only in the main process
    if rank == 0:
        logger.info(model)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=125, mode="min")
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=125, mode="min")
    
    trainer = Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        devices=1,  # Use 1 CPU per process
        accelerator='cpu',  # Use CPU for training
        log_every_n_steps=10,  # Adjust this value as needed
        enable_progress_bar=(rank == 0)  # Enable progress bar only in main process
    )
    
    # Start training
    trainer.fit(model, train_loader, val_loader)

    if rank == 0:
        logger.info("Training complete.")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # Set the start method for multiprocessing
    world_size = 6  # Number of processes (e.g., number of CPU cores to use)
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
