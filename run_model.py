
import torch
from train import TransformerTrainer, prepare_data
from config import *
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Run Script to Start Training using Config Parameters

if __name__ == '__main__':
    # Prepare Data
    train_loader, val_loader = prepare_data()

    # Initialize Model and Trainer
    model = TransformerTrainer()

    # Define Callbacks
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    
    # Initialize Trainer with Configuration Parameters
    trainer = Trainer(max_epochs=num_epochs, callbacks=[checkpoint_callback, early_stop_callback], devices="auto", accelerator="auto")

    # Train the Model
    trainer.fit(model, train_loader, val_loader)
