
import torch
from train import DiffTransformerTrainer, prepare_data
from config import *
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

if __name__ == '__main__':
    train_loader, val_loader = prepare_data()
    model = DiffTransformerTrainer()
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    trainer = Trainer(max_epochs=num_epochs, callbacks=[checkpoint_callback, early_stop_callback], devices="auto", accelerator="auto")
    trainer.fit(model, train_loader, val_loader)
