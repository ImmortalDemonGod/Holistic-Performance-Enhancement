
import torch
from train import TransformerTrainer, prepare_data
from config import *
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Run Script to Start Training using Config Parameters

if __name__ == '__main__':
    train_loader, val_loader = prepare_data()
    model = TransformerTrainer()
    model.model = torch.jit.script(model.model)  # Script the model here
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    trainer = Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        devices=1,  # Use 1 CPU
        accelerator='cpu'  # Explicitly set to use CPU
    )
    trainer.fit(model, train_loader, val_loader)
