
import argparse
import torch
from train import TransformerTrainer, prepare_data
import config
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Run Script to Start Training using Config Parameters

parser = argparse.ArgumentParser(description="Train Transformer Model")
parser.add_argument(
    '--checkpoint',
    type=str,
    default=None,
    help='Path to the checkpoint to resume training from'
)
args = parser.parse_args()
train_loader, val_loader = prepare_data()
if args.checkpoint:
    model = TransformerTrainer.load_from_checkpoint(args.checkpoint)
else:
    model = TransformerTrainer()
    model.model = torch.jit.script(model.model)  # Script the model here
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=1,
    mode="min",
    filename="epoch={epoch}-val_loss={val_loss:.4f}",
    save_last=True,  # Saves the last checkpoint with the suffix 'last'
)
early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")
trainer = Trainer(
    max_epochs=config.num_epochs,
    callbacks=[checkpoint_callback, early_stop_callback],
    devices=1,  # Use 1 CPU
    accelerator='cpu'  # Explicitly set to use CPU
)
trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint)
