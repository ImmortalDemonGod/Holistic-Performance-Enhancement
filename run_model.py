import argparse
import torch
from train import TransformerTrainer, prepare_data
import config
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

if __name__ == '__main__':
    # Argument parsing for optional checkpoint resumption
    parser = argparse.ArgumentParser(description="Train Transformer Model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to the checkpoint to resume training from'
    )
    args = parser.parse_args()

    # Prepare data loaders for training and validation
    train_loader, val_loader = prepare_data()

    # Set the quantization backend
    torch.backends.quantized.engine = 'qnnpack'  # Use 'fbgemm' for x86 platforms if needed

    # Prepare the model for static quantization
    if args.checkpoint:
        model = TransformerTrainer.load_from_checkpoint(args.checkpoint)
    model = TransformerTrainer()
    model.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model.model, inplace=True)

    # Calibrate the model with a few batches of data
    train_loader, val_loader = prepare_data()
    for batch in train_loader:
        model.training_step(batch, 0)  # Run a few batches through the model

    # Convert the model to a quantized version
    torch.quantization.convert(model.model, inplace=True)
    model = TransformerTrainer()
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

    # Configure the Trainer
    trainer = Trainer(
        max_epochs=config.num_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        devices=1,         # Use a single device (CPU)
        accelerator='cpu'  # Set explicitly to CPU
    )

    # Start model training
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint)
