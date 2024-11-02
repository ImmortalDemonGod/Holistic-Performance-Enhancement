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
    parser.add_argument(
        '--fast_dev_run',
        type=int,
        nargs='?',
        const=True,
        default=False,
        help=(
            'Run a limited number of batches for debugging purposes. '
            'If an integer is provided, runs that number of batches. '
            'If set without a value, runs one batch. '
            'Default is False.'
        )
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
    qconfig = torch.quantization.QConfig(
        activation=torch.quantization.MinMaxObserver.with_args(quant_min=0, quant_max=255),
        weight=torch.quantization.MinMaxObserver.with_args(quant_min=0, quant_max=255)
    )
    model.model.qconfig = qconfig
    torch.quantization.prepare(model.model, inplace=True)

    # Calibrate the model with a few batches of data without manually calling training_step
    for batch in train_loader:
        src, tgt, _ = batch
        _ = model(src, tgt)  # Forward pass to run observers

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
        accelerator='gpu' if config.device_choice == 'cuda' else 'cpu', # Use GPU if specified
        precision=config.precision,  # Use precision from config
        fast_dev_run=args.fast_dev_run  # Add this line
    )

    # Start model training
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint)
