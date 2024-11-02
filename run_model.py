import argparse
import config
import os
import torch
from train import TransformerTrainer, prepare_data
import config
from torch.quantization import get_default_qconfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

if __name__ == '__main__':
    # Argument parsing for optional checkpoint resumption
    parser = argparse.ArgumentParser(description="Train Transformer Model")
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

    # Validate checkpoint path
    if config.CHECKPOINT_PATH and not os.path.isfile(config.CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint file not found at {config.CHECKPOINT_PATH}. Please update config.py with a valid path.")

    # Prepare the model for static quantization
    if config.CHECKPOINT_PATH:
        model = TransformerTrainer.load_from_checkpoint(config.CHECKPOINT_PATH)
    else:
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

    # Configure the Trainer
    trainer = Trainer(
        max_epochs=config.num_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        devices=1,         # Use a single device (CPU)
        accelerator='gpu' if config.device_choice == 'cuda' else 'cpu', # Use GPU if specified
        precision=config.precision,  # Use precision from config
        fast_dev_run=args.fast_dev_run  # Add this line
    )

    # Set ckpt_path based on config.CHECKPOINT_PATH
    ckpt_path = config.CHECKPOINT_PATH if config.CHECKPOINT_PATH else None
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
