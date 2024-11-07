# train.py
import logging
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformer_model import TransformerModel
import torch.nn.functional as F
import os
import json
from Utils.padding_utils import pad_to_fixed_size
import torch
from Utils.data_preparation import prepare_data

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerTrainer(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        d_model,
        encoder_layers,
        decoder_layers,
        heads,
        d_ff,
        output_dim,
        learning_rate,
        dropout: Optional[float] = None,
        context_encoder_d_model: Optional[int] = None,
        context_encoder_heads: Optional[int] = None,
        include_synthetic_training_data: Optional[bool] = None,
        config=None,  # Add config parameter
        **kwargs
    ):
        super(TransformerTrainer, self).__init__()

        self.save_hyperparameters()

        # Assign config if provided
        self.config = config
        self.input_dim = self.hparams.input_dim
        self.d_model = self.hparams.d_model
        self.encoder_layers = self.hparams.encoder_layers
        self.decoder_layers = self.hparams.decoder_layers
        self.heads = self.hparams.heads
        self.d_ff = self.hparams.d_ff
        self.output_dim = self.hparams.output_dim
        self.learning_rate = self.hparams.learning_rate
        self.include_synthetic_training_data = self.hparams.include_synthetic_training_data
        self.dropout = self.hparams.dropout
        self.context_encoder_d_model = self.hparams.context_encoder_d_model
        self.context_encoder_heads = self.hparams.context_encoder_heads

        # Initialize the TransformerModel directly
        self.model = TransformerModel(
            input_dim=self.input_dim,
            d_model=self.d_model,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            heads=self.heads,
            d_ff=self.d_ff,
            output_dim=self.output_dim,
            dropout_rate=self.dropout,
            context_encoder_d_model=self.context_encoder_d_model,
            context_encoder_heads=self.context_encoder_heads
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, src, tgt, ctx_input=None, ctx_output=None):
        logger.debug(f"Forward pass input shapes:")
        logger.debug(f"  src: {src.shape}")
        logger.debug(f"  tgt: {tgt.shape}")
        # Move tensors to the same device as the model
        device = next(self.model.parameters()).device
        logger.debug(f"Model is on device: {device}")
        if ctx_input is not None:
            logger.debug(f"  ctx_input: {ctx_input.shape}")
        if ctx_output is not None:
            logger.debug(f"  ctx_output: {ctx_output.shape}")
        src = src.float().to(device)
        tgt = tgt.float().to(device)
        if ctx_input is not None:
            ctx_input = ctx_input.float().to(device)
        if ctx_output is not None:
            ctx_output = ctx_output.float().to(device)

        return self.model(src, tgt, ctx_input, ctx_output)

    def training_step(self, batch, batch_idx):
        src, tgt, ctx_input, ctx_output, task_ids = batch
        y_hat = self(src, tgt, ctx_input, ctx_output)

        # Debugging: Print shapes of y_hat and tgt
        # print(f"y_hat shape: {y_hat.shape}, tgt shape: {tgt.shape}")

        # Reshape y_hat to match the target's shape
        y_hat = y_hat.view(-1, 11)
        tgt = tgt.view(-1, 30)[:, 0].long()
        # Mask out padding values in the target
        valid_indices = tgt != -1
        y_hat = y_hat[valid_indices]
        tgt = tgt[valid_indices]

        loss = self.criterion(y_hat, tgt)
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt, ctx_input, ctx_output, task_ids = batch
        y_hat = self(src, tgt, ctx_input, ctx_output)

        # Debugging: Print shapes of y_hat and tgt
        # print(f"y_hat shape: {y_hat.shape}, tgt shape: {tgt.shape}")

        # Reshape y_hat to match the target's shape
        y_hat = y_hat.view(-1, 11)
        tgt = tgt.view(-1, 30)[:, 0].long()
        # Convert logits to class predictions using argmax
        predictions = torch.argmax(y_hat, dim=-1)  # Shape: [batch_size * 30]

        # Compute loss
        loss = self.criterion(y_hat, tgt)
        self.log('val_loss', loss, prog_bar=True)

        return {'val_loss': loss, 'predictions': predictions, 'targets': tgt}



    def test_step(self, batch, batch_idx):
        src, tgt, ctx_input, ctx_output, task_ids = batch
        y_hat = self(src, tgt, ctx_input, ctx_output)

        # Compute accuracy (modify according to your specific task)
        threshold = 0.1
        correct = (torch.abs(y_hat - tgt) < threshold).float()
        accuracy = correct.mean()

        self.log("test_accuracy", accuracy, prog_bar=True)
        return accuracy
