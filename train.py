# train.py
import pytorch_lightning as pl
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
    def __init__(self, config):
        super().__init__()
        self.model = TransformerModel(
            input_dim=config.input_dim,
            d_model=config.d_model,
            encoder_layers=config.encoder_layers,
            decoder_layers=config.decoder_layers,
            heads=config.heads,
            d_ff=config.d_ff,
            output_dim=config.output_dim,
            dropout_rate=config.dropout,
            context_encoder_d_model=config.context_encoder_d_model,
            context_encoder_heads=config.context_encoder_heads
        )
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = config.training.learning_rate  # Access learning_rate from config.training

    def forward(self, src, tgt, ctx_input=None, ctx_output=None):
        return self.model(src, tgt, ctx_input, ctx_output)

    def training_step(self, batch, batch_idx):
        src, tgt, ctx_input, ctx_output, task_ids = batch
        y_hat = self(src, tgt, ctx_input, ctx_output)
        loss = self.criterion(y_hat.view(-1, 11), tgt.view(-1, 30)[:, 0].long())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt, ctx_input, ctx_output, task_ids = batch
        y_hat = self(src, tgt, ctx_input, ctx_output)
        loss = self.criterion(y_hat.view(-1, 11), tgt.view(-1, 30)[:, 0].long())
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        src, tgt, ctx_input, ctx_output, task_ids = batch
        y_hat = self(src, tgt, ctx_input, ctx_output)
        accuracy = (torch.argmax(y_hat, dim=-1) == tgt.view(-1, 30)[:, 0].long()).float().mean()
        self.log('test_accuracy', accuracy, prog_bar=True)
        return accuracy

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
