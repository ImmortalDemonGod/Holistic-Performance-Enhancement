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

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
    save_weights_only=False  # Ensure full model state is saved
)
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
        self.config = config  # Store the config object

        # Initialize the model first
        # Initialize LoRA modules
        #self.lora_A = nn.Linear(config.model.lora_in_features, config.model.lora_out_features)
        #self.lora_B = nn.Linear(config.model.lora_in_features, config.model.lora_out_features)

        self.model = TransformerModel(
            input_dim=config.model.input_dim,  # Access input_dim through config.model
            seq_len=config.model.seq_len,  # Access seq_len through config.model
            d_model=config.d_model,
            encoder_layers=config.encoder_layers,
            decoder_layers=config.decoder_layers,
            heads=config.heads,
            d_ff=config.d_ff,
            output_dim=config.output_dim,
            dropout_rate=config.dropout,
            context_encoder_d_model=config.context_encoder_d_model,
            context_encoder_heads=config.context_encoder_heads,
            checkpoint_path=config.model.checkpoint_path,  # Ensure checkpoint_path is passed
            use_lora=config.model.use_lora,  # Ensure use_lora is passed
            lora_rank=config.model.lora_rank  # Pass lora_rank here
        )

        self.dropout = self.model.dropout  # Expose dropout attribute

        # Enable QAT
        self.model.train()  # Ensure model is in training mode for QAT
        self.model = torch.quantization.prepare_qat(self.model)  # Prepare model for QAT

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = config.training.learning_rate  # Access learning_rate from config.training

    def forward(self, src, tgt, ctx_input=None, ctx_output=None):
        return self.model(src, tgt, ctx_input, ctx_output)

    def training_step(self, batch, batch_idx):
        src, tgt, ctx_input, ctx_output, task_ids = batch
        y_hat = self(src, tgt, ctx_input, ctx_output)
        y_hat = self.model.dequant(y_hat)  # Dequantize outputs before computing metrics
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        if self.config.scheduler.use_cosine_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.config.scheduler.T_0,
                T_mult=self.config.scheduler.T_mult,
                eta_min=self.config.scheduler.eta_min
            )
            return [optimizer], [scheduler]
        else:
            return optimizer
