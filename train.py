from config import include_sythtraining_data
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformer_model import TransformerModel
from config import *
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
        include_sythtraining_data,
    ):
        super(TransformerTrainer, self).__init__()
        self.save_hyperparameters()
        self.model = TransformerModel(
            input_dim=self.hparams['input_dim'],
            d_model=self.hparams['d_model'],
            encoder_layers=self.hparams['encoder_layers'],
            decoder_layers=self.hparams['decoder_layers'],
            heads=self.hparams['heads'],
            d_ff=self.hparams['d_ff'],
            output_dim=self.hparams['output_dim'],
        )
        self.learning_rate = self.hparams['learning_rate']
        self.device_choice = 'cpu'
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, src, tgt, ctx_input=None, ctx_output=None):
        return self.model(src.to("cpu"), tgt.to("cpu"), ctx_input, ctx_output)
        return self.model(src.to("cpu"), tgt.to("cpu"))

    def training_step(self, batch, batch_idx):
        src, tgt, ctx_input, ctx_output, task_ids = batch  # Unpack all 5 elements
        y_hat = self(src, tgt, ctx_input, ctx_output)

        # Debugging: Print shapes of y_hat and tgt
        print(f"y_hat shape: {y_hat.shape}, tgt shape: {tgt.shape}")

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
        src, tgt, ctx_input, ctx_output, task_ids = batch  # Unpack all 5 elements
        y_hat = self(src, tgt, ctx_input, ctx_output)

        # Debugging: Print shapes of y_hat and tgt
        print(f"y_hat shape: {y_hat.shape}, tgt shape: {tgt.shape}")

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



