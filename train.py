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
        """
        Test step with shape debugging and proper reshaping
        """
        src, tgt, ctx_input, ctx_output, task_ids = batch
        
        # Debug input shapes
        logging.debug(f"Test step input shapes:")
        logging.debug(f"src: {src.shape}")
        logging.debug(f"tgt: {tgt.shape}")
        logging.debug(f"ctx_input: {ctx_input.shape}")
        logging.debug(f"ctx_output: {ctx_output.shape}")
        
        try:
            # Forward pass
            y_hat = self(src, tgt, ctx_input, ctx_output)
            logging.debug(f"Model output (y_hat) shape: {y_hat.shape}")
            
            # Get predictions
            predictions = y_hat.argmax(dim=-1)  # [batch_size, seq_len]
            logging.debug(f"Raw predictions shape: {predictions.shape}")
            
            # Reshape target to match predictions
            batch_size = tgt.size(0)
            tgt_flat = tgt.reshape(batch_size, -1)  # Flatten height and width dimensions
            predictions = predictions.reshape(batch_size, -1)  # Ensure predictions match target shape
            
            logging.debug(f"Reshaped tensors:")
            logging.debug(f"tgt_flat shape: {tgt_flat.shape}")
            logging.debug(f"predictions shape: {predictions.shape}")
            
            # Create mask for valid (non-padding) elements
            valid_mask = tgt_flat != 10  # Assuming 10 is padding value
            valid_preds = predictions[valid_mask]
            valid_targets = tgt_flat[valid_mask]
            
            logging.debug(f"Number of valid elements: {valid_mask.sum().item()}")
            
            # Calculate accuracy
            accuracy = (valid_preds == valid_targets).float().mean()
            logging.debug(f"Calculated accuracy: {accuracy.item():.4f}")
            
            # Log metrics
            self.log('test_accuracy', accuracy, prog_bar=True)
            return accuracy
            
        except Exception as e:
            logging.error(f"Error in test_step:")
            logging.error(f"Input shapes - src: {src.shape}, tgt: {tgt.shape}")
            if 'y_hat' in locals():
                logging.error(f"Model output shape: {y_hat.shape}")
            if 'predictions' in locals():
                logging.error(f"Predictions shape: {predictions.shape}")
            logging.error(f"Error message: {str(e)}")
            raise
    
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
