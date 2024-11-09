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
        
        logger.info("Initializing TransformerTrainer")

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

        # Modified criterion to handle padding separately
        self.criterion = nn.CrossEntropyLoss(ignore_index=10)  # Ignore padding tokens
        self.learning_rate = config.training.learning_rate  # Access learning_rate from config.training

    def debug_batch(self, batch, name=""):
        """Helper function to debug batch data"""
        src, tgt, ctx_input, ctx_output, task_ids = batch
        logger.info(f"\nDEBUG - Batch {name}:")
        logger.info(f"Source shape: {src.shape}, dtype: {src.dtype}")
        logger.info(f"Target shape: {tgt.shape}, dtype: {tgt.dtype}")
        logger.info(f"Context input shape: {ctx_input.shape if ctx_input is not None else None}")
        logger.info(f"Context output shape: {ctx_output.shape if ctx_output is not None else None}")
        logger.info(f"Task IDs: {task_ids}")
        return batch

    def forward(self, src, tgt, ctx_input=None, ctx_output=None):
        return self.model(src, tgt, ctx_input, ctx_output)

    def _compute_loss(self, y_hat, tgt):
        """
        Compute loss while maintaining 2D grid structure
        Args:
            y_hat: [batch, H, W, num_classes] - Model predictions
            tgt: [batch, H, W] - Target grid
        Returns:
            loss: scalar loss value
        """
        batch_size = y_hat.size(0)
        H = W = self.config.model.seq_len

        # Debug shapes and dtypes before reshaping
        logger.info(f"\nDEBUG - Loss computation input:")
        logger.info(f"y_hat shape: {y_hat.shape}, dtype: {y_hat.dtype}")
        logger.info(f"target shape: {tgt.shape}, dtype: {tgt.dtype}")

        # Reshape while maintaining grid structure
        y_hat_flat = y_hat.reshape(-1, y_hat.size(-1))  # [batch*H*W, num_classes]
        tgt_flat = tgt.reshape(-1).long()  # Convert to long type for CrossEntropyLoss

        # Debug shapes after reshaping
        logger.info(f"\nDEBUG - After reshaping:")
        logger.info(f"y_hat_flat shape: {y_hat_flat.shape}, dtype: {y_hat_flat.dtype}")
        logger.info(f"tgt_flat shape: {tgt_flat.shape}, dtype: {tgt_flat.dtype}")
        logger.info(f"Target value range: min={tgt_flat.min().item()}, max={tgt_flat.max().item()}")

        return self.criterion(y_hat_flat, tgt_flat)

    def _compute_accuracy(self, y_hat, tgt):
        """
        Compute grid accuracy metrics
        Args:
            y_hat: [batch, H, W, num_classes] - Model predictions
            tgt: [batch, H, W] - Target grid
        Returns:
            dict containing accuracy metrics
        """
        with torch.no_grad():
            # Get predicted classes
            predictions = torch.argmax(y_hat, dim=-1)  # [batch, H, W]
            
            # Debug predictions
            logger.info(f"\nDEBUG - Accuracy computation:")
            logger.info(f"Predictions shape: {predictions.shape}, dtype: {predictions.dtype}")
            logger.info(f"Target shape: {tgt.shape}, dtype: {tgt.dtype}")
            
            # Create mask for non-padding elements
            valid_mask = (tgt != 10)
            
            # Cell-wise accuracy
            correct_cells = (predictions == tgt) & valid_mask
            cell_accuracy = correct_cells.float().sum() / valid_mask.float().sum()
            
            # Full grid accuracy (only count grid as correct if all non-padded cells match)
            grid_matches = torch.all(correct_cells | ~valid_mask, dim=(1,2))
            grid_accuracy = grid_matches.float().mean()
            
            # Debug accuracy metrics
            logger.info(f"Cell accuracy: {cell_accuracy.item():.4f}")
            logger.info(f"Grid accuracy: {grid_accuracy.item():.4f}")
            
            return {
                'cell_accuracy': cell_accuracy,
                'grid_accuracy': grid_accuracy
            }

    def training_step(self, batch, batch_idx):
        # Debug incoming batch
        batch = self.debug_batch(batch, "training")
        src, tgt, ctx_input, ctx_output, task_ids = batch
        
        # Convert target to long type
        tgt = tgt.long()
        
        y_hat = self(src, tgt, ctx_input, ctx_output)
        y_hat = self.model.dequant(y_hat)  # Dequantize outputs before computing metrics
        
        # Compute loss maintaining grid structure
        loss = self._compute_loss(y_hat, tgt)
        
        # Compute and log accuracies
        accuracies = self._compute_accuracy(y_hat, tgt)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_cell_accuracy', accuracies['cell_accuracy'], prog_bar=True)
        self.log('train_grid_accuracy', accuracies['grid_accuracy'], prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Debug incoming batch
        batch = self.debug_batch(batch, "validation")
        src, tgt, ctx_input, ctx_output, task_ids = batch
        
        # Convert target to long type
        tgt = tgt.long()
        
        y_hat = self(src, tgt, ctx_input, ctx_output)
        
        # Compute loss maintaining grid structure
        loss = self._compute_loss(y_hat, tgt)
        
        # Compute and log accuracies
        accuracies = self._compute_accuracy(y_hat, tgt)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_cell_accuracy', accuracies['cell_accuracy'], prog_bar=True)
        self.log('val_grid_accuracy', accuracies['grid_accuracy'], prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        # Debug incoming batch
        batch = self.debug_batch(batch, "test")
        src, tgt, ctx_input, ctx_output, task_ids = batch
        
        # Convert target to long type
        tgt = tgt.long()
        
        y_hat = self(src, tgt, ctx_input, ctx_output)
        
        # Compute and log accuracies
        accuracies = self._compute_accuracy(y_hat, tgt)
        
        # Log metrics
        self.log('test_cell_accuracy', accuracies['cell_accuracy'], prog_bar=True)
        self.log('test_grid_accuracy', accuracies['grid_accuracy'], prog_bar=True)
        
        return accuracies['grid_accuracy']

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