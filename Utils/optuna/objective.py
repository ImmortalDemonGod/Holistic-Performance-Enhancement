# Utils/optuna/objective.py

import torch
import logging
from dataclasses import dataclass
import optuna
from config import Config
from Utils.data_preparation import prepare_data
from Utils.model_factory import create_transformer_trainer
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import EarlyStopping
from math import ceil

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class TrialMetrics:
    val_loss: float
    train_loss: float
    val_accuracy: float
    memory_used: float = 0.0

    def log_metrics(self):
        """Print all metrics for debugging"""
        logger.debug("Trial Metrics:")
        logger.debug(f"  Val Loss: {self.val_loss:.4f}")
        logger.debug(f"  Train Loss: {self.train_loss:.4f}")
        logger.debug(f"  Val Accuracy: {self.val_accuracy:.4f}")
        logger.debug(f"  Memory Used: {self.memory_used:.2f} MB")

def create_trial_config(trial, base_config):
    """Create a new config with trial-suggested values"""
    try:
        logger.debug("Creating trial config")
        
        # First suggest number of heads (power of 2)
        heads = trial.suggest_categorical("heads", [2, 4, 8, 16, 32])
        
        # Then suggest d_model ensuring it's divisible by heads
        d_model_min = ((32 + heads - 1) // heads) * heads  # Round up to nearest multiple of heads
        d_model_max = (2048 // heads) * heads  # Round down to nearest multiple of heads
        d_model = trial.suggest_int("d_model", d_model_min, d_model_max, step=heads)
        
        logger.debug(f"Selected heads={heads}, d_model={d_model}")
        assert d_model % heads == 0, f"d_model ({d_model}) must be divisible by heads ({heads})"
        
        # Ensure d_ff is larger than d_model
        d_ff_min = max(64, d_model)
        d_ff = trial.suggest_int("d_ff", d_ff_min, 2048, step=64)
        
        # Other parameters
        encoder_layers = trial.suggest_int("encoder_layers", 1, 12)
        decoder_layers = trial.suggest_int("decoder_layers", 1, 12)
        dropout = trial.suggest_float("dropout", 0.01, 0.7)
        
        # Context encoder parameters
        context_encoder_heads = trial.suggest_categorical("context_encoder_heads", [2, 4, 8])
        context_encoder_d_model = trial.suggest_int(
            "context_encoder_d_model",
            context_encoder_heads * 16,  # Minimum size that's divisible by heads
            512,
            step=context_encoder_heads
        )
        
        # Create new model config
        model_config = base_config.model.__class__(
            input_dim=base_config.model.input_dim,
            seq_len=base_config.model.seq_len,
            d_model=d_model,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            heads=heads,
            d_ff=d_ff,
            output_dim=base_config.model.output_dim,
            dropout_rate=dropout,
            context_encoder_d_model=context_encoder_d_model,
            context_encoder_heads=context_encoder_heads,
            context_dropout_rate=base_config.model.context_dropout_rate,
            encoder_dropout_rate=base_config.model.encoder_dropout_rate,
            decoder_dropout_rate=base_config.model.decoder_dropout_rate,
            lora_rank=base_config.model.lora_rank,
            use_lora=base_config.model.use_lora,
            checkpoint_path=base_config.model.checkpoint_path
        )
        
        # Create training config
        training_config = base_config.training.__class__(
            batch_size=trial.suggest_int("batch_size", 8, 512, step=8),
            learning_rate=trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True),
            include_synthetic_training_data=base_config.training.include_synthetic_training_data,
            num_epochs=trial.suggest_int("max_epochs", 4, 5, step=1),
            device_choice=base_config.training.device_choice,
            precision=base_config.training.precision,
            fast_dev_run=base_config.training.FAST_DEV_RUN,
            train_from_checkpoint=base_config.training.train_from_checkpoint
        )
        
        # Create a new config instance
        new_config = Config(
            model=model_config,
            training=training_config
        )
        
        # Validate dimensions
        if not validate_dimensions(model_config):
            raise ValueError("Invalid model dimensions")
        
        logger.debug(f"Trial config created with params: {trial.params}")
        return new_config
        
    except Exception as e:
        logger.error(f"Error creating trial config: {str(e)}")
        raise


def validate_dimensions(model):
    """Validate model dimensions before training"""
    try:
        logger.debug("Validating model dimensions")

        # Extract model configurations
        d_model = model.d_model
        heads = model.heads

        # Check divisibility by 4 for positional encoding
        if d_model % 4 != 0:
            logger.error(f"d_model ({d_model}) must be divisible by 4")
            return False

        # Check divisibility by heads
        if d_model % heads != 0:
            logger.error(f"d_model ({d_model}) must be divisible by number of heads ({heads})")
            return False

        # Check head dimensions
        head_dim = d_model // heads
        if head_dim * heads != d_model:
            logger.error(f"Invalid head dimension: {head_dim} * {heads} != {d_model}")
            return False

        # Additional Checks:

        # Check that decoder_layers and encoder_layers are positive integers
        encoder_layers = model.encoder_layers
        decoder_layers = model.decoder_layers

        # Check that d_ff is greater than d_model
        d_ff = model.d_ff
        if d_ff < d_model:
            logger.error(f"d_ff ({d_ff}) must be greater than or equal to d_model ({d_model})")
            return False

        # Check that dropout is between 0 and 1
        dropout = model.dropout
        if not (0.0 <= dropout <= 1.0):
            logger.error(f"Dropout rate ({dropout}) must be between 0 and 1")
            return False

        logger.debug("Model dimensions validated successfully")
        return True

    except Exception as e:
        logger.error(f"Error validating dimensions: {str(e)}")
        return False

from torch.utils.data import DataLoader  # Ensure DataLoader is imported

def create_objective(base_config, train_dataset, val_dataset):
    """Creates an objective function with closure over base_config"""

    def objective(trial):
        logger.debug(f"\nStarting trial {trial.number}")
        try:
            # Create config and model using the factory function
            trial_config = create_trial_config(trial, base_config)
        
            # Use the factory function to create the model
            model = create_transformer_trainer(
                config=trial_config,
                checkpoint_path=None
            )

            # **Create DataLoaders with the suggested batch_size from trial_config**
            batch_size = trial_config.training.batch_size
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=0
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=0
            )
            callbacks = [
                PerformancePruningCallback(
                    trial,
                    monitor=base_config.optuna.pruning.get("monitor", "val_loss"),
                    patience=base_config.optuna.pruning.get("patience", 3)
                ),
                EarlyStopping(
                    monitor="val_loss",
                    patience=base_config.optuna.pruning.get("patience", 3),
                    mode="min"
                )
            ]
            
            # Conditionally add ResourcePruningCallback if CUDA is available
            if torch.cuda.is_available():
                callbacks.append(
                    ResourcePruningCallback(trial)
                )
                logger.debug("ResourcePruningCallback added to callbacks.")
            else:
                logger.info("CUDA not available. Skipping ResourcePruningCallback.")

            epoch_logging_callback = EpochLoggingCallback()
            callbacks.append(epoch_logging_callback)
            trainer = Trainer(
                max_epochs=trial_config.training.max_epochs,  # Use max_epochs from trial config
                callbacks=callbacks,
                enable_progress_bar=True,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                gradient_clip_val=trial_config.training.gradient_clip_val,
                log_every_n_steps=10  # Set log_every_n_steps to 10 to avoid warnings
            )
            
            # Prepare data loaders based on trial_config.training.batch_size
            train_loader, val_loader = prepare_data(batch_size=trial_config.training.batch_size)
            
            # Log memory usage before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated() / 1e9
                logger.debug(f"GPU memory before training: {memory_before:.2f} GB")
            
            logger.debug(f"Trainer initialized with max_epochs={trial_config.training.max_epochs}")
            logger.debug("Starting training")
            trainer.fit(model, train_loader, val_loader)
            
            # Retrieve metrics
            def get_metric(metric_name, default):
                metric = trainer.callback_metrics.get(metric_name, default)
                return metric.item() if isinstance(metric, torch.Tensor) else metric

            metrics = TrialMetrics(
                val_loss=get_metric("val_loss", float('inf')),
                train_loss=get_metric("train_loss", float('inf')),
                val_accuracy=get_metric("val_accuracy", 0.0)
            )
            
            # Log memory usage after training
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1e9
                metrics.memory_used = memory_after - memory_before
                logger.debug(f"GPU memory used: {metrics.memory_used:.2f} GB")
            
            # Log metrics
            metrics.log_metrics()
            
            # Prune trial if validation loss is invalid
            if not metrics.val_loss or torch.isnan(torch.tensor(metrics.val_loss)):
                logger.warning("Invalid validation loss detected")
                raise optuna.TrialPruned()
            
            return metrics.val_loss
        
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise optuna.TrialPruned()
            
    return objective
class EpochLoggingCallback(Callback):
    def on_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1  # Trainer starts from 0
        logger.debug(f"Starting epoch {current_epoch}/{trainer.max_epochs}")

    def on_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1
        logger.debug(f"Finished epoch {current_epoch}/{trainer.max_epochs}")

class PerformancePruningCallback(Callback):
    def __init__(self, trial, monitor="val_loss", patience=5):  # Increased patience from 3 to 5
        self.trial = trial
        self.monitor = monitor
        self.patience = patience
        self.best_value = float('inf')
        self.no_improvement_count = 0
        
    def on_validation_end(self, trainer, pl_module):
        current_value = trainer.callback_metrics.get(self.monitor)
        if current_value is None:
            return
            
        # Report to Optuna
        self.trial.report(current_value.item(), step=trainer.current_epoch)
        
        # Check for improvement
        if current_value < self.best_value:
            self.best_value = current_value
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            
        # Prune if no improvement for patience epochs
        if self.no_improvement_count >= self.patience:
            logger.info(f"Trial {self.trial.number} pruned due to no improvement for {self.patience} epochs")
            raise optuna.TrialPruned()
            
        # Let Optuna decide if trial should be pruned
        if self.trial.should_prune():
            logger.info(f"Trial {self.trial.number} pruned by Optuna")
            raise optuna.TrialPruned()

class ResourcePruningCallback(Callback):
    def __init__(self, trial, max_memory_gb=None):
        self.trial = trial
        if torch.cuda.is_available():
            self.max_memory_gb = max_memory_gb or (torch.cuda.get_device_properties(0).total_memory / 1e9 * 0.9)
            logger.debug(f"ResourcePruningCallback initialized with max_memory_gb={self.max_memory_gb} GB")
        else:
            self.max_memory_gb = None
            logger.info("CUDA not available. ResourcePruningCallback will not monitor memory usage.")
        
    def on_batch_end(self, trainer, pl_module):
        if self.max_memory_gb is not None and torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1e9
            if current_memory > self.max_memory_gb:
                logger.warning(f"Trial {self.trial.number} pruned due to excessive memory usage: {current_memory:.2f} GB")
                raise optuna.TrialPruned()
