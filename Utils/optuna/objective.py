# Utils/optuna/objective.py

import torch
import logging
from dataclasses import dataclass
import optuna
from config import Config
from Utils.data_preparation import prepare_data
from train import TransformerTrainer
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import EarlyStopping

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
        ranges = base_config.optuna.param_ranges
        
        # Create new config objects
        model_config = base_config.model.__class__()
        training_config = base_config.training.__class__()
        
        # First determine core architecture parameters
        model_config.heads = trial.suggest_categorical("heads", ranges["heads"])
        
        # Ensure d_model is divisible by both heads and 4
        d_model = trial.suggest_int(
            "d_model", 
            ranges["d_model"][0], 
            ranges["d_model"][1], 
            step=ranges["d_model"][2]
        )
        
        # Validate dimensions
        if d_model % model_config.heads != 0 or d_model % 4 != 0:
            logger.warning(f"Invalid dimension combination: d_model={d_model}, heads={model_config.heads}")
            raise optuna.TrialPruned()
            
        model_config.d_model = d_model
        
        # Layer configuration
        model_config.encoder_layers = trial.suggest_int("encoder_layers", *ranges["encoder_layers"])
        model_config.decoder_layers = trial.suggest_int("decoder_layers", *ranges["decoder_layers"])
        
        # Suggest d_ff, ensuring it's greater than d_model
        min_d_ff = d_model + 64  # Constrain d_ff to be at least d_model + 64
        if min_d_ff > ranges["d_ff"][1]:
            logger.warning(f"Cannot set d_ff > d_model={d_model} within the parameter range.")
            raise optuna.TrialPruned()
        
        model_config.d_ff = trial.suggest_int(
            "d_ff", 
            max(min_d_ff, ranges["d_ff"][0]), 
            ranges["d_ff"][1], 
            step=ranges["d_ff"][2]
        )
        
        # Dropout
        model_config.dropout = trial.suggest_float("dropout", *ranges["dropout"])
        
        # Context encoder parameters
        model_config.context_encoder_d_model = trial.suggest_int(
            "context_encoder_d_model",
            ranges["context_encoder_d_model"][0],
            ranges["context_encoder_d_model"][1],
            step=ranges["context_encoder_d_model"][2]
        )
        model_config.context_encoder_heads = trial.suggest_categorical(
            "context_encoder_heads",
            ranges["context_encoder_heads"]
        )
        
        # Training parameters
        training_config.batch_size = trial.suggest_int("batch_size", *ranges["batch_size"])
        training_config.learning_rate = trial.suggest_float(
            "learning_rate", 
            *ranges["learning_rate"], 
            log=True
        )
        
        # Suggest gradient_clip_val
        training_config.gradient_clip_val = trial.suggest_float(
            "gradient_clip_val",
            0.0,       # Minimum value
            5.0,       # Maximum value
            step=0.1   # Step size (adjust as needed)
        )
        training_config.max_epochs = trial.suggest_int(
            "max_epochs",
            ranges["max_epochs"][0],
            ranges["max_epochs"][1],
            step=ranges["max_epochs"][2]
        )
        
        logger.debug(f"Model dimensions:")
        logger.debug(f"  d_model: {model_config.d_model}")
        logger.debug(f"  heads: {model_config.heads}")
        logger.debug(f"  head_dim: {model_config.d_model // model_config.heads}")
        logger.debug(f"  d_ff: {model_config.d_ff}")
        logger.debug(f"  context_encoder_d_model: {model_config.context_encoder_d_model}")
        
        logger.debug(f"Trial config created with params: {trial.params}")
        return Config(model=model_config, training=training_config)
        
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
        if encoder_layers <= 0 or decoder_layers <= 0:
            logger.error(
                f"Encoder layers ({encoder_layers}) and decoder layers ({decoder_layers}) must be positive integers"
            )
            return False

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

def create_objective(base_config):
    """Creates an objective function with closure over base_config"""

    def objective(trial):
        logger.debug(f"\nStarting trial {trial.number}")
        try:
            # Create config and model
            trial_config = create_trial_config(trial, base_config)
            logger.debug(f"Trial {trial.number} - max_epochs: {trial_config.training.max_epochs}")
            model = TransformerTrainer(
                input_dim=trial_config.model.input_dim,
                d_model=trial_config.model.d_model,
                encoder_layers=trial_config.model.encoder_layers,
                decoder_layers=trial_config.model.decoder_layers,
                heads=trial_config.model.heads,
                d_ff=trial_config.model.d_ff,
                output_dim=trial_config.model.output_dim,
                learning_rate=trial_config.training.learning_rate,
                include_sythtraining_data=trial_config.training.include_sythtraining_data,
            )

            # Setup callbacks with advanced pruning
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
                max_epochs=trial_config.training.max_epochs,  # Use max_epochs from trial
                callbacks=callbacks,
                enable_progress_bar=True,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                gradient_clip_val=trial_config.training.gradient_clip_val
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
            metrics = TrialMetrics(
                val_loss=trainer.callback_metrics.get("val_loss", float('inf')).item(),
                train_loss=trainer.callback_metrics.get("train_loss", float('inf')).item(),
                val_accuracy=trainer.callback_metrics.get("val_accuracy", 0.0).item()
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
    def __init__(self, trial, monitor="val_loss", patience=3):
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
