# Utils/optuna/objective.py

import torch
import logging
from dataclasses import dataclass
from cultivation.systems.arc_reactor.jarc_reactor.config_schema import (
    IntRange,
    FloatRange,
    CategoricalChoice,
)
import optuna
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate # Hydra config classes
from cultivation.systems.arc_reactor.jarc_reactor.utils.model_factory import create_transformer_trainer
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import EarlyStopping

from cultivation.utils.logging_config import setup_logging

# Setup logging
setup_logging()
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

def create_trial_config(trial: optuna.Trial, base_config: DictConfig) -> DictConfig:
    """Create a new config with trial-suggested values based on the structured schema."""
    logger.debug("Creating trial config from structured schema")
    
    trial_cfg_dict = OmegaConf.to_container(base_config, resolve=True)
    suggested_params = {}

    # --- Suggest parameters based on the structured schema ---
    # Manually instantiate each parameter range from the config
    param_ranges = {
        name: instantiate(p_range_cfg)
        for name, p_range_cfg in base_config.optuna.param_ranges.items()
    }



    # Suggest all other parameters
    for name, p_range in param_ranges.items():
        if name in suggested_params:
            continue

        value = None
        if isinstance(p_range, CategoricalChoice):
            value = trial.suggest_categorical(name, p_range.choices)
        elif isinstance(p_range, IntRange):
            low, high, step = p_range.low, p_range.high, p_range.step
            # Handle special dependency logic
            if name == 'd_model':
                heads = suggested_params.get("heads", base_config.model.heads)
                low = ((low + heads - 1) // heads) * heads
                high = (high // heads) * heads
                step = step if step % heads == 0 else heads
            elif name == 'd_ff':
                d_model = suggested_params.get('d_model', base_config.model.d_model)
                low = max(low, d_model)
            elif name == 'context_encoder_d_model':
                ce_heads = suggested_params.get("context_encoder_heads", base_config.model.context_encoder.heads)
                low = ((low + ce_heads - 1) // ce_heads) * ce_heads
                high = (high // ce_heads) * ce_heads
                step = step if step % ce_heads == 0 else ce_heads

            value = trial.suggest_int(name, low, high, step=step, log=p_range.log)
        elif isinstance(p_range, FloatRange):
            value = trial.suggest_float(name, p_range.low, p_range.high, step=p_range.step, log=p_range.log)
        else:
            raise TypeError(f"Unsupported parameter range type for '{name}': {type(p_range)}")
        
        suggested_params[name] = value

    # --- Update the config dictionary with the suggested parameters ---
    for name, value in suggested_params.items():
        if name.startswith('context_encoder_'):
            key = name.replace('context_encoder_', '')
            trial_cfg_dict['model']['context_encoder'][key] = value
        elif name in trial_cfg_dict.get('model', {}):
            trial_cfg_dict['model'][name] = value
        elif name in trial_cfg_dict.get('training', {}):
            trial_cfg_dict['training'][name] = value
        else:
            logger.warning(f"Parameter '{name}' not found in model or training config sections.")

    new_config = OmegaConf.create(trial_cfg_dict)

    if not validate_dimensions(new_config.model):
        raise ValueError("Invalid model dimensions")
    
    logger.debug(f"Trial config created with params: {trial.params}")
    return new_config


def validate_dimensions(model_cfg: DictConfig): # Expects model section of DictConfig
    """Validate model dimensions before training"""
    try:
        logger.debug("Validating model dimensions")

        # Extract model configurations
        d_model = model_cfg.d_model
        heads = model_cfg.heads

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

        # Check that d_ff is greater than d_model
        d_ff = model_cfg.d_ff
        if d_ff < d_model:
            logger.error(f"d_ff ({d_ff}) must be greater than or equal to d_model ({d_model})")
            return False

        # Check that dropout is between 0 and 1
        dropout = model_cfg.dropout_rate # Assuming dropout_rate is the key in schema
        if not (0.0 <= dropout <= 1.0):
            logger.error(f"Dropout rate ({dropout}) must be between 0 and 1")
            return False

        logger.debug("Model dimensions validated successfully")
        return True

    except Exception as e:
        logger.error(f"Error validating dimensions: {str(e)}")
        return False

def create_objective(base_config: DictConfig, train_dataset, val_dataset):
    """Creates an objective function with closure over base_config and datasets"""

    def objective(trial):
        logger.debug(f"\nStarting trial {trial.number}")
        try:
            # Create config for this trial
            trial_config = create_trial_config(trial, base_config)
        
            # Create model
            model = create_transformer_trainer(
                config=trial_config,
                checkpoint_path=None
            )

            # Create DataLoaders from existing datasets
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
            
            # Remove redundant data loading - we already have the loaders
            
            # Log memory usage before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated() / 1e9
                logger.debug(f"GPU memory before training: {memory_before:.2f} GB")
            
            logger.debug(f"Trainer initialized with max_epochs={trial_config.training.max_epochs}")
            logger.debug("Starting training")
            trainer.fit(model, train_loader, val_loader)
            
            # Rest of the code remains the same...
            def get_metric(metric_name, default):
                metric = trainer.callback_metrics.get(metric_name, default)
                return metric.item() if isinstance(metric, torch.Tensor) else metric

            metrics = TrialMetrics(
                val_loss=get_metric("val_loss", float('inf')),
                train_loss=get_metric("train_loss", float('inf')),
                val_accuracy=get_metric("val_accuracy", 0.0)
            )
            
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1e9
                metrics.memory_used = memory_after - memory_before
                logger.debug(f"GPU memory used: {metrics.memory_used:.2f} GB")
            
            metrics.log_metrics()
            
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
