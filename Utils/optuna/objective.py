# Utils/optuna/objective.py

import torch
import logging
from dataclasses import dataclass
import optuna
from config import Config
from Utils.data_preparation import prepare_data
from train import TransformerTrainer
from pytorch_lightning import Trainer
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

        # Get ranges from OptunaConfig
        ranges = base_config.optuna.param_ranges

        # Create new config objects based on existing classes
        model_config = base_config.model.__class__()
        training_config = base_config.training.__class__()

        # Suggest hyperparameters using Optuna
        # Suggest the number of heads first
        model_config.heads = trial.suggest_int("heads", *ranges["heads"])

        # Suggest a multiplier to ensure d_model is divisible by heads
        d_model_multiplier = trial.suggest_int("d_model_multiplier", *ranges["d_model_multiplier"])

        # Set d_model as a multiple of heads
        model_config.d_model = model_config.heads * d_model_multiplier

        model_config.decoder_layers = trial.suggest_int("decoder_layers", *ranges["decoder_layers"])
        model_config.d_ff = trial.suggest_int("d_ff", *ranges["d_ff"])
        model_config.dropout = trial.suggest_float("dropout", *ranges["dropout"])

        training_config.batch_size = trial.suggest_int("batch_size", *ranges["batch_size"])
        training_config.learning_rate = trial.suggest_float(
            "learning_rate", *ranges["learning_rate"], log=True
        )

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
            # Create trial-specific config
            trial_config = create_trial_config(trial, base_config)

            # Initialize the model with trial-specific hyperparameters
            model = TransformerTrainer(
                input_dim=trial_config.model.input_dim,
                d_model=trial_config.model.d_model,
                encoder_layers=trial_config.model.encoder_layers,
                decoder_layers=trial_config.model.decoder_layers,
                heads=trial_config.model.heads,
                d_ff=trial_config.model.d_ff,
                output_dim=trial_config.model.output_dim,
                learning_rate=trial_config.training.learning_rate,
                include_sythtraining_data=trial_config.training.include_sythtraining_data
            )

            # Validate dimensions before training
            if not validate_dimensions(model):
                logger.warning("Model dimension validation failed")
                raise optuna.TrialPruned()

            # Prepare data
            train_loader, val_loader = prepare_data()

            # Setup early stopping callback
            early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min")

            # Initialize Trainer
            trainer = Trainer(
                max_epochs=10,  # Adjust epochs as needed
                callbacks=[early_stop],
                enable_progress_bar=True,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1
            )

            # Log memory usage before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated() / 1024**2
                logger.debug(f"GPU memory before training: {memory_before:.2f} MB")

            # Train the model
            logger.debug("Starting training")
            trainer.fit(model, train_loader, val_loader)

            # Retrieve metrics
            metrics = TrialMetrics(
                val_loss=trainer.callback_metrics.get("val_loss", float('inf')),
                train_loss=trainer.callback_metrics.get("train_loss", float('inf')),
                val_accuracy=trainer.callback_metrics.get("val_accuracy", 0.0)
            )

            # Log memory usage after training
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1024**2
                metrics.memory_used = memory_after - memory_before
                logger.debug(f"GPU memory used: {metrics.memory_used:.2f} MB")

            # Log metrics
            metrics.log_metrics()

            # Prune trial if validation loss is invalid
            if not metrics.val_loss or torch.isnan(torch.tensor(metrics.val_loss)):
                logger.warning("Invalid validation loss detected")
                raise optuna.TrialPruned()

            return metrics.val_loss

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise optuna.TrialPruned()

    return objective