import sys
import logging
import os
import signal
import torch
from pathlib import Path
from .config_schema import LoggingConfigSchema
from cultivation.utils import logging_config as project_logging
import hydra
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from cultivation.systems.arc_reactor.jarc_reactor.data.data_module import MyDataModule
from cultivation.systems.arc_reactor.jarc_reactor.utils.train import TransformerTrainer
from cultivation.systems.arc_reactor.jarc_reactor.optimization.best_params_manager import BestParamsManager
from cultivation.systems.arc_reactor.jarc_reactor.hydra_setup import register_hydra_configs

# Register Hydra configs before @hydra.main
register_hydra_configs()

# Global logger
logger = logging.getLogger(__name__)

def signal_handler(sig, frame):
    logger.warning("KeyboardInterrupt received. Exiting immediately.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def setup_logging(logging_config: LoggingConfigSchema) -> logging.Logger:
    """Configures logging using the centralized utility."""
    resolved_log_dir_str = hydra.utils.to_absolute_path(str(logging_config.log_dir))
    log_dir = Path(resolved_log_dir_str)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = None
    if logging_config.file_logging.enable:
        log_file_path = log_dir / str(logging_config.file_logging.log_file_name)
    
    project_logging.setup_logging(log_file=log_file_path)
    main_logger = logging.getLogger(__name__)
    main_logger.info(f"Logging initialized. Log directory: {log_dir}")
    main_logger.info(f"Hydra CWD: {os.getcwd()}")
    return main_logger

def setup_cuda_optimizations(cfg: DictConfig, logger: logging.Logger):
    """Sets up CUDA optimizations based on config."""
    if cfg.enable_cuda_optimizations and torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major >= 7:
            torch.set_float32_matmul_precision('medium')
            logger.info("Enabled Tensor Core optimizations (matmul precision set to 'medium')")
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN benchmark mode.")
    elif cfg.enable_cuda_optimizations and not torch.cuda.is_available():
        logger.warning("CUDA optimizations enabled, but CUDA is not available.")

def initialize_model(cfg: DictConfig, logger: logging.Logger) -> TransformerTrainer:
    """Initializes or loads the TransformerTrainer model."""
    if cfg.use_best_params:
        logger.info("Loading best parameters from Optuna study...")
        params_manager = BestParamsManager(storage_url=cfg.optuna.storage_url, study_name=cfg.optuna.study_name)
        if params_manager.update_config(cfg):
            logger.info("Successfully applied best parameters from Optuna study.")
        else:
            logger.warning("Failed to load best parameters, using current configuration.")

    logger.info(f"Final Model parameters: {OmegaConf.to_yaml(cfg.model)}")
    logger.info(f"Final Training parameters: {OmegaConf.to_yaml(cfg.training)}")

    checkpoint_path = cfg.model.get('checkpoint_path')
    if cfg.training.train_from_checkpoint and checkpoint_path:
        abs_checkpoint_path = hydra.utils.to_absolute_path(checkpoint_path)
        if Path(abs_checkpoint_path).exists():
            try:
                logger.info(f"Loading model from checkpoint: {abs_checkpoint_path}")
                return TransformerTrainer.load_from_checkpoint(abs_checkpoint_path, config=cfg)
            except Exception as e:
                logger.error(f"Failed to load from checkpoint {abs_checkpoint_path}: {e}. Initializing new model.")
        else:
            logger.error(f"Checkpoint file not found at: {abs_checkpoint_path}. Initializing new model.")
    
    logger.info("Initializing a new model.")
    return TransformerTrainer(config=cfg)

def setup_callbacks(cfg: DictConfig, logger: logging.Logger) -> list:
    """Initializes and returns a list of PyTorch Lightning callbacks."""
    checkpoint_callback = ModelCheckpoint(
        dirpath=hydra.utils.to_absolute_path(f"{cfg.training.training_log_dir}/checkpoints"),
        filename='model-{epoch:02d}-{step}-val_loss={val_loss:.4f}',
        save_top_k=cfg.training.get('save_top_k', 1),
        monitor=cfg.training.get('monitor_metric', 'val_loss'),
        mode=cfg.training.get('monitor_mode', 'min'),
        verbose=True
    )
    logger.info(f"Checkpoint callback configured. Dirpath: {checkpoint_callback.dirpath}")

    early_stopping_callback = EarlyStopping(
        monitor=cfg.training.get('monitor_metric', 'val_loss'),
        patience=cfg.training.get('early_stopping_patience', 35),
        mode=cfg.training.get('monitor_mode', 'min'),
        verbose=True
    )
    logger.info("EarlyStopping callback configured.")
    return [checkpoint_callback, early_stopping_callback]

def initialize_trainer(cfg: DictConfig, callbacks: list, t_logger: TensorBoardLogger, main_logger: logging.Logger) -> Trainer:
    """Initializes and returns the PyTorch Lightning Trainer."""
    accelerator = 'cpu'
    devices = 1
    if cfg.training.device_choice == "auto" and torch.cuda.is_available():
        accelerator = 'gpu'
        devices = torch.cuda.device_count()
    elif cfg.training.device_choice in ("gpu", "cuda"):
        if torch.cuda.is_available():
            accelerator = 'gpu'
            devices = torch.cuda.device_count()
        else:
            main_logger.warning("Configured for GPU, but CUDA not available. Falling back to CPU.")

    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        precision=cfg.training.precision,
        log_every_n_steps=cfg.training.get('log_every_n_steps', 50),
        logger=t_logger,
        accelerator=accelerator,
        devices=devices,
        fast_dev_run=cfg.training.fast_dev_run
    )
    main_logger.info(f"Trainer initialized. Accelerator: {accelerator}, Devices: {devices}")
    return trainer

@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main_app(cfg: DictConfig) -> None:
    """Main application entry point."""
    main_logger = setup_logging(cfg.logging)
    setup_cuda_optimizations(cfg, main_logger)
    
    main_logger.info("Resolved Hydra configuration:\n" + OmegaConf.to_yaml(cfg))

    model = initialize_model(cfg, main_logger)
    data_module = MyDataModule(cfg=cfg)
    main_logger.info("DataModule initialized.")

    callbacks = setup_callbacks(cfg, main_logger)
    tensorboard_logger = TensorBoardLogger(
        save_dir=hydra.utils.to_absolute_path(cfg.training.training_log_dir),
        name="lightning_logs"
    )
    main_logger.info(f"TensorBoard logger configured. Save dir: {tensorboard_logger.save_dir}")

    trainer = initialize_trainer(cfg, callbacks, tensorboard_logger, main_logger)

    try:
        main_logger.info("Starting training...")
        trainer.fit(model, data_module, ckpt_path=cfg.training.get('resume_from_checkpoint_pl'))
        main_logger.info("Training finished.")

        best_model_path = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None
        if best_model_path:
            main_logger.info(f"Best model saved to: {best_model_path}")
        else:
            main_logger.warning("No best model path found from checkpoint callback.")

        if cfg.training.get('do_test', True):
            main_logger.info("Starting testing phase...")
            trainer.test(model, data_module, ckpt_path='best' if best_model_path else None)
            main_logger.info("Testing finished.")

    except KeyboardInterrupt:
        main_logger.warning("Training interrupted by user. Exiting gracefully.")
    except Exception as e:
        main_logger.error(f"An error occurred during training/testing: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main_app()
