import logging
import sys
import os
import signal
import torch
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from cultivation.systems.arc_reactor.jarc_reactor.data.data_module import MyDataModule
from cultivation.systems.arc_reactor.jarc_reactor.utils.train import TransformerTrainer
from cultivation.systems.arc_reactor.jarc_reactor.optimization.best_params_manager import BestParamsManager
from cultivation.systems.arc_reactor.jarc_reactor.hydra_setup import register_hydra_configs
from cultivation.utils.logging_config import setup_logging as setup_central_logging # Aliased

# Call to register Hydra configurations with ConfigStore
# This needs to be done before @hydra.main is encountered
register_hydra_configs()

# Global logger, will be configured by setup_central_logging
logger = logging.getLogger(__name__) # Get logger for this module

class StreamToLogger(object):
    """Redirect writes to a logger."""
    def __init__(self, logger_instance, log_level=logging.INFO):
        self.logger = logger_instance
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass # sys.stdout has no flush method, so this is a no-op

def signal_handler(sig, frame):
    logger.warning("KeyboardInterrupt received. Exiting immediately.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def setup_cuda_optimizations_script(cfg: DictConfig):
    """Sets up CUDA optimizations based on config."""
    if cfg.enable_cuda_optimizations and torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major >= 7:  # Volta or newer
            torch.set_float32_matmul_precision('medium')
            logger.info("Enabled Tensor Core optimizations for faster matrix operations (matmul precision set to 'medium')")
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("Enabled TF32 for matmul operations.")
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN benchmark mode.")
    elif cfg.enable_cuda_optimizations and not torch.cuda.is_available():
        logger.warning("CUDA optimizations enabled in config, but CUDA is not available.")

def setup_model_training(cfg: DictConfig):
    """Setup model and training configuration using Hydra config."""
    if cfg.use_best_params:
        logger.info("Loading best parameters from Optuna study...")
        params_manager = BestParamsManager(
            storage_url=cfg.optuna.storage_url,
            study_name=cfg.optuna.study_name
        )
        # Assuming update_config can work with DictConfig or is adapted.
        # This might involve OmegaConf.merge or direct updates.
        if params_manager.update_config_hydra(cfg): 
            logger.info("Successfully applied best parameters from Optuna study.")
        else:
            logger.warning("Failed to load or apply best parameters, using current configuration.")

    logger.info("Final configuration after potential Optuna updates:")
    logger.info(f"Model parameters: {OmegaConf.to_yaml(cfg.model)}")
    logger.info(f"Training parameters: {OmegaConf.to_yaml(cfg.training)}")

    try:
        # Ensure checkpoint_path is absolute or resolvable by Pytorch Lightning
        checkpoint_path_to_load = cfg.model.checkpoint_path
        if checkpoint_path_to_load and not os.path.isabs(checkpoint_path_to_load):
            # Assuming hydra.utils.to_absolute_path or similar might be used if running from original CWD
            # For now, let's log if it's relative. Pytorch Lightning might handle it.
            logger.info(f"Provided checkpoint_path is relative: {checkpoint_path_to_load}")

        logger.info(f"Attempting to load model from checkpoint: {checkpoint_path_to_load}")
        if cfg.training.train_from_checkpoint and checkpoint_path_to_load:
            if not Path(checkpoint_path_to_load).exists():
                logger.error(f"Checkpoint file not found at: {checkpoint_path_to_load}. Initializing new model.")
                model = TransformerTrainer(config=cfg) # Pass DictConfig
            else:
                try:
                    model = TransformerTrainer.load_from_checkpoint(checkpoint_path_to_load, config=cfg) # Pass DictConfig
                    logger.info(f"Model loaded successfully from checkpoint: {checkpoint_path_to_load}")
                except Exception as e:
                    logger.error(f"Failed to load model from checkpoint {checkpoint_path_to_load}: {str(e)}. Initializing new model.")
                    model = TransformerTrainer(config=cfg) # Pass DictConfig
        else:
            logger.info("Initializing a new model (train_from_checkpoint is false or no path provided).")
            model = TransformerTrainer(config=cfg) # Pass DictConfig

        # Log device information
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        
        actual_device_choice = cfg.training.device_choice
        if actual_device_choice == "auto":
            actual_device_choice = 'cuda' if torch.cuda.is_available() else 'cpu' # PL uses 'cuda'
        logger.info(f"Effective device for Pytorch Lightning Trainer: {actual_device_choice}")

        return model

    except Exception as e:
        logger.error(f"Error in setup_model_training: {str(e)}", exc_info=True)
        raise

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main_app(cfg: DictConfig) -> None:
    # 1. Setup Centralized Logging
    # The log_dir from cfg.logging will be relative to Hydra's output directory
    setup_central_logging(cfg.logging)
    main_logger = logging.getLogger(__name__) # Use a logger specific to this main function
    main_logger.info(f"Logging initialized. Hydra CWD: {os.getcwd()}")
    main_logger.info(f"Original CWD (if available via hydra.utils.get_original_cwd()): {hydra.utils.get_original_cwd() if hasattr(hydra.utils, 'get_original_cwd') else 'N/A'}")

    # Redirect stdout/stderr after central logging is set up
    # Use a specific logger for these streams if desired, e.g., logging.getLogger('stdio')
    # For simplicity, using the main_logger here.
    sys.stdout = StreamToLogger(main_logger, logging.INFO)
    sys.stderr = StreamToLogger(main_logger, logging.ERROR)
    main_logger.info("Stdout and Stderr redirected to logger.")

    # 2. Setup CUDA Optimizations
    setup_cuda_optimizations_script(cfg)

    # 3. Print the resolved configuration
    main_logger.info("Resolved Hydra configuration:")
    main_logger.info(OmegaConf.to_yaml(cfg))

    # 4. Setup model and training configurations
    model = setup_model_training(cfg)
    
    # 5. Setup DataModule
    # MyDataModule needs to be compatible with cfg.training.batch_size
    data_module = MyDataModule(batch_size=cfg.training.batch_size, data_dir_config=cfg.training.training_data_dir) 
    main_logger.info(f"DataModule initialized with batch_size: {cfg.training.batch_size}")

    # 6. Setup ModelCheckpoint callback
    # dirpath will be relative to Hydra's output directory
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints", 
        filename='model-{epoch:02d}-{step}-val_loss={val_loss:.4f}',
        save_top_k=cfg.training.get('save_top_k', 1),
        monitor=cfg.training.get('monitor_metric', 'val_loss'),
        mode=cfg.training.get('monitor_mode', 'min'),
        save_weights_only=cfg.training.get('save_weights_only', False),
        every_n_epochs=cfg.training.get('save_every_n_epochs', 1),
        save_last=cfg.training.get('save_last_checkpoint', True),
        verbose=True
    )
    main_logger.info(f"Checkpoint callback configured. Dirpath: {checkpoint_callback.dirpath}")

    # 7. Setup EarlyStopping callback
    early_stopping_callback = EarlyStopping(
        monitor=cfg.training.get('monitor_metric', 'val_loss'), 
        patience=cfg.training.get('early_stopping_patience', 35),
        mode=cfg.training.get('monitor_mode', 'min'),
        verbose=True
    )
    main_logger.info("EarlyStopping callback configured.")

    # 8. Initialize the Pytorch Lightning Trainer
    # Determine accelerator and devices for PL Trainer
    accelerator_type = 'cpu'
    devices_val = 1
    if cfg.training.device_choice == "auto":
        if torch.cuda.is_available():
            accelerator_type = 'gpu'
            devices_val = torch.cuda.device_count() if cfg.training.get('use_all_gpus', True) else 1
    elif cfg.training.device_choice == "gpu" or cfg.training.device_choice == "cuda":
        if torch.cuda.is_available():
            accelerator_type = 'gpu'
            devices_val = torch.cuda.device_count() if cfg.training.get('use_all_gpus', True) else 1
        else:
            main_logger.warning("Configured for GPU, but CUDA not available. Falling back to CPU.")
            accelerator_type = 'cpu'
    
    trainer_precision = cfg.training.precision
    # Ensure precision is string for 'bf16' etc.
    if not isinstance(trainer_precision, str) and trainer_precision in [16,32,64]:
        trainer_precision = int(trainer_precision)

    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        enable_progress_bar=cfg.training.get('enable_progress_bar', True),
        callbacks=[checkpoint_callback, early_stopping_callback],
        precision=trainer_precision,
        log_every_n_steps=cfg.training.get('log_every_n_steps', 50),
        detect_anomaly=cfg.training.get('detect_anomaly', False),
        enable_checkpointing=True, # Handled by ModelCheckpoint callback
        default_root_dir=".", # Log files will go into Hydra's output directory
        accelerator=accelerator_type,
        devices=devices_val,
        fast_dev_run=cfg.training.fast_dev_run
    )
    main_logger.info(f"Pytorch Lightning Trainer initialized. Accelerator: {accelerator_type}, Devices: {devices_val}, Precision: {trainer_precision}")

    # 9. Start training and testing
    try:
        main_logger.info("Starting training...")
        trainer.fit(model, data_module, ckpt_path=cfg.training.get('resume_from_checkpoint_pl', None))
        main_logger.info("Training finished.")
        
        if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
            main_logger.info(f"Best model saved to: {trainer.checkpoint_callback.best_model_path}")
        else:
            main_logger.warning("No best model path found from checkpoint callback.")
        
        main_logger.info("Starting testing...")
        trainer.test(model, data_module, ckpt_path='best' if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path else None)
        main_logger.info("Testing finished.")

    except KeyboardInterrupt:
        main_logger.warning("Training interrupted by user. Exiting gracefully.")
    except Exception as e:
        main_logger.error(f"An error occurred during training/testing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main_app()
