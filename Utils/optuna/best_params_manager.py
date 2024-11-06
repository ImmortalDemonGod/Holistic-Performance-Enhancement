import json
import logging
import optuna
from train import TransformerTrainer
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BestParams:
    model_params: Dict[str, Any]
    training_params: Dict[str, Any]
    study_name: str
    trial_number: int

class BestParamsManager:
    def __init__(self, storage_url: str, study_name: str, save_path: str = "best_params.json"):
        self.storage_url = storage_url
        self.study_name = study_name
        self.save_path = Path(save_path)
        
        logger.debug(f"Initialized BestParamsManager")
        logger.debug(f"Storage URL: {storage_url}")
        logger.debug(f"Study name: {study_name}")
        logger.debug(f"Save path: {save_path}")

    def load_best_trial(self) -> Optional[optuna.Trial]:
        """Load the best trial from the Optuna study."""
        try:
            study = optuna.load_study(
                study_name=self.study_name,
                storage=self.storage_url
            )
            return study.best_trial
        except Exception as e:
            logger.error(f"Failed to load best trial: {str(e)}")
            return None
    
    def extract_params(self, trial: optuna.Trial) -> BestParams:
        """Extract and categorize parameters from the best trial."""
        params = trial.params
        
        # Separate parameters into model and training categories
        model_params = {
            'd_model': params['d_model'],
            'heads': params['heads'],
            'encoder_layers': params['encoder_layers'],
            'decoder_layers': params['decoder_layers'],
            'd_ff': params['d_ff'],
            'dropout': params['dropout'],
            'context_encoder_d_model': params['context_encoder_d_model'],
            'context_encoder_heads': params['context_encoder_heads']
        }
        
        training_params = {
            'batch_size': params['batch_size'],
            'learning_rate': params['learning_rate'],
            'gradient_clip_val': params['gradient_clip_val']
        }
        
        return BestParams(
            model_params=model_params,
            training_params=training_params,
            study_name=self.study_name,
            trial_number=trial.number
        )

    def save_params(self, best_params: BestParams):
        """Save the best parameters to a JSON file."""
        try:
            params_dict = {
                'model_params': best_params.model_params,
                'training_params': best_params.training_params,
                'study_name': best_params.study_name,
                'trial_number': best_params.trial_number
            }
            
            with open(self.save_path, 'w') as f:
                json.dump(params_dict, f, indent=2)
            
            logger.info(f"Saved best parameters to {self.save_path}")
        except Exception as e:
            logger.error(f"Failed to save parameters: {str(e)}")

    def load_params(self) -> Optional[BestParams]:
        """Load the best parameters from the JSON file."""
        try:
            if not self.save_path.exists():
                logger.warning(f"No saved parameters found at {self.save_path}")
                return None
                
            with open(self.save_path, 'r') as f:
                params_dict = json.load(f)
            
            return BestParams(
                model_params=params_dict['model_params'],
                training_params=params_dict['training_params'],
                study_name=params_dict['study_name'],
                trial_number=params_dict['trial_number']
            )
        except Exception as e:
            logger.error(f"Failed to load parameters: {str(e)}")
            return None

    def update_config(self, config):
        """Update config with best parameters, overwriting existing values.
        
        Args:
            config: The configuration object to update
        """
        try:
            # First try loading from JSON
            best_params = self.load_params()
            
            # If no JSON, try loading from Optuna study
            if best_params is None:
                logger.info("No saved parameters found, loading from Optuna study...")
                best_trial = self.load_best_trial()
                if best_trial is None:
                    logger.warning("No best trial found in Optuna study")
                    return False
                    
                best_params = self.extract_params(best_trial)
                self.save_params(best_params)

            # Update model parameters - always overwrite
            for key, value in best_params.model_params.items():
                old_value = getattr(config.model, key, None)
                setattr(config.model, key, value)
                logger.info(f"Updated model parameter '{key}': {old_value} -> {value}")
            
            # Update training parameters - always overwrite except for 'max_epochs'
            for key, value in best_params.training_params.items():
                if key != 'max_epochs':  # Don't override epochs
                    old_value = getattr(config.training, key, None)
                    setattr(config.training, key, value)
                    logger.info(f"Updated training parameter '{key}': {old_value} -> {value}")
            
            # Add: Post-update validation
            self.validate_updated_config(config, best_params)
            
            # Add: Log the entire updated configuration
            self.log_updated_config(config)
            
            logger.info("Configuration updated successfully with best parameters")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update config: {str(e)}")
            return False

        # Ensure all required parameters are passed when loading from checkpoint
        checkpoint_path = config.model.checkpoint_path
        model = TransformerTrainer.load_from_checkpoint(
            checkpoint_path,
            input_dim=config.model.input_dim,
            d_model=config.model.d_model,
            encoder_layers=config.model.encoder_layers,
            decoder_layers=config.model.decoder_layers,
            heads=config.model.heads,
            d_ff=config.model.d_ff,
            output_dim=config.model.output_dim,
            learning_rate=config.training.learning_rate,
            include_synthetic_training_data=config.training.include_synthetic_training_data,
            dropout=config.model.dropout,
            context_encoder_d_model=config.model.context_encoder_d_model,
            context_encoder_heads=config.model.context_encoder_heads
        )

    def validate_updated_config(self, config, best_params: BestParams):
        """Validate that the configuration has been updated correctly.
        
        Args:
            config: The configuration object to validate
            best_params: The BestParams object containing expected values
        """
        # Validate model parameters
        for key, expected_value in best_params.model_params.items():
            actual_value = getattr(config.model, key, None)
            assert actual_value == expected_value, f"Model parameter '{key}' mismatch: expected {expected_value}, got {actual_value}"
            logger.debug(f"Validated model parameter '{key}': {actual_value} == {expected_value}")
        
        # Validate training parameters
        for key, expected_value in best_params.training_params.items():
            if key != 'max_epochs':  # Skip 'max_epochs' as it's not overwritten
                actual_value = getattr(config.training, key, None)
                assert actual_value == expected_value, f"Training parameter '{key}' mismatch: expected {expected_value}, got {actual_value}"
                logger.debug(f"Validated training parameter '{key}': {actual_value} == {expected_value}")

    def log_updated_config(self, config):
        """Log the entire updated configuration for verification.
        
        Args:
            config: The configuration object to log
        """
        try:
            config_str = json.dumps({
                'model': config.model.__dict__,
                'training': config.training.__dict__,
                'use_best_params': config.use_best_params,
                'optuna': {
                    'study_name': config.optuna.study_name,
                    'storage_url': config.optuna.storage_url
                }
            }, indent=2)
            logger.info("Updated Configuration:\n" + config_str)
        except Exception as e:
            logger.error(f"Failed to log updated configuration: {str(e)}")
