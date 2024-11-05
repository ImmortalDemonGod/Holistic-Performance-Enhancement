import json
import logging
import optuna
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
        """Update config with best parameters while preserving config-specified values.
        
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

            # Update model parameters
            for key, value in best_params.model_params.items():
                if not hasattr(config.model, key) or getattr(config.model, key) is None:
                    setattr(config.model, key, value)
                    logger.debug(f"Updated model parameter {key}={value}")
            
            # Update training parameters
            for key, value in best_params.training_params.items():
                if not hasattr(config.training, key) or getattr(config.training, key) is None:
                    setattr(config.training, key, value)
                    logger.debug(f"Updated training parameter {key}={value}")
            
            logger.info("Configuration updated successfully")
            return True
                
        except Exception as e:
            logger.error(f"Failed to update config: {str(e)}")
            return False
