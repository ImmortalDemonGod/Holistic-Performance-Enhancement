import torch
from torch.utils.data import DataLoader
import logging
from pathlib import Path

class EvaluationDataManager:
    def __init__(self, config, debug_logger):
        self.config = config
        self.debug = debug_logger
        self.logger = logging.getLogger('evaluation_data')
        
        # Store task ID mappings
        self.train_task_map = {}
        self.eval_task_map = {}
    
    def _log_dataset_info(self, dataset, name: str):
        """Log detailed information about a dataset"""
        self.debug.logger.debug(f"\n{'='*50}\nDataset Info: {name}")
        self.debug.logger.debug(f"Dataset length: {len(dataset)}")
        
        # Log first batch
        sample_batch = next(iter(DataLoader(dataset, batch_size=1)))
        self.debug.log_batch(sample_batch, f'{name}_sample')
    
    def get_training_data(self):
        """Load training data with detailed logging"""
        self.debug.logger.debug("Loading training data...")
        
        try:
            from jarc_reactor.data.data_preparation import prepare_data
            train_dataset, val_dataset = prepare_data(
                return_datasets=True
            )
            
            # Log dataset information
            self._log_dataset_info(train_dataset, 'training')
            self._log_dataset_info(val_dataset, 'validation')
            
            # Load task mapping
            try:
                with open('task_id_map.json', 'r') as f:
                    self.train_task_map = json.load(f)
                self.debug.logger.debug(
                    f"Loaded training task map with {len(self.train_task_map)} tasks"
                )
            except Exception as e:
                self.debug.logger.error(f"Failed to load task_id_map.json: {e}")
            
            return train_dataset, val_dataset
            
        except Exception as e:
            self.debug.logger.error(f"Error loading training data: {str(e)}")
            raise
    
    def get_evaluation_data(self):
        """Load evaluation data with detailed logging"""
        self.debug.logger.debug("Loading evaluation data...")
        
        try:
            from jarc_reactor.data.eval_data_prep import prepare_data as prepare_eval_data
            _, eval_dataset = prepare_eval_data(
                return_datasets=True
            )
            
            self._log_dataset_info(eval_dataset, 'evaluation')
            
            # Load evaluation task mapping
            try:
                with open('eval_id_map.json', 'r') as f:
                    self.eval_task_map = json.load(f)
                self.debug.logger.debug(
                    f"Loaded evaluation task map with {len(self.eval_task_map)} tasks"
                )
            except Exception as e:
                self.debug.logger.error(f"Failed to load eval_id_map.json: {e}")
            
            return eval_dataset
            
        except Exception as e:
            self.debug.logger.error(f"Error loading evaluation data: {str(e)}")
            raise