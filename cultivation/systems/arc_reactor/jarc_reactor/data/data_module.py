from torch.utils.data import DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig # Added for Hydra
from cultivation.systems.arc_reactor.jarc_reactor.data.data_preparation import prepare_data
import logging
class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        """
        Initializes the data module with the provided Hydra configuration.
        
        Stores the configuration, extracts the batch size for data loading, and sets up a logger for the module.
        """
        super().__init__()
        self.cfg = cfg # Store Hydra config
        self.batch_size = cfg.training.batch_size # Get batch_size from cfg
        self.logger = logging.getLogger(__name__)

    def setup(self, stage=None):
        """
        Prepares and assigns the training and validation datasets for the data module.
        
        Attempts to load datasets using the provided configuration. Logs the number of samples loaded for each dataset. If dataset preparation fails, logs the error and re-raises the exception.
        """
        try:
            # Pass the full cfg object to prepare_data
            self.train_dataset, self.val_dataset = prepare_data(
                cfg=self.cfg,
                return_datasets=True
            )
            self.logger.info(f"Successfully loaded training data: {len(self.train_dataset)} train samples, {len(self.val_dataset)} validation samples")
        except Exception as e:
            self.logger.error(f"Failed to load training data: {str(e)}")
            raise

    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset with shuffling enabled.
        
        The DataLoader uses the batch size specified in the configuration and does not use worker processes for data loading.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.
        
        The DataLoader uses the configured batch size, does not shuffle the data, and operates with zero worker processes.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        """
        Returns a DataLoader for the validation dataset to be used as test data.
        
        The DataLoader uses the configured batch size, does not shuffle the data, and operates with a single worker process.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)