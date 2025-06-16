from torch.utils.data import DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig # Added for Hydra
from cultivation.systems.arc_reactor.jarc_reactor.data.data_preparation import prepare_data
import logging
class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg # Store Hydra config
        self.batch_size = cfg.training.batch_size # Get batch_size from cfg
        self.logger = logging.getLogger(__name__)

    def setup(self, stage=None):
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)