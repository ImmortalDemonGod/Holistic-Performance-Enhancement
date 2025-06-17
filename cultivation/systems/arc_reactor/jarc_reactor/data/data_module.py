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
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """
        Prepares and assigns the training and validation datasets for the data module.

        Attempts to load datasets using the provided configuration. Logs the number of samples loaded for each dataset. If dataset preparation fails, logs the error and re-raises the exception.
        """
        # train_dataset, val_dataset, test_dataset are initialized to None in __init__
        try:
            datasets = prepare_data(
                cfg=self.cfg,
                return_datasets=True
            )

            num_datasets = len(datasets) if isinstance(datasets, (list, tuple)) else 0

            if num_datasets == 3:
                self.train_dataset, self.val_dataset, self.test_dataset = datasets
                self.logger.info(
                    f"Successfully loaded data: {len(self.train_dataset)} train, "
                    f"{len(self.val_dataset)} val, {len(self.test_dataset)} test samples."
                )
            elif num_datasets == 2:
                self.train_dataset, self.val_dataset = datasets
                # self.test_dataset remains None (initialized in __init__)
                self.logger.info(
                    f"Successfully loaded data: {len(self.train_dataset)} train, "
                    f"{len(self.val_dataset)} val samples. No test dataset provided."
                )
            else:
                self.logger.error(f"prepare_data returned an unexpected result: {datasets}. Expected 2 or 3 datasets.")
                raise ValueError(f"prepare_data returned an unexpected number/type of datasets: got {num_datasets}, expected 2 or 3.")

        except Exception as e:
            self.logger.error(f"Failed to setup data module: {str(e)}", exc_info=True)
            # Ensure datasets are reset on failure to prevent inconsistent state
            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None
            raise

    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset with shuffling enabled.

        The DataLoader uses the batch size specified in the configuration and configurable worker processes for data loading.
        """
        dl_cfg = self.cfg.dataloader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=dl_cfg.num_workers,
            pin_memory=dl_cfg.pin_memory,
            drop_last=dl_cfg.drop_last_train
        )

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.

        The DataLoader uses the configured batch size, does not shuffle the data, and uses configurable worker processes.
        """
        dl_cfg = self.cfg.dataloader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=dl_cfg.num_workers,
            pin_memory=dl_cfg.pin_memory,
            drop_last=dl_cfg.drop_last_eval
        )

    def test_dataloader(self):
        """
        Returns a DataLoader for the test dataset, falling back to validation dataset if needed.

        The DataLoader uses the configured batch size, does not shuffle the data, and uses configurable worker processes.
        Logs appropriate warnings when falling back to validation dataset.
        """
        dl_cfg = self.cfg.dataloader
        if self.test_dataset is not None:
            self.logger.info(f"Using dedicated test_dataset for testing with {len(self.test_dataset)} samples.")
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=dl_cfg.num_workers,
                pin_memory=dl_cfg.pin_memory,
                drop_last=dl_cfg.drop_last_eval
            )
        elif self.val_dataset is not None:
            self.logger.warning(
                "Test dataset not available or not loaded. "
                "Falling back to use validation dataset for testing. "
                "This might not be the intended behavior for evaluation."
            )
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=dl_cfg.num_workers,
                pin_memory=dl_cfg.pin_memory,
                drop_last=dl_cfg.drop_last_eval
            )
        else:
            self.logger.error("Neither test nor validation dataset is available for test_dataloader.")
            raise RuntimeError("No data available for test_dataloader. Both test_dataset and val_dataset are None.")
