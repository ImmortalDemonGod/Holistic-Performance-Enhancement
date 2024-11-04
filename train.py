from config import include_sythtraining_data
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformer_model import TransformerModel
from config import *
import torch.nn.functional as F
import os
import json
from Utils.padding_utils import pad_to_fixed_size
import torch
from Utils.data_preparation import prepare_data

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerTrainer(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        d_model,
        encoder_layers,
        decoder_layers,
        heads,
        d_ff,
        output_dim,
        learning_rate,
        include_sythtraining_data,
    ):
        super(TransformerTrainer, self).__init__()
        self.save_hyperparameters()
        self.model = TransformerModel(
            input_dim=self.hparams.input_dim,
            d_model=self.hparams.d_model,
            encoder_layers=self.hparams.encoder_layers,
            decoder_layers=self.hparams.decoder_layers,
            heads=self.hparams.heads,
            d_ff=self.hparams.d_ff,
            output_dim=self.hparams.output_dim,
        )
        self.learning_rate = self.hparams.learning_rate
        self.device_choice = 'cpu'
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, src, tgt):
        return self.model(src.to("cpu"), tgt.to("cpu"))

    def training_step(self, batch, batch_idx):
        src, tgt, _ = batch  # Unpack three elements, ignore task_id
        y_hat = self(src, tgt)
        # Debugging: Print shapes of y_hat and tgt
        print(f"y_hat shape: {y_hat.shape}, tgt shape: {tgt.shape}")

        # Reshape y_hat to match the target's shape
        y_hat = y_hat.view(-1, 11)
        tgt = tgt.view(-1, 30)[:, 0].long()
        # Mask out padding values in the target
        valid_indices = tgt != -1
        y_hat = y_hat[valid_indices]
        tgt = tgt[valid_indices]

        # Mask out padding values in the target
        valid_indices = tgt != -1
        y_hat = y_hat[valid_indices]
        tgt = tgt[valid_indices]

        loss = self.criterion(y_hat, tgt)
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt, _ = batch  # Unpack three elements, ignore task_id
        y_hat = self(src, tgt)

        # Debugging: Print shapes of y_hat and tgt
        print(f"y_hat shape: {y_hat.shape}, tgt shape: {tgt.shape}")

        # Reshape y_hat to match the target's shape
        y_hat = y_hat.view(-1, 11)
        tgt = tgt.view(-1, 30)[:, 0].long()
        # Convert logits to class predictions using argmax
        predictions = torch.argmax(y_hat, dim=-1)  # Shape: [batch_size * 30]

        # Compute loss
        loss = self.criterion(y_hat, tgt)
        self.log('val_loss', loss, prog_bar=True)

        # Optionally, log or return predictions for further evaluation
        # For example, you might want to calculate accuracy or other metrics
        return {'val_loss': loss, 'predictions': predictions, 'targets': tgt}


    def test_step(self, batch, batch_idx):
        src, tgt, task_ids = batch  # Ensure task_ids are included in the batch
        y_hat = self(src, tgt)

        # Compute accuracy (modify according to your specific task)
        threshold = 0.1
        correct = (torch.abs(y_hat - tgt) < threshold).float()
        accuracy = correct.mean()

        self.log("test_accuracy", accuracy, prog_bar=True)
        return accuracy


def prepare_data():
    train_inputs, train_outputs, train_task_ids = [], [], []
    test_inputs, test_outputs, test_task_ids = [], [], []

    # Iterate over all JSON files in the 'training' directory
    for filename in os.listdir("training"):
        if filename.endswith(".json"):
            # Extract task_id from filename (e.g., 'task_1.json' -> 'task_1')
            task_id = os.path.splitext(filename)[0]

            # logger.info(f"Loading training data for task_id: {task_id} from file: {filename}")
            with open(os.path.join("training", filename), "r") as f:
                data = json.load(f)

            # Extract and pad training data
            for item in data["train"]:
                input_tensor = pad_to_fixed_size(
                    torch.tensor(item["input"], dtype=torch.float32),
                    target_shape=(30, 30),
                )
                output_tensor = pad_to_fixed_size(
                    torch.tensor(item["output"], dtype=torch.float32),
                    target_shape=(30, 30),
                )
                train_inputs.append(input_tensor)
                train_outputs.append(output_tensor)

                # Assign task_id based on filename
                train_task_ids.append(task_id)

            # Extract and pad test data
            for item in data["test"]:
                input_tensor = pad_to_fixed_size(
                    torch.tensor(item["input"], dtype=torch.float32),
                    target_shape=(30, 30),
                )
                output_tensor = pad_to_fixed_size(
                    torch.tensor(item["output"], dtype=torch.float32),
                    target_shape=(30, 30),
                )
                test_inputs.append(input_tensor)
                test_outputs.append(output_tensor)

                # Assign task_id based on filename
                test_task_ids.append(task_id)

    # Conditionally load data from the 'sythtraining' directory
    if include_sythtraining_data:
        logger.info("Including synthetic training data from 'sythtraining' directory.")
        for filename in os.listdir("sythtraining"):
            if filename.endswith(".json"):
                # Extract task_id from filename
                task_id = os.path.splitext(filename)[0]

                logger.info(f"Loading synthetic training data for task_id: {task_id}")
                with open(os.path.join("sythtraining", filename), "r") as f:
                    data = json.load(f)

                # Extract and pad data
                for item in data:
                    input_tensor = pad_to_fixed_size(
                        torch.tensor(item["input"], dtype=torch.float32),
                        target_shape=(30, 30),
                    )
                    output_tensor = pad_to_fixed_size(
                        torch.tensor(item["output"], dtype=torch.float32),
                        target_shape=(30, 30),
                    )
                    train_inputs.append(input_tensor)
                    train_outputs.append(output_tensor)

                    # Assign task_id based on filename
                    train_task_ids.append(task_id)
    # Stack inputs, outputs, and task_ids
    train_inputs = torch.stack(train_inputs)
    train_outputs = torch.stack(train_outputs)
    test_inputs = torch.stack(test_inputs)
    test_outputs = torch.stack(test_outputs)

    # Create a sorted list of unique task_ids
    unique_task_ids = sorted(set(train_task_ids + test_task_ids))

    logger.info(
        f"Total unique task_ids (including synthetic if any): {len(unique_task_ids)}"
    )
    if len(unique_task_ids) != (len(set(train_task_ids)) + len(set(test_task_ids))):
        logger.warning(
            "There are overlapping task_ids between training and test datasets."
        )
    task_id_map = {task_id: idx for idx, task_id in enumerate(unique_task_ids)}

    # Encode task_ids as integers using the mapping
    train_task_ids_encoded = [task_id_map[tid] for tid in train_task_ids]
    test_task_ids_encoded = [task_id_map[tid] for tid in test_task_ids]

    # Convert the encoded task_ids to tensors
    train_task_ids_tensor = torch.tensor(train_task_ids_encoded, dtype=torch.long)
    test_task_ids_tensor = torch.tensor(test_task_ids_encoded, dtype=torch.long)

    # Create TensorDatasets with encoded task_ids
    train_dataset = TensorDataset(train_inputs, train_outputs, train_task_ids_tensor)
    test_dataset = TensorDataset(test_inputs, test_outputs, test_task_ids_tensor)

    # Optional: Save the task_id_map
    logger.info("Saved task_id_map.json with the current task mappings.")
    with open("task_id_map.json", "w") as f:
        json.dump(task_id_map, f)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader

