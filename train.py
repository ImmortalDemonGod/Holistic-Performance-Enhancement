
from config import include_sythtraining_data
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
from padding_utils import pad_to_fixed_size
import torch
from torch.utils.data import DataLoader, TensorDataset
import json
import os
from config import batch_size

# Sample Training Module

class TransformerTrainer(pl.LightningModule):
    def __init__(self):
        super(TransformerTrainer, self).__init__()
        self.model = TransformerModel(input_dim, d_model, encoder_layers, decoder_layers, heads, d_ff, output_dim)
        self.learning_rate = learning_rate
        self.device_choice = 'cpu'
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, src, tgt):
        return self.model(src.to('cpu'), tgt.to('cpu'))

    def training_step(self, batch, batch_idx):
        src, tgt = batch
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
        src, tgt = batch
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

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)


def prepare_data():
    train_inputs, train_outputs = [], []
    test_inputs, test_outputs = [], []

    # Iterate over all JSON files in the 'training' directory
    for filename in os.listdir('training'):
        if filename.endswith('.json'):
            with open(os.path.join('training', filename), 'r') as f:
                data = json.load(f)

            # Extract and pad training data
            for item in data['train']:
                input_tensor = pad_to_fixed_size(torch.tensor(item['input'], dtype=torch.float32), target_shape=(30, 30))
                output_tensor = pad_to_fixed_size(torch.tensor(item['output'], dtype=torch.float32), target_shape=(30, 30))
                train_inputs.append(input_tensor)
                train_outputs.append(output_tensor)

            # Extract and pad test data
            for item in data['test']:
                input_tensor = pad_to_fixed_size(torch.tensor(item['input'], dtype=torch.float32), target_shape=(30, 30))
                output_tensor = pad_to_fixed_size(torch.tensor(item['output'], dtype=torch.float32), target_shape=(30, 30))
                test_inputs.append(input_tensor)
                test_outputs.append(output_tensor)

    # Conditionally load data from the 'sythtraining' directory
    if include_sythtraining_data:
        for filename in os.listdir('sythtraining'):
            if filename.endswith('.json'):
                with open(os.path.join('sythtraining', filename), 'r') as f:
                    data = json.load(f)

                # Extract and pad data
                for item in data:
                    input_tensor = pad_to_fixed_size(torch.tensor(item['input'], dtype=torch.float32), target_shape=(30, 30))
                    output_tensor = pad_to_fixed_size(torch.tensor(item['output'], dtype=torch.float32), target_shape=(30, 30))
                    train_inputs.append(input_tensor)
                    train_outputs.append(output_tensor)
    # Stack inputs and outputs
    train_inputs = torch.stack(train_inputs)
    train_outputs = torch.stack(train_outputs)
    test_inputs = torch.stack(test_inputs)
    test_outputs = torch.stack(test_outputs)

    # Create TensorDatasets with source and target pairs
    train_dataset = TensorDataset(train_inputs, train_outputs)
    test_dataset = TensorDataset(test_inputs, test_outputs)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
