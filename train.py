
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformer_model import TransformerModel
from config import *
import torch.nn.functional as F
import os
import json
from padding_utils import pad_to_fixed_size

# Sample Training Module

class TransformerTrainer(pl.LightningModule):
    def __init__(self):
        super(TransformerTrainer, self).__init__()
        self.model = TransformerModel(input_dim, d_model, N, heads, d_ff, output_dim)
        self.learning_rate = learning_rate
        self.device_choice = 'cpu'  # Ensure device is set to "cpu"

    def forward(self, x):
        return self.model(x.to('cpu'))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)

# Prepare training and validation data
def prepare_data():
    train_inputs, train_outputs = [], []
    val_inputs, val_outputs = [], []

    # Path to the training folder
    training_folder = 'training'

    # Iterate over each JSON file in the training folder
    for filename in os.listdir(training_folder):
        if filename.endswith('.json'):
            with open(os.path.join(training_folder, filename), 'r') as file:
                data = json.load(file)
                # Process training examples
                for example in data['train']:
                    input_tensor = torch.tensor(example['input'], dtype=torch.float32, device='cpu')
                    output_tensor = torch.tensor(example['output'], dtype=torch.float32, device='cpu')
                    padded_input = pad_to_fixed_size(input_tensor, pad_value=-1)
                    padded_output = pad_to_fixed_size(output_tensor, pad_value=-1)
                    train_inputs.append(padded_input)
                    train_outputs.append(padded_output)
                # Process test examples
                for example in data['test']:
                    input_tensor = torch.tensor(example['input'], dtype=torch.float32)
                    output_tensor = torch.tensor(example['output'], dtype=torch.float32)
                    padded_input = pad_to_fixed_size(input_tensor, pad_value=-1)
                    padded_output = pad_to_fixed_size(output_tensor, pad_value=-1)
                    val_inputs.append(padded_input)
                    val_outputs.append(padded_output)

    # Convert lists to tensors
    train_inputs = torch.stack(train_inputs)
    train_outputs = torch.stack(train_outputs)
    val_inputs = torch.stack(val_inputs)
    val_outputs = torch.stack(val_outputs)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_inputs, train_outputs)
    val_dataset = TensorDataset(val_inputs, val_outputs)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader
