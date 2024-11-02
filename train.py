
import torch
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
        self.model = TransformerModel(input_dim, d_model, N, heads, d_ff, output_dim)
        self.learning_rate = learning_rate
        self.device_choice = 'cpu'

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

    # Convert lists to tensors
    train_inputs = torch.stack(train_inputs)
    train_outputs = torch.stack(train_outputs)
    test_inputs = torch.stack(test_inputs)
    test_outputs = torch.stack(test_outputs)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_inputs, train_outputs)
    test_dataset = TensorDataset(test_inputs, test_outputs)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
