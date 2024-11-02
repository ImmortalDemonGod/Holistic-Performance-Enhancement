
from config import include_sythtraining_data
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
        self.model = TransformerModel(input_dim, d_model, encoder_layers, decoder_layers, heads, d_ff, output_dim)
        self.learning_rate = learning_rate
        self.device_choice = 'cpu'

    def forward(self, src, tgt):
        return self.model(src.to('cpu'), tgt.to('cpu'))

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        y_hat = self(src, tgt)
        loss = F.mse_loss(y_hat, tgt)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        y_hat = self(src, tgt)
        loss = F.mse_loss(y_hat, tgt)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        src, tgt, task_ids = batch  # Ensure task_ids are included in the batch
        y_hat = self(src, tgt)
        
        # Compute accuracy (modify according to your specific task)
        threshold = 0.1
        correct = (torch.abs(y_hat - tgt) < threshold).float()
        accuracy = correct.mean()
        
        self.log('test_accuracy', accuracy, prog_bar=True)
        return accuracy


def prepare_data():
    train_inputs, train_outputs, train_task_ids = [], [], []
    test_inputs, test_outputs, test_task_ids = [], [], []

    # Iterate over all JSON files in the 'training' directory
    for filename in os.listdir('training'):
        if filename.endswith('.json'):
            # Extract task_id from filename (e.g., 'task_1.json' -> 'task_1')
            task_id = os.path.splitext(filename)[0]

            with open(os.path.join('training', filename), 'r') as f:
                data = json.load(f)

            # Extract and pad training data
            for item in data['train']:
                input_tensor = pad_to_fixed_size(torch.tensor(item['input'], dtype=torch.float32), target_shape=(30, 30))
                output_tensor = pad_to_fixed_size(torch.tensor(item['output'], dtype=torch.float32), target_shape=(30, 30))
                train_inputs.append(input_tensor)
                train_outputs.append(output_tensor)

                # Assign task_id based on filename
                train_task_ids.append(task_id)

            # Extract and pad test data
            for item in data['test']:
                input_tensor = pad_to_fixed_size(torch.tensor(item['input'], dtype=torch.float32), target_shape=(30, 30))
                output_tensor = pad_to_fixed_size(torch.tensor(item['output'], dtype=torch.float32), target_shape=(30, 30))
                test_inputs.append(input_tensor)
                test_outputs.append(output_tensor)

                # Assign task_id based on filename
                test_task_ids.append(task_id)

    # Conditionally load data from the 'sythtraining' directory
    if include_sythtraining_data:
        for filename in os.listdir('sythtraining'):
            if filename.endswith('.json'):
                # Extract task_id from filename
                task_id = os.path.splitext(filename)[0]

                with open(os.path.join('sythtraining', filename), 'r') as f:
                    data = json.load(f)

                # Extract and pad data
                for item in data:
                    input_tensor = pad_to_fixed_size(torch.tensor(item['input'], dtype=torch.float32), target_shape=(30, 30))
                    output_tensor = pad_to_fixed_size(torch.tensor(item['output'], dtype=torch.float32), target_shape=(30, 30))
                    train_inputs.append(input_tensor)
                    train_outputs.append(output_tensor)

                    # Assign task_id based on filename
                    train_task_ids.append(task_id)
    # Stack inputs, outputs, and task_ids
    train_inputs = torch.stack(train_inputs)
    train_outputs = torch.stack(train_outputs)
    test_inputs = torch.stack(test_inputs)
    test_outputs = torch.stack(test_outputs)

    # Create TensorDatasets with source, target pairs, and task_ids
    train_dataset = TensorDataset(train_inputs, train_outputs, torch.tensor(train_task_ids))
    test_dataset = TensorDataset(test_inputs, test_outputs, torch.tensor(test_task_ids))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
