# Step 1: Modifying train.py to include synthetic dataset logic for Gestalt principles

train_content_with_gestalt_data = '''
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformer_model import TransformerModel
from config import *
import torch.nn.functional as F

# Sample Training Module

class TransformerTrainer(pl.LightningModule):
    def __init__(self):
        super(TransformerTrainer, self).__init__()
        self.model = TransformerModel(input_dim, d_model, N, heads, d_ff, output_dim)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

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

# Prepare training and validation data with Gestalt Principles synthetic dataset
def prepare_data():
    # Function to generate synthetic data based on Gestalt Principles
    def generate_gestalt_data(num_samples, input_dim, output_dim):
        # Placeholder function to generate synthetic data with perceptual patterns (Closure, Proximity, etc.)
        # You can modify this function to generate specific visual patterns for each Gestalt principle
        x_data = torch.rand((num_samples, input_dim))
        y_data = torch.rand((num_samples, output_dim))
        return TensorDataset(x_data, y_data)

    # Generate synthetic data
    gestalt_dataset = generate_gestalt_data(1000, input_dim, output_dim)

    # Split data into training and validation sets
    train_size = int(0.8 * len(gestalt_dataset))
    val_size = len(gestalt_dataset) - train_size
    train_dataset, val_dataset = random_split(gestalt_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader
'''

with open(f'{directory_path}/train.py', 'w') as file:
    file.write(train_content_with_gestalt_data)

# Step 2: Updating transformer_model.py to include context embedding and 2D positional encoding
transformer_model_with_context = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encoding import PositionalEncoding

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, N, heads, d_ff, output_dim):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.context_embedding = nn.Linear(d_model, d_model)
        self.positional_encoding = PositionalEncoding(d_model, mode='2d')  # Updated for 2D encoding
        encoder_layer = nn.TransformerEncoderLayer(d_model, heads, d_ff)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N)
        self.output_fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # Apply input fully connected layer
        x = self.input_fc(x)

        # Context Embedding: Enrich input data with contextual information
        x = self.context_embedding(x)

        # Add an additional dimension for sequence length of 1 for positional encoding
        x = x.unsqueeze(1)  # Shape becomes [batch_size, 1, d_model]

        # Apply 2D positional encoding
        x = self.positional_encoding(x)

        # Transformer Encoder expects input shape as [sequence_length, batch_size, d_model]
        x = x.transpose(0, 1)  # Shape becomes [1, batch_size, d_model]

        # Pass through Transformer Encoder
        x = self.encoder(x)

        # Reshape back and pass through final fully connected layer
        x = x.transpose(0, 1).squeeze(1)  # Shape becomes [batch_size, d_model]
        x = self.output_fc(x)  # Shape becomes [batch_size, output_dim]

        return x
'''

with open(f'{directory_path}/transformer_model.py', 'w') as file:
    file.write(transformer_model_with_context)

# Step 3: Updating positional_encoding.py to support 2D positional encoding
positional_encoding_with_2d = '''
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, mode='1d'):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.mode = mode

        if mode == '1d':
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
        elif mode == '2d':
            # Assuming a square grid for simplicity
            max_len_sqrt = int(max_len ** 0.5)
            pe = torch.zeros(max_len_sqrt, max_len_sqrt, d_model)
            position_h = torch.arange(0, max_len_sqrt, dtype=torch.float).unsqueeze(1)
            position_w = torch.arange(0, max_len_sqrt, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
            for i in range(max_len_sqrt):
                pe[i, :, 0::2] = torch.sin(position_h * div_term)
                pe[:, i, 1::2] = torch.cos(position_w * div_term)
            pe = pe.unsqueeze(0)
        else:
            raise ValueError("Unsupported mode for PositionalEncoding: {}".format(mode))
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.mode == '1d':
            x = x + self.pe[:, :x.size(1), :]
        elif self.mode == '2d':
            batch_size, _, d_model = x.size()
            x = x + self.pe[:, :batch_size, :d_model].view_as(x)
        return x
'''

with open(f'{directory_path}/positional_encoding.py', 'w') as file:
    file.write(positional_encoding_with_2d)

# These updates provide context embedding, 2D positional encoding, and the ability to work with synthetic data inspired by Gestalt principles.

