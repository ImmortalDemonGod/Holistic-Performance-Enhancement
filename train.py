
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformer_model import DiffTransformerModel
from config import *
import torch.nn.functional as F

class DiffTransformerTrainer(pl.LightningModule):
    def __init__(self):
        super(DiffTransformerTrainer, self).__init__()
        self.model = DiffTransformerModel(input_dim, d_model, N, heads, d_ff, output_dim, lambda_init)
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

def prepare_data():
    x_data = torch.rand((1000, input_dim))
    y_data = torch.rand((1000, output_dim))
    dataset = TensorDataset(x_data, y_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader
