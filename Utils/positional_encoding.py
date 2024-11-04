# Utils/positional_encoding.py
import math
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    # Implementation of Positional Encoding with Mask Awareness
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class Grid2DPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_height=30, max_width=30):  # Ensure defaults are 30x30
        super().__init__()
        self.d_model = d_model
        
        # Create position encoding matrices
        pe = torch.zeros(max_height * max_width, d_model)
        
        # Generate positions for both dimensions
        pos_h = torch.arange(0, max_height).unsqueeze(1)
        pos_w = torch.arange(0, max_width).unsqueeze(1)
        
        # Create all combinations of positions
        pos_grid_h = pos_h.repeat(max_width, 1)
        pos_grid_w = pos_w.repeat(max_height, 1).t().reshape(-1, 1)
        
        # Compute division terms
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # Compute sinusoidal encodings for both dimensions
        # Use first half of dimensions for height, second half for width
        pe[:, 0:d_model:4] = torch.sin(pos_grid_h * div_term[::2])
        pe[:, 1:d_model:4] = torch.cos(pos_grid_h * div_term[::2])
        pe[:, 2:d_model:4] = torch.sin(pos_grid_w * div_term[1::2])
        pe[:, 3:d_model:4] = torch.cos(pos_grid_w * div_term[1::2])
        
        # Reshape to match our model's expected input
        pe = pe.view(1, max_height * max_width, d_model)
        
        # Register as buffer
        self.register_buffer('pe', pe)
        
        logger.info(f"Initialized 2D Grid Positional Encoding: {pe.shape}")

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
                where seq_len is height * width
        """
        seq_len = x.size(1)
        # Add positional encoding
        x = x + self.pe[:, :seq_len]
        return x
