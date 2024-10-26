
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encoding import PositionalEncoding

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, N, heads, d_ff, output_dim):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, heads, d_ff)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N)
        self.output_fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # Add an additional dimension for sequence length of 1
        x = self.input_fc(x)
        x = x.unsqueeze(1)  # Shape becomes [batch_size, 1, d_model]

        # Apply positional encoding
        x = self.positional_encoding(x)

        # Transformer Encoder expects input shape as [sequence_length, batch_size, d_model]
        x = x.transpose(0, 1)  # Shape becomes [1, batch_size, d_model]

        # Pass through Transformer Encoder
        x = self.encoder(x)

        # Reshape back and pass through final fully connected layer
        x = x.transpose(0, 1).squeeze(1)  # Shape becomes [batch_size, d_model]
        x = self.output_fc(x)  # Shape becomes [batch_size, output_dim]

        return x
