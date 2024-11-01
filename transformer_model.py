
import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_activation import CustomSigmoidActivation
from positional_encoding import PositionalEncoding
from config import dropout_rate  # Import the dropout rate from config

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, N, heads, d_ff, output_dim):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, heads, d_ff)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N)
        self.output_fc = nn.Linear(d_model, output_dim)

        self.custom_activation = CustomSigmoidActivation(min_value=-1, max_value=9)
        self.dropout = nn.Dropout(dropout_rate)  # Define a dropout layer

    def forward(self, x):
        # Add an additional dimension for sequence length of 1
        # Add an additional dimension for sequence length of 1
        x = self.input_fc(x)
        x = self.dropout(x)  # Apply dropout after the input fully connected layer
        x = x.view(x.size(0), -1, self.input_fc.out_features)  # Reshape to [batch_size, sequence_length, d_model]

        # Apply positional encoding
        x = self.positional_encoding(x)

        # Transformer Encoder expects input shape as [sequence_length, batch_size, d_model]
        x = x.transpose(0, 1)  # Shape becomes [1, batch_size, d_model]

        # Pass through Transformer Encoder
        x = self.encoder(x)

        # Reshape back and pass through final fully connected layer
        x = x.transpose(0, 1).squeeze(1)  # Shape becomes [batch_size, d_model]
        x = self.dropout(x)  # Apply dropout before the final fully connected layer
        x = self.output_fc(x)  # Shape becomes [batch_size, output_dim]

        # Apply custom activation to constrain output
        x = self.custom_activation(x)
        return x
