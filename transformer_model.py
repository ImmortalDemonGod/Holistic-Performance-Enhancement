
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
from config import dropout_rate
from custom_activation import CustomSigmoidActivation
from positional_encoding import PositionalEncoding

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, N, heads, d_ff, output_dim):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=heads, 
            dim_feedforward=d_ff, 
            dropout=0.1,
            norm_first=True,  # Normalize before attention and feedforward
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N)
        self.output_fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.custom_activation = CustomSigmoidActivation(min_value=-1, max_value=9)


        # Add quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)  # Quantize input
        x = self.input_fc(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        x = x.transpose(0, 1)  # Shape becomes [sequence_length, batch_size, d_model]
        x = self.encoder(x)
        x = x.transpose(0, 1)  # Shape back to [batch_size, sequence_length, d_model]
        x = self.output_fc(x)
        x = self.custom_activation(x)
        x = self.dequant(x)  # Dequantize output
        return x
