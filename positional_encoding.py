
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, mode='1d'):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.mode = mode
        self.max_len = max_len

    def forward(self, x):
        if self.mode == '1d':
            batch_size, seq_len, d_model = x.size()
            assert d_model == self.d_model, "Input feature dimension must match positional encoding dimension"
            pe = torch.zeros(seq_len, d_model, device=x.device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)
            x = x + pe

        elif self.mode == '2d':
            batch_size, height, d_model = x.size()
            assert d_model == self.d_model, "Input feature dimension must match positional encoding dimension"
            
            # Assuming a rectangular grid (not necessarily square)
            pe = torch.zeros(height, d_model, device=x.device)
            position_h = torch.arange(0, height, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
            pe[:, 0::2] = torch.sin(position_h * div_term)
            pe[:, 1::2] = torch.cos(position_h * div_term)
            pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)
            x = x + pe

        else:
            raise ValueError("Unsupported mode for PositionalEncoding: {}".format(self.mode))

        return x
