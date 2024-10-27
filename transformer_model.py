
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encoding import PositionalEncoding

class DiffTransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, N, heads, d_ff, output_dim, lambda_init=0.5):
        super(DiffTransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.context_embedding = nn.Linear(d_model, d_model)
        self.positional_encoding = PositionalEncoding(d_model, mode='2d')  # Updated for 2D encoding
        self.d_model = d_model  # Store d_model as a class attribute for later use
        self.lambda_init = lambda_init  # Lambda for differential attention balance

        encoder_layer = nn.TransformerEncoderLayer(d_model, heads, d_ff)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N)
        self.output_fc = nn.Linear(d_model, output_dim)

    def diff_attention(self, X, W_q, W_k, W_v):
        """
        Implements Differential Attention mechanism.
        """
        # Split the input projections into two groups
        Q1, Q2 = torch.chunk(X @ W_q, 2, dim=-1)
        K1, K2 = torch.chunk(X @ W_k, 2, dim=-1)
        V = X @ W_v

        # Calculate the attention scores
        s = 1 / (self.d_model ** 0.5)  # scaling factor
        A1 = torch.matmul(Q1, K1.transpose(-1, -2)) * s
        A2 = torch.matmul(Q2, K2.transpose(-1, -2)) * s

        # Apply differential attention
        diff_attention_scores = torch.softmax(A1, dim=-1) - self.lambda_init * torch.softmax(A2, dim=-1)

        # Compute the output
        return torch.matmul(diff_attention_scores, V)

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

        # Pass through Transformer Encoder using differential attention
        for layer in self.encoder.layers:
            # Use self.d_model instead of d_model to ensure it's correctly referenced
            W_q, W_k, W_v = layer.self_attn.in_proj_weight.split(self.d_model, dim=0)
            x = self.diff_attention(x, W_q, W_k, W_v)

        # Reshape back and pass through final fully connected layer
        x = x.transpose(0, 1).squeeze(1)  # Shape becomes [batch_size, d_model]
        x = self.output_fc(x)  # Shape becomes [batch_size, output_dim]

        return x
