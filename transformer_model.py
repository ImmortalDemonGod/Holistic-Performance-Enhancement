# transformer_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
from config import dropout_rate, encoder_layers, decoder_layers
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoderLayer, TransformerEncoder
from Utils.positional_encoding import Grid2DPositionalEncoding
from Utils.context_encoder import ContextEncoderModule
from config import context_encoder_d_model, context_encoder_heads

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, encoder_layers, decoder_layers, heads, d_ff, output_dim, dropout_rate, context_encoder_d_model, context_encoder_heads):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.positional_encoding = Grid2DPositionalEncoding(d_model, max_height=30, max_width=30)
        # Conditionally create the encoder
        if encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(d_model, heads, d_ff, batch_first=True)
            self.encoder = TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        else:
            print("No context embedding available.")  # Indicate absence of context embedding
            self.encoder = None
        
        # Conditionally create the decoder
        if decoder_layers > 0:
            decoder_layer = TransformerDecoderLayer(d_model, heads, d_ff, batch_first=True)
            self.decoder = TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        else:
            self.decoder = None
        
        # Add Context Encoder with optimized parameters
        self.context_encoder = ContextEncoderModule(
            d_model=context_encoder_d_model,
            heads=context_encoder_heads
        )
        
        # Store context_encoder_d_model as an instance variable
        self.context_encoder_d_model = context_encoder_d_model

        # Context Integration Layer
        self.context_integration = nn.Sequential(
            nn.Linear(d_model + context_encoder_d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        self.output_fc = nn.Linear(d_model, 11)
        self.dropout = nn.Dropout(p=dropout_rate)



        # Add quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, src, tgt, ctx_input=None, ctx_output=None):
        src = self.quant(src)  # Quantize input
        # print(f"Quantized src shape: {src.shape} (Batch Size: {src.shape[0]}, Sequence Length: {src.shape[1]}, Feature Size: {src.shape[2]})")
        tgt = self.quant(tgt)  # Quantize target
        # print(f"Quantized tgt shape: {tgt.shape} (Batch Size: {tgt.shape[0]}, Sequence Length: {tgt.shape[1]}, Feature Size: {tgt.shape[2]})")

        # Process context if available
        if ctx_input is not None and ctx_output is not None:
            ctx_input = self.quant(ctx_input)
            ctx_output = self.quant(ctx_output)
            context_embedding = self.context_encoder(ctx_input, ctx_output)
        else:
            context_embedding = None

        # Process main input
        x = self.input_fc(src)
        # print(f"After input_fc (Linear layer: {src.shape[-1]} → {self.input_fc.out_features}), x shape: {x.shape}")
        x = self.positional_encoding(x)
        # print(f"After positional_encoding (adds positional information), x shape: {x.shape}")

        # Integrate context if available
        if context_embedding is not None:
            # Expand context to match input dimensions
            context_expanded = context_embedding.unsqueeze(1).expand(-1, x.size(1), -1)
            # print(f"Context expanded shape: {context_expanded.shape}")  # Print shape of context_expanded
            x = self.context_integration(
                torch.cat([x, context_expanded], dim=-1)
            )
            # print(f"After context_integration (Concatenated with context_embedding and transformed from {x.shape[-1] * 2} → {x.shape[-1]}), x shape: {x.shape}")
        # Reshape input for 2D positional encoding
        batch_size, seq_len, _ = src.shape
        x = self.input_fc(src)
        x = self.positional_encoding(x)  # Now uses 2D positional encoding
        
        # Context integration
        if context_embedding is not None:
            context_expanded = context_embedding.unsqueeze(1).expand(-1, x.size(1), -1)
            x = self.context_integration(torch.cat([x, context_expanded], dim=-1))
        
        # Rest of the processing remains the same
        x = self.dropout(x)
        x = x.transpose(0, 1)
        
        if self.encoder is not None:
            memory = self.encoder(x)
        else:
            memory = x
        
        if self.decoder is not None:
            tgt = self.input_fc(tgt)
            tgt = self.positional_encoding(tgt)
            tgt = self.dropout(tgt)
            tgt = tgt.transpose(0, 1)
            output = self.decoder(tgt, memory)
        else:
            output = memory
        
        output = output.transpose(0, 1)
        output = self.output_fc(output)
        return self.dequant(output)
