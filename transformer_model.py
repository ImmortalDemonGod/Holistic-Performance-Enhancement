
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
from config import dropout_rate, encoder_layers, decoder_layers
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from Utils.positional_encoding import PositionalEncoding
from Utils.context_encoder import ContextEncoderModule

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, encoder_layers, decoder_layers, heads, d_ff, output_dim):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        # Conditionally create the encoder
        if encoder_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(d_model, heads, d_ff, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        else:
            print("No context embedding available.")  # Indicate absence of context embedding
            self.encoder = None
        
        # Conditionally create the decoder
        if decoder_layers > 0:
            decoder_layer = nn.TransformerDecoderLayer(d_model, heads, d_ff, batch_first=True)
            self.decoder = TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        else:
            self.decoder = None
        
        # Add Context Encoder
        self.context_encoder = ContextEncoderModule(d_model, heads)
        
        # Context Integration Layer
        self.context_integration = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
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
        print(f"Quantized src shape: {src.shape}")  # Print shape of quantized src
        tgt = self.quant(tgt)  # Quantize target
        print(f"Quantized tgt shape: {tgt.shape}")  # Print shape of quantized tgt

        # Process context if available
        if ctx_input is not None and ctx_output is not None:
            ctx_input = self.quant(ctx_input)
            ctx_output = self.quant(ctx_output)
            context_embedding = self.context_encoder(ctx_input, ctx_output)
        else:
            context_embedding = None

        # Process main input
        x = self.input_fc(src)
        print(f"After input_fc (Linear layer: {src.shape[-1]} → {self.input_fc.out_features}), x shape: {x.shape}")
        x = self.positional_encoding(x)
        print(f"After positional_encoding (adds positional information), x shape: {x.shape}")

        # Integrate context if available
        if context_embedding is not None:
            # Expand context to match input dimensions
            context_expanded = context_embedding.unsqueeze(1).expand(-1, x.size(1), -1)
            print(f"Context expanded shape: {context_expanded.shape}")  # Print shape of context_expanded
            x = self.context_integration(
                torch.cat([x, context_expanded], dim=-1)
            )
            print(f"After context_integration (Concatenated with context_embedding and transformed from {x.shape[-1] * 2} → {x.shape[-1]}), x shape: {x.shape}")
        src = self.input_fc(src)
        src = self.positional_encoding(src)
        src = self.dropout(src)
        src = src.transpose(0, 1)  # Shape becomes [sequence_length, batch_size, d_model]
        print(f"Shape before encoder: {src.shape}")  # Print shape before encoder
        if self.encoder is not None:
            memory = self.encoder(src)
            print(f"Memory shape after encoder (TransformerEncoder applied), memory shape: {memory.shape}")
        else:
            memory = src  # If no encoder, pass input directly to decoder
            print(f"Memory shape (no encoder applied), memory shape: {memory.shape}")
        
        # Decoder
        tgt = self.input_fc(tgt)
        tgt = self.positional_encoding(tgt)
        tgt = self.dropout(tgt)
        tgt = tgt.transpose(0, 1)  # Shape becomes [sequence_length, batch_size, d_model]
        
        print(f"Shape before decoder: {tgt.shape}")  # Print shape before decoder
        if self.decoder is not None:
            output = self.decoder(tgt, memory)
            print(f"Output shape after decoder (TransformerDecoder applied), output shape: {output.shape}")
        else:
            output = memory  # If no decoder, use memory directly
            print(f"Output shape (no decoder applied), output shape: {output.shape}")
        
        output = output.transpose(0, 1)  # Shape back to [batch_size, sequence_length, d_model]
        output = output.transpose(0, 1)
        print(f"After transpose (Shape changed to [Batch Size, Sequence Length, Feature Size]): {output.shape}")

        output = self.output_fc(output)
        print(f"After output_fc (Linear layer: {output.shape[2]} → {self.output_fc.out_features}), output shape: {output.shape}")

        output = self.dequant(output)
        print(f"After dequant (Dequantized output), output shape: {output.shape}")
        output = self.dequant(output)  # Dequantize output
        return output
