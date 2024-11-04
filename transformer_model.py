
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
        tgt = self.quant(tgt)  # Quantize target

        # Process context if available
        if ctx_input is not None and ctx_output is not None:
            ctx_input = self.quant(ctx_input)
            ctx_output = self.quant(ctx_output)
            context_embedding = self.context_encoder(ctx_input, ctx_output)
        else:
            context_embedding = None

        # Process main input
        x = self.input_fc(src)
        x = self.positional_encoding(x)

        # Integrate context if available
        if context_embedding is not None:
            # Expand context to match input dimensions
            context_expanded = context_embedding.unsqueeze(1).expand(-1, x.size(1), -1)
            x = self.context_integration(
                torch.cat([x, context_expanded], dim=-1)
            )
        src = self.input_fc(src)
        src = self.positional_encoding(src)
        src = self.dropout(src)
        src = src.transpose(0, 1)  # Shape becomes [sequence_length, batch_size, d_model]
        if self.encoder is not None:
            memory = self.encoder(src)
        else:
            memory = src  # If no encoder, pass input directly to decoder
        
        # Decoder
        tgt = self.input_fc(tgt)
        tgt = self.positional_encoding(tgt)
        tgt = self.dropout(tgt)
        tgt = tgt.transpose(0, 1)  # Shape becomes [sequence_length, batch_size, d_model]
        
        if self.decoder is not None:
            output = self.decoder(tgt, memory)
        else:
            output = memory  # If no decoder, use memory directly
        
        output = output.transpose(0, 1)  # Shape back to [batch_size, sequence_length, d_model]
        output = self.output_fc(output)
        output = self.dequant(output)  # Dequantize output
        return output
