# transformer_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoderLayer, TransformerEncoder, Parameter
from Utils.positional_encoding import Grid2DPositionalEncoding
from Utils.context_encoder import ContextEncoderModule
from config import Config  # Import the Config class

class TransformerModel(nn.Module):
    def __init__(self, input_dim, seq_len, d_model, encoder_layers, decoder_layers, heads, d_ff, output_dim, dropout_rate, context_encoder_d_model, context_encoder_heads, checkpoint_path, use_lora=False, lora_rank=None):
        super(TransformerModel, self).__init__()
        # Create a config instance to access dropout rates
        config = Config()
        
        self.input_fc_dim = nn.Linear(input_dim, d_model)
        self.input_fc_seq = nn.Linear(seq_len, d_model)  # Ensure consistent output dimension
        self.positional_encoding = Grid2DPositionalEncoding(d_model, max_height=seq_len, max_width=input_dim)
        self.dropout = nn.Dropout(p=dropout_rate)  # Initialize dropout layer

        # Conditionally create the encoder
        if encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(d_model, heads, d_ff, batch_first=True)
            self.encoder = TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        else:
            self.encoder = None
            
        # Conditionally create the decoder
        if decoder_layers > 0:
            decoder_layer = TransformerDecoderLayer(d_model, heads, d_ff, batch_first=True)
            self.decoder = TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        else:
            self.decoder = None
        
        self.use_lora = use_lora
        self.lora_rank = lora_rank  # Initialize lora_rank
        self.checkpoint_path = checkpoint_path
        if self.use_lora:
            # LoRA parameters
            self.lora_A = Parameter(torch.randn(d_model, self.lora_rank))
            self.lora_B = Parameter(torch.randn(self.lora_rank, d_model))
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
        # Initialize dropout layers using config
        self.context_dropout = nn.Dropout(p=config.model.context_dropout_rate)
        self.encoder_dropout = nn.Dropout(p=config.model.encoder_dropout_rate)
        self.decoder_dropout = nn.Dropout(p=config.model.decoder_dropout_rate)



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

        # Dequantize before passing to linear layers and ensure float type
        src = self.dequant(src).float()
        tgt = self.dequant(tgt).float()

        # Debugging: Check data types after dequantization
        #print(f"src dtype after dequant: {src.dtype}, tgt dtype after dequant: {tgt.dtype}")

        # Project both dimensions
        x_dim = self.input_fc_dim(src)
        x_seq = self.input_fc_seq(src)

        # Ensure both projections have the same shape
        if x_dim.size(2) != x_seq.size(2):
            raise ValueError(f"Dimension mismatch: x_dim has {x_dim.size(2)} and x_seq has {x_seq.size(2)}")

        # Combine the projections
        x = x_dim + x_seq
        x = self.positional_encoding(x)

        # Apply LoRA to the attention mechanism
        if self.encoder is not None:
            memory = self.encoder(x)
            if self.use_lora:
                # Apply LoRA
                memory = memory + torch.matmul(torch.matmul(memory, self.lora_A), self.lora_B)
        else:
            memory = x

        # Ensure both projections have the same shape
        if x_dim.size(2) != x_seq.size(2):
            raise ValueError(f"Dimension mismatch: x_dim has {x_dim.size(2)} and x_seq has {x_seq.size(2)}")

        # Combine the projections
        x = x_dim + x_seq
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
            x = self.context_dropout(x)  # Apply context dropout here
        x = x.transpose(0, 1)

        if self.encoder is not None:
            memory = self.encoder(x)
        else:
            memory = x

        if self.decoder is not None:
            tgt = self.input_fc_dim(tgt)
            tgt = self.positional_encoding(tgt)
            tgt = tgt.transpose(0, 1)
            output = self.decoder(tgt, memory)
        else:
            output = memory

        output = output.transpose(0, 1)
        output = self.output_fc(output)
        return self.dequant(output)
