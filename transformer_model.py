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
        
        #self.use_lora = use_lora
        #self.lora_rank = lora_rank  # Initialize lora_rank
        self.checkpoint_path = checkpoint_path
        #if self.use_lora:
        #    # LoRA parameters
        #    self.lora_A = Parameter(torch.randn(d_model, self.lora_rank))
        #    self.lora_B = Parameter(torch.randn(self.lora_rank, d_model))
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
        # Quantization
        src = self.quant(src).float()  # [batch, seq_len, input_dim]
        tgt = self.quant(tgt).float()
        
        # Process context if available
        if ctx_input is not None and ctx_output is not None:
            ctx_input = self.quant(ctx_input)
            ctx_output = self.quant(ctx_output)
            context_embedding = self.context_encoder(ctx_input, ctx_output)
        else:
            context_embedding = None
        
        # Debug shapes
        print(f"src shape: {src.shape}")  # Should be [batch, seq_len, input_dim]
        
        # Project input along both dimensions
        x_dim = self.input_fc_dim(src)  # [batch, seq_len, d_model]
        x_seq = src.transpose(-1, -2)    # [batch, input_dim, seq_len]
        x_seq = self.input_fc_seq(x_seq) # [batch, input_dim, d_model]
        x_seq = x_seq.transpose(-1, -2)  # [batch, d_model, input_dim]
        
        # Add positional encoding
        x = self.positional_encoding(x_dim)  # Use only x_dim for positional encoding
        x = self.dropout(x)
        
        # Integrate context if available
        if context_embedding is not None:
            context_expanded = context_embedding.unsqueeze(1).expand(-1, x.size(1), -1)
            x = self.context_integration(
                torch.cat([x, context_expanded], dim=-1)
            )
            x = self.context_dropout(x)
        
        # Encoder
        if self.encoder is not None:
            memory = self.encoder(x)
            memory = self.encoder_dropout(memory)
            #if self.use_lora:
            #    lora_output = torch.matmul(torch.matmul(memory, self.lora_A), self.lora_B)
            #    memory = memory + self.dropout(lora_output)
        else:
            memory = x
        
        # Decoder
        if self.decoder is not None:
            tgt = self.input_fc_dim(tgt)
            tgt = self.positional_encoding(tgt)
            tgt = self.dropout(tgt)
            output = self.decoder(tgt, memory)
            output = self.decoder_dropout(output)
        else:
            output = memory
        
        # Final output processing
        output = self.output_fc(output)
        output = output * 5.0  # Scale logits
        
        return self.dequant(output)
    
    @staticmethod
    def create_src_mask(src):
        """Create padding mask for source sequence"""
        src_mask = (src != 0).float()
        return src_mask.unsqueeze(-2)

    @staticmethod
    def create_tgt_mask(tgt):
        """Create causal mask for target sequence"""
        sz = tgt.size(1)
        mask = torch.triu(torch.ones(sz, sz), diagonal=1) == 0
        return mask