# transformer_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoderLayer, TransformerEncoder, Parameter
from jarc_reactor.utils.positional_encoding import Grid2DPositionalEncoding
from jarc_reactor.models.context_encoder import ContextEncoderModule
from jarc_reactor.config import Config  # Import the Config class

class TransformerModel(nn.Module):
    def __init__(self, input_dim, seq_len, d_model, encoder_layers, decoder_layers, heads, d_ff, output_dim, dropout_rate, context_encoder_d_model, context_encoder_heads, checkpoint_path, use_lora=False, lora_rank=None):
        super(TransformerModel, self).__init__()
        # Create a config instance to access dropout rates
        config = Config()
        
        # Store all dimension-related parameters
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.grid_size = seq_len * seq_len  # Total elements in grid (e.g., 900 for 30x30)
        
        # DEBUG: Print initialization dimensions
        print(f"\nInitializing TransformerModel with dimensions:")
        print(f"input_dim: {input_dim}, seq_len: {seq_len}, d_model: {d_model}")
        print(f"grid_size (seq_len * seq_len): {self.grid_size}")
        
        # Modified input projections for grid structure
        self.input_fc = nn.Linear(1, d_model)  # Project each grid cell to d_model dimensions
        print(f"Input projection will convert each element from 1 → {d_model} dimensions")
        
        # Positional encoding and dropout
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
        
        # Modified output layers
        self.output_projection = nn.Linear(d_model, d_model)
        self.output_fc = nn.Linear(d_model, 11)  # Still 11 classes for padding handling
        
        # Initialize dropout layers using config
        self.context_dropout = nn.Dropout(p=config.model.context_dropout_rate)
        self.encoder_dropout = nn.Dropout(p=config.model.encoder_dropout_rate)
        self.decoder_dropout = nn.Dropout(p=config.model.decoder_dropout_rate)

        # Add quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def to(self, *args, **kwargs):
        """Override to() to properly handle device movement of quantization stubs and all components"""
        # First move the entire model using parent's to()
        model = super().to(*args, **kwargs)
        
        # Explicitly move quantization stubs to the target device
        device = torch.device(args[0]) if args else \
                 kwargs.get('device', next(model.parameters()).device)
        
        model.quant = model.quant.to(device)
        model.dequant = model.dequant.to(device)
        
        # Ensure all components are on the correct device
        model.input_fc = model.input_fc.to(device)
        model.positional_encoding = model.positional_encoding.to(device)
        if model.encoder is not None:
            model.encoder = model.encoder.to(device)
        if model.decoder is not None:
            model.decoder = model.decoder.to(device)
        model.context_encoder = model.context_encoder.to(device)
        model.context_integration = model.context_integration.to(device)
        model.output_projection = model.output_projection.to(device)
        model.output_fc = model.output_fc.to(device)
        
        return model

    def debug_shape(self, tensor, name):
        """Helper function to debug tensor shapes and content"""
        print(f"\nDEBUG - {name}:")
        print(f"Shape: {tensor.shape}")
        print(f"Total elements: {tensor.numel()}")
        if len(tensor.shape) >= 2:
            print(f"Elements per batch: {tensor.shape[1] * (tensor.shape[2] if len(tensor.shape) > 2 else 1)}")
        print(f"Tensor type: {tensor.dtype}")
        print(f"Min value: {tensor.min().item():.6f}")
        print(f"Max value: {tensor.max().item():.6f}")
        print(f"Mean value: {tensor.mean().item():.6f}")
        print("-" * 50)
        return tensor

    def forward(self, src, tgt, ctx_input=None, ctx_output=None):
        """
        Forward pass with detailed shape tracking
        Args:
            src: Input tensor [batch, seq_len, input_dim]
            tgt: Target tensor [batch, seq_len, input_dim]
            ctx_input: Optional context input
            ctx_output: Optional context output
        Returns:
            output: Output tensor [batch, seq_len, seq_len, 11]
        """
        # Get model device and ensure quantization stubs are on it
        model_device = next(self.parameters()).device
        self.quant = self.quant.to(model_device)
        self.dequant = self.dequant.to(model_device)

        # Ensure inputs are on correct device
        src = src.to(model_device)
        if tgt is not None:
            tgt = tgt.to(model_device)
        
        # 1. Initial Setup and Input Processing
        src = self.quant(src).float()
        #self.debug_shape(src, "1. Initial input")
        
        # Get batch size from input
        batch_size = src.size(0)
        #print(f"\nProcessing batch of size: {batch_size}")
        
        # 2. Reshape input from [batch, seq_len, input_dim] to [batch, grid_size, 1]
        src_flat = src.view(batch_size, -1, 1)  # Flatten the grid to a sequence
        #self.debug_shape(src_flat, "2. After flattening to sequence")
        
        # 3. Project each grid cell to d_model dimensions
        x = self.input_fc(src_flat)  # Shape: [batch, grid_size, d_model]
        #self.debug_shape(x, "3. After initial projection")
        
        # 4. Add positional encoding
        x = self.positional_encoding(x)
        # self.debug_shape(x, "4. After positional encoding")
        x = self.dropout(x)
        #self.debug_shape(x, "4b. After dropout")
        
        # 5. Process context if available
        if ctx_input is not None and ctx_output is not None:
            # Move context inputs to correct device
            ctx_input = ctx_input.to(model_device)
            ctx_output = ctx_output.to(model_device)
            
            ctx_input = self.quant(ctx_input)
            ctx_output = self.quant(ctx_output)
            #print("\nProcessing context:")
            #self.debug_shape(ctx_input, "5. Context input")
            #self.debug_shape(ctx_output, "5. Context output")
            
            context_embedding = self.context_encoder(ctx_input, ctx_output)
            #self.debug_shape(context_embedding, "5a. Context embedding")
            
            context_expanded = context_embedding.unsqueeze(1).expand(-1, x.size(1), -1)
            #self.debug_shape(context_expanded, "5b. Expanded context")
            
            x = self.context_integration(torch.cat([x, context_expanded], dim=-1))
            #self.debug_shape(x, "5c. After context integration")
            x = self.context_dropout(x)
            #self.debug_shape(x, "5d. After context dropout")
        
        # 6. Encoder
        if self.encoder is not None:
            memory = self.encoder(x)
            #self.debug_shape(memory, "6a. After encoder")
            memory = self.encoder_dropout(memory)
            #self.debug_shape(memory, "6b. After encoder dropout")
        else:
            memory = x
        
        # 7. Decoder
        if self.decoder is not None:
            # Process target similarly to source
            tgt = self.quant(tgt).float()
            #self.debug_shape(tgt, "7a. Initial target")
            
            tgt_flat = tgt.view(batch_size, -1, 1)
            #self.debug_shape(tgt_flat, "7b. Flattened target")
            
            tgt_proj = self.input_fc(tgt_flat)
            #self.debug_shape(tgt_proj, "7c. Projected target")
            
            tgt_proj = self.positional_encoding(tgt_proj)
            #self.debug_shape(tgt_proj, "7d. Target with positional encoding")
            
            tgt_proj = self.dropout(tgt_proj)
            
            output = self.decoder(tgt_proj, memory)
            #self.debug_shape(output, "7e. After decoder")
            output = self.decoder_dropout(output)
            #self.debug_shape(output, "7f. After decoder dropout")
        else:
            output = memory
        
        # 8. Final projections
        output = self.output_projection(output)
        #self.debug_shape(output, "8a. After output projection")
        
        output = self.output_fc(output)
        #self.debug_shape(output, "8b. After final linear layer")
        
        # 9. Reshape back to grid structure
        try:
            output = output.view(batch_size, self.seq_len, self.seq_len, -1)
            #self.debug_shape(output, "9. Final output (grid structure)")
        except RuntimeError as e:
            print("\nERROR during final reshape:")
            print(f"Attempted to reshape tensor of size {output.shape}")
            print(f"Total elements: {output.numel()}")
            print(f"Target shape: [{batch_size}, {self.seq_len}, {self.seq_len}, -1]")
            print(f"This would require total elements to be divisible by {batch_size * self.seq_len * self.seq_len}")
            print(f"Current elements per grid cell: {output.numel() / (batch_size * self.seq_len * self.seq_len)}")
            raise e
        
        # 10. Scale and return
        output = output * 5.0
        output = self.dequant(output)
        return output.to(model_device)  # Ensure final output is on correct device

            
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
