# model/context_encoder.py                                                                                                                                   
import torch                                                                                                                                                 
import torch.nn as nn                                                                                                                                        
import logging                                                                                                                                               
                                                                                                                                                            
logger = logging.getLogger(__name__)                                                                                                                         
                                                                                                                                                            
class ContextEncoderModule(nn.Module):                                                                                                                       
    """Context encoder following PQA architecture"""                                                                                                         
    def __init__(self, d_model, heads, dropout=0.1):                                                                                                         
        super().__init__()                                                                                                                                   
        self.input_projection = nn.Linear(1, d_model)                                                                                                        
        self.self_attention = nn.MultiheadAttention(                                                                                                         
            d_model, heads, batch_first=True, dropout=dropout                                                                                                
        )                                                                                                                                                    
        self.ffn = nn.Sequential(                                                                                                                            
            nn.Linear(d_model, d_model * 4),                                                                                                                 
            nn.ReLU(),                                                                                                                                       
            nn.Linear(d_model * 4, d_model)                                                                                                                  
        )                                                                                                                                                    
        self.norm1 = nn.LayerNorm(d_model)                                                                                                                   
        self.norm2 = nn.LayerNorm(d_model)                                                                                                                   
        self.dropout = nn.Dropout(dropout)                                                                                                                   
                                                                                                                                                            
        logger.info(f"Initialized ContextEncoderModule with d_model={d_model}, heads={heads}")                                                               
                                                                                                                                                            
    def forward(self, ctx_input, ctx_output):                                                                                                                
        """                                                                                                                                                  
        Process context pair to create context embedding                                                                                                     
        Args:                                                                                                                                                
            ctx_input: [batch_size, height, width]                                                                                                           
            ctx_output: [batch_size, height, width]                                                                                                          
        Returns:                                                                                                                                             
            context_embedding: [batch_size, d_model]                                                                                                         
        """                                                                                                                                                  
        # Log shapes for debugging                                                                                                                           
        logger.debug(f"Context input shape: {ctx_input.shape}")                                                                                              
        logger.debug(f"Context output shape: {ctx_output.shape}")                                                                                            
                                                                                                                                                            
        # Concatenate along sequence dimension                                                                                                               
        x = torch.cat([ctx_input, ctx_output], dim=1)                                                                                                        
                                                                                                                                                            
        # Add channel dimension and project                                                                                                                  
        x = self.input_projection(x.unsqueeze(-1))                                                                                                           
                                                                                                                                                            
        # Self-attention with residual and normalization                                                                                                     
        attn_out, _ = self.self_attention(x, x, x)                                                                                                           
        attn_out = self.dropout(attn_out)                                                                                                                    
        x = self.norm1(x + attn_out)                                                                                                                         
                                                                                                                                                            
        # Feed-forward with residual and normalization                                                                                                       
        ff_out = self.ffn(x)                                                                                                                                 
        ff_out = self.dropout(ff_out)                                                                                                                        
        x = self.norm2(x + ff_out)                                                                                                                           
                                                                                                                                                            
        # Pool context embedding                                                                                                                             
        context_embedding = x.mean(dim=1)                                                                                                                    
                                                                                                                                                            
        logger.debug(f"Context embedding shape: {context_embedding.shape}")                                                                                  
        return context_embedding 