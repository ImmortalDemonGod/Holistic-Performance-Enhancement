import torch
from train import TransformerTrainer

def test_transformer_trainer_dimensions():
    # Sample hyperparameters
    input_dim = 30
    d_model = 128
    encoder_layers = 2
    decoder_layers = 2
    heads = 8
    d_ff = 256
    output_dim = 30
    learning_rate = 0.0001
    include_sythtraining_data = True
    
    # Instantiate the trainer
    trainer = TransformerTrainer(
        input_dim=input_dim,
        d_model=d_model,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        heads=heads,
        d_ff=d_ff,
        output_dim=output_dim,
        learning_rate=learning_rate,
        include_sythtraining_data=include_sythtraining_data
    )
    
    # Generate dummy data
    batch_size = 50
    seq_length = 30
    num_features = input_dim
    
    src = torch.randn(batch_size, seq_length, num_features)
    tgt = torch.randn(batch_size, seq_length, output_dim)
    ctx_input = torch.randn(batch_size, seq_length, trainer.d_model)
    ctx_output = torch.randn(batch_size, seq_length, trainer.d_model)
    
    # Perform forward pass
    try:
        output = trainer(src, tgt, ctx_input, ctx_output)
        print("Forward pass successful. Output shape:", output.shape)
    except Exception as e:
        print("Forward pass failed with error:", str(e))

if __name__ == "__main__":
    test_transformer_trainer_dimensions()
