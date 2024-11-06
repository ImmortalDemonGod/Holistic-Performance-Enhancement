from train import TransformerTrainer
from typing import Union
from config import Config

def create_transformer_trainer(
    config: Config,
    checkpoint_path: Union[str, None] = None
) -> TransformerTrainer:
    """
    Creates a TransformerTrainer instance either by loading from a checkpoint
    or by instantiating with parameters from the config.

    Args:
        config (Config): Configuration object containing all necessary parameters.
        checkpoint_path (str, optional): Path to the checkpoint file. Defaults to None.

    Returns:
        TransformerTrainer: An instance of TransformerTrainer.
    """
    if checkpoint_path:
        # Load from checkpoint without modifying parameters
        return TransformerTrainer.load_from_checkpoint(
            checkpoint_path,
            **config.model.__dict__,
            **config.training.__dict__
        )
    else:
        # Instantiate a new model with parameters from config
        return TransformerTrainer(
            input_dim=config.model.input_dim,
            d_model=config.model.d_model,
            encoder_layers=config.model.encoder_layers,
            decoder_layers=config.model.decoder_layers,
            heads=config.model.heads,
            d_ff=config.model.d_ff,
            output_dim=config.model.output_dim,
            learning_rate=config.training.learning_rate,
            dropout=config.model.dropout,
            context_encoder_d_model=config.model.context_encoder_d_model,
            context_encoder_heads=config.model.context_encoder_heads,
            include_synthetic_training_data=config.training.include_synthetic_training_data
        )
