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
    # Instantiate a new model with parameters from config
    return TransformerTrainer(config=config)
