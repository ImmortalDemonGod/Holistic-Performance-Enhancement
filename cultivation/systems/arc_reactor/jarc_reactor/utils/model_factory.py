from cultivation.systems.arc_reactor.jarc_reactor.utils.train import TransformerTrainer
from typing import Union
from omegaconf import DictConfig

def create_transformer_trainer(
    config: DictConfig, # Changed to DictConfig
    checkpoint_path: Union[str, None] = None
) -> TransformerTrainer:
    """
    Creates a TransformerTrainer instance either by loading from a checkpoint
    or by instantiating with parameters from the config.

    Args:
        config (DictConfig): Hydra configuration object containing all necessary parameters.
        checkpoint_path (str, optional): Path to the checkpoint file. Defaults to None.

    Returns:
        TransformerTrainer: An instance of TransformerTrainer.
    """
    # Instantiate a new model with parameters from config
    return TransformerTrainer(config=config)
