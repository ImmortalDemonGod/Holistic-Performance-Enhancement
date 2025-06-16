import pytest
from omegaconf import OmegaConf
from cultivation.systems.arc_reactor.jarc_reactor.utils.model_factory import create_transformer_trainer
from cultivation.systems.arc_reactor.jarc_reactor.utils.train import TransformerTrainer

def test_create_transformer_trainer_importable():
    """
    Tests that create_transformer_trainer function can be imported.
    """
    assert create_transformer_trainer is not None

def test_create_transformer_trainer_returns_trainer():
    """
    Tests that create_transformer_trainer returns a TransformerTrainer instance
    when given a minimal valid config.
    """
    try:
        # Create a minimal mock config for TransformerTrainer
        # This needs to align with what TransformerTrainer expects in its __init__
        # For now, let's assume it primarily needs model_config and training_config parts.
        # We'll need to ensure these nested structures are present.
        
        mock_config_dict = {
            "model": {
                "name": "bert-base-uncased", # From original mock
                "dropout_rate": 0.1,        # From original mock
                # Add other fields TransformerTrainer's model setup might need from model_config
            },
            "training": {
                "num_labels": 2, # Example, was previously a direct arg to ModelFactory
                "learning_rate": 5e-5,
                "batch_size": 8,
                "max_epochs": 1,
                # Add other fields TransformerTrainer might need from training_config
            },
            "logging": { # Add logging config as TransformerTrainer likely uses it
                "log_level": "INFO",
                "log_dir": "test_logs/",
                "log_file": "test_trainer.log"
            }
            # Add other top-level keys if TransformerTrainer's __init__ directly accesses them
            # from the main config object (e.g., config.some_other_param)
        }
        mock_config = OmegaConf.create(mock_config_dict)

        trainer_instance = create_transformer_trainer(config=mock_config)
        assert isinstance(trainer_instance, TransformerTrainer)
    except Exception as e:
        pytest.fail(f"create_transformer_trainer failed: {e}")
