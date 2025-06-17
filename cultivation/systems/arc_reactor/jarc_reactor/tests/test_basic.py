import pytest
import torch
from omegaconf import OmegaConf
from pathlib import Path
from cultivation.systems.arc_reactor.jarc_reactor.utils.model_factory import create_transformer_trainer
from cultivation.systems.arc_reactor.jarc_reactor.utils.train import TransformerTrainer
from cultivation.systems.arc_reactor.jarc_reactor.data.data_module import MyDataModule

@pytest.fixture(scope="module")
def test_output_dirs():
    """Creates test output directories before any tests in this module run."""
    current_file_path = Path(__file__).resolve()
    test_logs_dir = current_file_path.parent / "test_outputs" / "logs"
    test_checkpoints_dir = current_file_path.parent / "test_outputs" / "checkpoints"
    
    test_logs_dir.mkdir(parents=True, exist_ok=True)
    test_checkpoints_dir.mkdir(parents=True, exist_ok=True)

def test_collection_works():
    """
    A simple, dependency-free test to see if pytest collection runs.
    """
    assert True

def test_create_transformer_trainer_importable():
    """
    Tests that create_transformer_trainer function can be imported.
    """
    assert create_transformer_trainer is not None

def _get_base_test_config_dict() -> dict:
    """Helper function to create a base mock config for tests."""
    # Determine the absolute path to the dummy data directory
    current_file_path = Path(__file__).resolve()
    dummy_data_dir = current_file_path.parent / "data" / "dummy_training_data"
    test_logs_dir = current_file_path.parent / "test_outputs" / "logs"
    test_checkpoints_dir = current_file_path.parent / "test_outputs" / "checkpoints"

    # Directory creation is now handled by the 'test_output_dirs' fixture

    return {
        "model": {
            "name": "bert-base-uncased",
            "dropout_rate": 0.1,
            "input_dim": 12,  # Vocab size: 0-9 colors, 10 padding, 11 EOS
            "seq_len": 3,     # Grid side length (e.g., 3 for 3x3)
            "d_model": 32,    # Small for testing
            "encoder_layers": 2, # Small for testing
            "decoder_layers": 0,
            "heads": 2,       # Small for testing
            "d_ff": 64,       # Small for testing
            "output_dim": 12, # Vocab size
            "context_encoder": {
                "d_model": 32,
                "path": "bert-base-uncased",
                "heads": 2,
                "dropout_rate": 0.1
            },
            "encoder_dropout_rate": 0.1,
            "decoder_dropout_rate": 0.1,
            "checkpoint_path": None,
            "lora": {
                "use_lora": False,
                "rank": 0
            }
        },
        "training": {
            "num_labels": 12, # Corresponds to output_dim/vocab_size
            "learning_rate": 5e-5,
            "batch_size": 1,
            "max_epochs": 1,
            "num_workers": 0,
            "training_data_dir": str(dummy_data_dir),
            "synthetic_data_dir": None,
            "include_synthetic_training_data": False,
            "fast_dev_run": False,
            "checkpoint_dir": str(test_checkpoints_dir) # Added for trainer init
        },
        "logging": {
            "log_level": "DEBUG",
            "log_dir": str(test_logs_dir),
            "file_logging": { # Nested as per LoggingConfigSchema
                "enable": True,
                "log_file_name": "test_trainer.log"
            }
        },
        "data_preparation": { # Added for MyDataModule
            "max_grid_size": 3,
            "padding_value": 10,
            "eos_token_id": 11,
            "max_output_length": 10 # 3*3 grid + 1 EOS token
        }
    }

def test_create_transformer_trainer_returns_trainer(test_output_dirs):
    """
    Tests that create_transformer_trainer returns a TransformerTrainer instance
    when given a minimal valid config.
    """
    try:
        mock_config_dict = _get_base_test_config_dict()
        mock_config = OmegaConf.create(mock_config_dict)
    
        trainer_instance = create_transformer_trainer(config=mock_config)
        assert isinstance(trainer_instance, TransformerTrainer)
    except Exception as e:
        pytest.fail(f"create_transformer_trainer failed: {e}")

def test_data_module_loads_dummy_data(test_output_dirs):
    """
    Tests that MyDataModule can load and process the dummy ARC task data.
    """
    try:
        config_dict = _get_base_test_config_dict()
        cfg = OmegaConf.create(config_dict)

        data_module = MyDataModule(cfg=cfg)
        data_module.setup(stage='fit') # 'fit' to setup training data

        train_dataloader = data_module.train_dataloader()
        assert train_dataloader is not None, "Train dataloader should not be None"

        batch = next(iter(train_dataloader))
        src, tgt, ctx_input, ctx_output, task_ids = batch

        expected_batch_size = cfg.training.batch_size
        expected_seq_len = cfg.data_preparation.max_output_length

        assert src.shape == (expected_batch_size, expected_seq_len), f"src shape mismatch: {src.shape}"
        assert tgt.shape == (expected_batch_size, expected_seq_len), f"tgt shape mismatch: {tgt.shape}"
        
        assert task_ids.shape == (expected_batch_size,), f"task_ids shape mismatch: {task_ids.shape}"

        assert src.dtype == torch.long, "src dtype should be torch.long"
        assert tgt.dtype == torch.long, "tgt dtype should be torch.long"

    except Exception as e:
        pytest.fail(f"test_data_module_loads_dummy_data failed: {e}")

def test_trainer_runs_one_training_step(test_output_dirs):
    """
    Tests that the model's training_step method executes and returns a valid loss.
    """
    try:
        # 1. Get Config and Data
        config_dict = _get_base_test_config_dict()
        cfg = OmegaConf.create(config_dict)
        data_module = MyDataModule(cfg=cfg)
        data_module.setup(stage='fit')
        batch = next(iter(data_module.train_dataloader()))

        # 2. Create Model
        model = create_transformer_trainer(config=cfg)

        # 3. Execute a single training step
        loss = model.training_step(batch, batch_idx=0)

        # 4. Assertions
        assert loss is not None, "training_step should return a loss"
        assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinity"

    except Exception as e:
        pytest.fail(f"test_trainer_runs_one_training_step failed: {e}")

