# cultivation/systems/arc_reactor/jarc_reactor/data/data_loading_utils.py
import os
import orjson
import logging
import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def inspect_data_structure(cfg: DictConfig, filename: str, directory: str | None = None):
    """Debug helper to examine JSON structure."""
    # Determine the effective directory based on cfg structure if directory is None
    # This part needs to be context-aware or the caller must always provide a directory.
    # For now, assuming 'directory' will be provided or cfg has a known path.
    # Example: if directory is None: directory = cfg.data.raw_dir (adjust as per actual cfg)
    # Safest is to require 'directory' if it's not universally determinable from cfg.

    # Simplified: Caller must ensure 'directory' is correctly set or passed.
    # If called from eval_data_prep, cfg.evaluation.data_dir is used by original.
    # If called from data_preparation, cfg.data.train_dir (or similar) might be used.
    # For a util, it's better if the caller resolves the directory from cfg.
    # Let's assume 'directory' is always passed correctly for now.
    if directory is None:
        # This is a placeholder. The caller should ideally resolve this.
        # For now, trying to mimic original behavior if possible, but it's risky.
        if hasattr(cfg, 'evaluation') and hasattr(cfg.evaluation, 'data_dir'):
            directory = cfg.evaluation.data_dir
        elif hasattr(cfg, 'data') and hasattr(cfg.data, 'train_dir'): # Example, adjust if needed
            directory = cfg.data.train_dir
        else:
            logger.error("inspect_data_structure: 'directory' is None and could not be determined from cfg.")
            return False

    filepath = os.path.join(directory, filename)
    try:
        with open(filepath, 'rb') as f:
            data = orjson.loads(f.read())
        logger.debug(f"File structure for {filepath}:")
        logger.debug(f"Keys in data: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        if isinstance(data, dict):
            logger.debug(f"Number of train examples: {len(data.get('train', []))}")
            logger.debug(f"Number of test examples: {len(data.get('test', []))}")
            if data.get('train'):
                sample_train = data['train'][0]
                logger.debug(f"Sample train input shape: {np.array(sample_train.get('input', [])).shape}")
                logger.debug(f"Sample train context_input exists: {'context_input' in sample_train}")
                logger.debug(f"Sample train context_output exists: {'context_output' in sample_train}")
            if data.get('test'):
                sample_test = data['test'][0]
                logger.debug(f"Sample test context_input exists: {'context_input' in sample_test}")
                logger.debug(f"Sample test context_output exists: {'context_output' in sample_test}")
        return True
    except Exception as e:
        logger.error(f"Error inspecting {filepath}: {str(e)}")
        return False
