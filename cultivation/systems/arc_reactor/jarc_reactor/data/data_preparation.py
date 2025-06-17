# jarc_reactor/data/data_preparation.py
import os
import orjson
import json
import logging
from pathlib import Path
import hydra # For hydra.utils.to_absolute_path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import DictConfig # Added for Hydra
from cultivation.utils.logging_config import setup_logging
# from cultivation.systems.arc_reactor.jarc_reactor.config import config # Removed old config
from cultivation.systems.arc_reactor.jarc_reactor.data.context_data import ContextPair
from cultivation.systems.arc_reactor.jarc_reactor.utils.padding_utils import pad_to_fixed_size
from .data_loading_utils import inspect_data_structure # Added import

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

def load_context_pair(filepath, task_id, context_map):
    try:
        with open(filepath, 'rb') as f:
            data = orjson.loads(f.read())
        examples = data.get('train', []) if 'train' in data else data
        if not examples:
            return
        context_example = examples[0]
        input_key = 'input' if 'input' in context_example else 'input_data'
        output_key = 'output' if 'output' in context_example else 'output_data'
        context_input = pad_to_fixed_size(
            torch.tensor(context_example[input_key], dtype=torch.long),
            target_shape=(30, 30)
        )
        context_output = pad_to_fixed_size(
            torch.tensor(context_example[output_key], dtype=torch.long),
            target_shape=(30, 30)
        )
        context_map[task_id] = ContextPair(
            context_input=context_input,
            context_output=context_output
        )
    except Exception as e:
        logger.error(f"Error loading context for task '{task_id}' from '{filepath}': {str(e)}")

def load_context_pairs(directory, context_map):
    logger.info(f"Loading context pairs from '{directory}'...")
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(load_context_pair, os.path.join(directory, filename), os.path.splitext(filename)[0], context_map): filename
            for filename in json_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading context pairs"):
            future.result()  # Ensure any exceptions are raised

def load_main_data_concurrently(directory, context_map, train_inputs, train_outputs, train_task_ids, train_context_pairs,
                                test_inputs, test_outputs, test_task_ids, test_context_pairs, is_synthetic=False):
    """Load main dataset from the specified directory concurrently"""
    logger.info(f"Loading main dataset from '{directory}'{' (synthetic)' if is_synthetic else ''}...")

    def load_single_file(filepath, task_id):
        try:
            with open(filepath, 'rb') as f:
                data = orjson.loads(f.read())

            # Determine data structure based on directory
            if not is_synthetic:  # Use is_synthetic flag instead of directory string comparison
                train_data = data.get('train', [])  # Use all examples for training; context is handled separately
                test_data = data.get('test', [])
            else:  # 'sythtraining' assumed to have a flat list
                train_data = data[1:]  # Skip the first example if needed
                test_data = []  # Synthetic data may not have a separate test set

            # Process training data
            train_results = []
            for item in train_data:
                input_tensor = pad_to_fixed_size(
                    torch.tensor(item['input'], dtype=torch.long),
                    target_shape=(30, 30)
                )
                output_tensor = pad_to_fixed_size(
                    torch.tensor(item['output'], dtype=torch.long),
                    target_shape=(30, 30)
                )
                train_results.append((input_tensor, output_tensor, task_id, context_map[task_id]))

            # Process test data
            test_results = []
            for item in test_data:
                input_tensor = pad_to_fixed_size(
                    torch.tensor(item['input'], dtype=torch.long),
                    target_shape=(30, 30)
                )
                output_tensor = pad_to_fixed_size(
                    torch.tensor(item['output'], dtype=torch.long),
                    target_shape=(30, 30)
                )
                test_results.append((input_tensor, output_tensor, task_id, context_map[task_id]))

            return train_results, test_results
        except Exception as e:
            logger.error(f"Error processing file '{filepath}': {str(e)}")
            return [], []

    # List all JSON files in the directory
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

    # Use ThreadPoolExecutor to load files concurrently
    with ThreadPoolExecutor() as executor:
        # Submit tasks and collect futures
        futures = {
            executor.submit(load_single_file, os.path.join(directory, filename), os.path.splitext(filename)[0]): filename
            for filename in json_files
        }

        # Use tqdm to display progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading data"):
            train_results, test_results = future.result()
            for input_tensor, output_tensor, task_id, context_pair in train_results:
                train_inputs.append(input_tensor)
                train_outputs.append(output_tensor)
                train_task_ids.append(task_id)
                train_context_pairs.append(context_pair)
            for input_tensor, output_tensor, task_id, context_pair in test_results:
                test_inputs.append(input_tensor)
                test_outputs.append(output_tensor)
                test_task_ids.append(task_id)
                test_context_pairs.append(context_pair)

def _validate_and_inspect_path(cfg: DictConfig, directory: str) -> Path:
    """Validates the data directory path and inspects a few files."""
    logger.info(f"Preparing data from directory: {directory}")
    # Resolve path relative to Hydra's original working directory
    absolute_directory_path = hydra.utils.to_absolute_path(directory)
    data_path = Path(absolute_directory_path)
    if not data_path.exists():
        logger.error(f"Absolute data directory does not exist: {absolute_directory_path} (original input: {directory})")
        raise FileNotFoundError(f"Absolute data directory does not exist: {absolute_directory_path} (original input: {directory})")
    if not data_path.is_dir():
        logger.error(f"Provided absolute path is not a directory: {absolute_directory_path} (original input: {directory})")
        raise NotADirectoryError(f"Provided absolute path is not a directory: {absolute_directory_path} (original input: {directory})")

    # Inspect a limited number of files for basic structure validation
    logger.info(f"Inspecting data structure for a sample of files in {directory}...")
    successful_files = 0
    total_files = 0
    log_limit = 2
    for filename in os.listdir(data_path):
        if filename.endswith('.json'):
            total_files += 1
            if successful_files < log_limit:
                # The original `directory` can be a relative path.
                # Using `data_path` ensures the absolute path is used, which is robust
                # against Hydra's CWD changes.
                if inspect_data_structure(cfg, filename, str(data_path)):
                    successful_files += 1
            else:
                break
    logger.info(f"Successfully inspected {successful_files}/{total_files} files for structure.")
    return data_path

def _load_raw_data(data_path: Path) -> dict:
    """Loads context pairs and main data concurrently."""
    train_inputs, train_outputs, train_task_ids = [], [], []
    test_inputs, test_outputs, test_task_ids = [], [], []
    context_map = {}
    train_context_pairs, test_context_pairs = [], []

    load_context_pairs(str(data_path), context_map)
    
    logger.info("Loading main dataset...")
    load_main_data_concurrently(
        directory=str(data_path),
        context_map=context_map,
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        train_task_ids=train_task_ids,
        train_context_pairs=train_context_pairs,
        test_inputs=test_inputs,
        test_outputs=test_outputs,
        test_task_ids=test_task_ids,
        test_context_pairs=test_context_pairs
    )
    
    return {
        "train_inputs": train_inputs, "train_outputs": train_outputs, "train_task_ids": train_task_ids,
        "test_inputs": test_inputs, "test_outputs": test_outputs, "test_task_ids": test_task_ids,
        "train_context_pairs": train_context_pairs, "test_context_pairs": test_context_pairs
    }

def _process_and_create_tensors(raw_data: dict) -> dict:
    """Converts data lists to tensors and creates task ID mappings."""
    # Create a sorted list of unique task_ids and the mapping
    unique_task_ids = sorted(set(raw_data["train_task_ids"] + raw_data["test_task_ids"]))
    task_id_map = {task_id: idx for idx, task_id in enumerate(unique_task_ids)}
    logger.info(f"Total unique task_ids: {len(unique_task_ids)}")

    # Convert lists to tensors
    train_inputs = torch.stack(raw_data["train_inputs"])
    train_outputs = torch.stack(raw_data["train_outputs"])
    test_inputs = torch.stack(raw_data["test_inputs"])
    test_outputs = torch.stack(raw_data["test_outputs"])

    # Convert task_ids to tensors using the map
    train_task_ids_tensor = torch.tensor([task_id_map[tid] for tid in raw_data["train_task_ids"]], dtype=torch.long)
    test_task_ids_tensor = torch.tensor([task_id_map[tid] for tid in raw_data["test_task_ids"]], dtype=torch.long)

    # Convert context pairs to tensors
    train_ctx_inputs = torch.stack([pair.context_input for pair in raw_data["train_context_pairs"]])
    train_ctx_outputs = torch.stack([pair.context_output for pair in raw_data["train_context_pairs"]])
    test_ctx_inputs = torch.stack([pair.context_input for pair in raw_data["test_context_pairs"]])
    test_ctx_outputs = torch.stack([pair.context_output for pair in raw_data["test_context_pairs"]])

    # Assertions to ensure data consistency
    assert train_inputs.size(0) == train_outputs.size(0) == train_task_ids_tensor.size(0) == train_ctx_inputs.size(0), "Mismatch in training data tensor sizes."
    assert test_inputs.size(0) == test_outputs.size(0) == test_task_ids_tensor.size(0) == test_ctx_inputs.size(0), "Mismatch in testing data tensor sizes."

    return {
        "train_inputs": train_inputs, "train_outputs": train_outputs, "train_task_ids": train_task_ids_tensor,
        "train_ctx_inputs": train_ctx_inputs, "train_ctx_outputs": train_ctx_outputs,
        "test_inputs": test_inputs, "test_outputs": test_outputs, "test_task_ids": test_task_ids_tensor,
        "test_ctx_inputs": test_ctx_inputs, "test_ctx_outputs": test_ctx_outputs,
        "task_id_map": task_id_map
    }

def prepare_data(cfg: DictConfig, return_datasets: bool = False):
    logger.info(f"prepare_data called. cfg.training.training_data_dir = {cfg.training.training_data_dir}")
    logger.info(f"prepare_data: cfg.training.synthetic_data_dir = {cfg.training.synthetic_data_dir}")
    logger.info(f"prepare_data: cfg.training.include_synthetic_training_data = {cfg.training.include_synthetic_training_data}")
    """
    Prepares training and validation data from a specified directory using Hydra config.
    This function orchestrates the loading, processing, and batching of data.
    """
    # 1. Initialize paths and parameters from Hydra config
    directory = cfg.training.training_data_dir
    batch_size = cfg.training.batch_size
    
    # 2. Validate path and inspect a sample of data files
    data_path = _validate_and_inspect_path(cfg, directory)

    # 3. Load raw data from files
    raw_data = _load_raw_data(data_path)

    # 4. Process raw data into tensors and create mappings
    processed_data = _process_and_create_tensors(raw_data)

    # 5. Create TensorDatasets
    train_dataset = TensorDataset(
        processed_data["train_inputs"], processed_data["train_outputs"],
        processed_data["train_ctx_inputs"], processed_data["train_ctx_outputs"],
        processed_data["train_task_ids"]
    )
    test_dataset = TensorDataset(
        processed_data["test_inputs"], processed_data["test_outputs"],
        processed_data["test_ctx_inputs"], processed_data["test_ctx_outputs"],
        processed_data["test_task_ids"]
    )

    # 6. Save the task_id_map for reference
    task_id_map = processed_data["task_id_map"]
    logger.info("Saving task_id_map.json with the current task mappings.")
    try:
        with open('task_id_map.json', 'w') as f:
            json.dump(task_id_map, f)
    except Exception as e:
        logger.error(f"Failed to save task_id_map.json: {str(e)}")

    # 7. Return datasets or dataloaders as requested
    if return_datasets:
        logger.info("Returning TensorDatasets instead of DataLoaders.")
        return train_dataset, test_dataset
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    logger.info("Data preparation completed successfully.")
    return train_loader, val_loader