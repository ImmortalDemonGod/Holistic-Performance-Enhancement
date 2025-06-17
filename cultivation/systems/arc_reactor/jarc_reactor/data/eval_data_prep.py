# jarc_reactor/data/eval_data_prep.py
import os
import orjson
import json
from tqdm import tqdm
from typing import List, Dict, Tuple # Re-added Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import DictConfig

# from cultivation.systems.arc_reactor.jarc_reactor.config import config, include_synthetic_training_data, synthetic_dir # Removed old config
from cultivation.systems.arc_reactor.jarc_reactor.data.context_data import ContextPair # Keep this one
from cultivation.systems.arc_reactor.jarc_reactor.utils.padding_utils import pad_to_fixed_size
from .data_loading_utils import inspect_data_structure

logger = logging.getLogger(__name__)

# Module-level Type Aliases for clarity and static analysis
SingleFileResultType = Tuple[torch.Tensor, torch.Tensor, str, ContextPair]
LoadSingleFileReturnType = Tuple[List[SingleFileResultType], List[SingleFileResultType]]

def load_context_pair(filepath: str, task_id: str, context_map: Dict[str, ContextPair]):
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
            torch.tensor(context_example[input_key], dtype=torch.float32),
            target_shape=(30, 30)
        )
        context_output = pad_to_fixed_size(
            torch.tensor(context_example[output_key], dtype=torch.float32),
            target_shape=(30, 30)
        )
        context_map[task_id] = ContextPair(
            context_input=context_input,
            context_output=context_output
        )
    except Exception as e:
        logger.error(f"Error loading context for task '{task_id}' from '{filepath}': {str(e)}")

def load_context_pairs(directory: str, context_map: Dict[str, ContextPair]):
    logger.info(f"Loading context pairs from '{directory}'...")
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(load_context_pair, os.path.join(directory, filename), os.path.splitext(filename)[0], context_map): filename
            for filename in json_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading context pairs"):
            future.result()  # Ensure any exceptions are raised

def load_main_data_concurrently(
    directory: str,
    context_map: Dict[str, ContextPair],
    train_inputs: List[torch.Tensor],
    train_outputs: List[torch.Tensor],
    train_task_ids: List[str],
    train_context_pairs: List[ContextPair],
    test_inputs: List[torch.Tensor],
    test_outputs: List[torch.Tensor],
    test_task_ids: List[str],
    test_context_pairs: List[ContextPair],
    is_synthetic: bool = False
):
    """Load main dataset from the specified directory concurrently"""
    logger.info(f"Loading main dataset from '{directory}'{' (synthetic)' if is_synthetic else ''}...")

    def load_single_file(filepath: str, task_id: str) -> LoadSingleFileReturnType:
        try:
            with open(filepath, 'rb') as f:
                data = orjson.loads(f.read())

            if not is_synthetic:
                train_data = data.get('train', [])[1:]
                test_data = data.get('test', [])
            else:
                train_data = data
                test_data = []

            train_results: List[SingleFileResultType] = []
            for item in train_data:
                input_tensor = pad_to_fixed_size(
                    torch.tensor(item['input'], dtype=torch.float32),
                    target_shape=(30, 30)
                )
                output_tensor = pad_to_fixed_size(
                    torch.tensor(item['output'], dtype=torch.float32),
                    target_shape=(30, 30)
                )
                train_results.append((input_tensor, output_tensor, task_id, context_map[task_id]))

            test_results: List[SingleFileResultType] = []
            for item in test_data:
                input_tensor = pad_to_fixed_size(
                    torch.tensor(item['input'], dtype=torch.int8),
                    target_shape=(30, 30)
                )
                output_tensor = pad_to_fixed_size(
                    torch.tensor(item['output'], dtype=torch.int8),
                    target_shape=(30, 30)
                )
                test_results.append((input_tensor, output_tensor, task_id, context_map[task_id]))

            return train_results, test_results
        except KeyError as e:
            logger.error(f"KeyError: Task ID '{task_id}' not found in context_map while processing '{filepath}'. Original error: {str(e)}")
            return [], [] # Return empty lists to prevent further issues
        except Exception as e:
            logger.error(f"Error processing file '{filepath}': {str(e)}")
            return [], []

    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(load_single_file, os.path.join(directory, filename), os.path.splitext(filename)[0]): filename
            for filename in json_files
        }

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

def prepare_data(cfg: DictConfig, return_datasets: bool = False):
    """Prepare evaluation data with robust error handling using Hydra config."""
    # from jarc_reactor.config import Config # Old config system no longer needed
    
    logger = logging.getLogger(__name__)

    batch_size = cfg.training.batch_size # Assuming eval uses training batch size for now
    data_dir = cfg.evaluation.data_dir
    include_synthetic = cfg.evaluation.include_synthetic_data
    synthetic_data_path = cfg.evaluation.synthetic_data_dir

    effective_data_dir = data_dir
    is_synthetic_mode = False
    if include_synthetic and synthetic_data_path:
        logger.info(f"Including synthetic data for evaluation from: {synthetic_data_path}")
        # This script seems to load EITHER normal OR synthetic, not merge.
        # The old logic with global `include_synthetic_training_data` and `synthetic_dir` implied this.
        # For now, let's assume if include_synthetic is true, we ONLY use synthetic_data_path for this eval prep.
        # If merging is intended, this logic needs to be more complex.
        effective_data_dir = synthetic_data_path
        is_synthetic_mode = True 

    logger.info(f"Starting data preparation with batch_size={batch_size} from directory {effective_data_dir}")

    # Validate directory
    if not os.path.exists(effective_data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {effective_data_dir}")
    if not os.path.isdir(effective_data_dir):
        raise NotADirectoryError(f"Path is not a directory: {effective_data_dir}")

    # Initialize counters before using them
    total_files = 0
    successful_files = 0
    log_limit = 2
    
    # Get list of JSON files first
    json_files = [f for f in os.listdir(effective_data_dir) if f.endswith('.json')]
    if not json_files:
        raise ValueError(f"No JSON files found in {effective_data_dir}")
    
    # Inspect data structure for a limited number of files
    for filename in json_files:
        total_files += 1
        if successful_files < log_limit:
            if inspect_data_structure(cfg, filename, effective_data_dir): # Pass cfg and use effective_data_dir
                successful_files += 1

    logger.info(f"Found {total_files} JSON files, successfully inspected {successful_files}")
    if successful_files == 0:
        raise ValueError("No valid data files found in directory")

    # Initialize data containers
    train_inputs: List[torch.Tensor] = []
    train_outputs: List[torch.Tensor] = []
    train_task_ids: List[str] = [] # Task IDs are strings initially
    test_inputs: List[torch.Tensor] = []
    test_outputs: List[torch.Tensor] = []
    test_task_ids: List[str] = [] # Task IDs are strings initially
    context_map: Dict[str, ContextPair] = {}
    train_context_pairs: List[ContextPair] = []
    test_context_pairs: List[ContextPair] = [] 

    try:
        # Load context pairs first
        load_context_pairs(effective_data_dir, context_map) # Use effective_data_dir

        # Load main data
        load_main_data_concurrently(
            directory=effective_data_dir, # Use effective_data_dir
            context_map=context_map,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            train_task_ids=train_task_ids,
            train_context_pairs=train_context_pairs,
            test_inputs=test_inputs,
            test_outputs=test_outputs,
            test_task_ids=test_task_ids,
            test_context_pairs=test_context_pairs,
            is_synthetic=is_synthetic_mode # Use the flag derived from cfg
        )

        # Validate data was loaded
        if not train_inputs or not test_inputs:
            raise ValueError("No data loaded from files")

        # Convert lists to tensors with validation
        try:
            train_inputs = torch.stack(train_inputs)
            train_outputs = torch.stack(train_outputs)
            test_inputs = torch.stack(test_inputs)
            test_outputs = torch.stack(test_outputs)
        except Exception as e:
            raise ValueError(f"Failed to convert data to tensors: {str(e)}")

        # Create task mapping
        unique_task_ids = sorted(set(train_task_ids + test_task_ids))
        if not unique_task_ids:
            raise ValueError("No task IDs found in data")
            
        task_id_map = {task_id: idx for idx, task_id in enumerate(unique_task_ids)}
        logger.info(f"Created mapping for {len(unique_task_ids)} unique tasks")

        # Convert task IDs to tensors
        train_task_ids_tensor = torch.tensor([task_id_map[tid] for tid in train_task_ids], dtype=torch.long)
        test_task_ids_tensor = torch.tensor([task_id_map[tid] for tid in test_task_ids], dtype=torch.long)

        # Validate tensor shapes match
        if not (train_inputs.size(0) == train_outputs.size(0) == train_task_ids_tensor.size(0)):
            raise ValueError(
                f"Mismatched training tensor sizes: inputs={train_inputs.size(0)}, "
                f"outputs={train_outputs.size(0)}, task_ids={train_task_ids_tensor.size(0)}"
            )
        if not (test_inputs.size(0) == test_outputs.size(0) == test_task_ids_tensor.size(0)):
            raise ValueError(
                f"Mismatched test tensor sizes: inputs={test_inputs.size(0)}, "
                f"outputs={test_outputs.size(0)}, task_ids={test_task_ids_tensor.size(0)}"
            )

        # Convert context pairs to tensors
        try:
            train_ctx_inputs = torch.stack([pair.context_input for pair in train_context_pairs])
            train_ctx_outputs = torch.stack([pair.context_output for pair in train_context_pairs])
            test_ctx_inputs = torch.stack([pair.context_input for pair in test_context_pairs])
            test_ctx_outputs = torch.stack([pair.context_output for pair in test_context_pairs])
        except Exception as e:
            raise ValueError(f"Failed to process context pairs: {str(e)}")

        # Create datasets
        train_dataset = TensorDataset(
            train_inputs, train_outputs, 
            train_ctx_inputs, train_ctx_outputs, 
            train_task_ids_tensor
        )
        test_dataset = TensorDataset(
            test_inputs, test_outputs,
            test_ctx_inputs, test_ctx_outputs, 
            test_task_ids_tensor
        )

        # Save task mapping for future use
        try:
            with open('eval_id_map.json', 'w') as f:
                json.dump(task_id_map, f, indent=2)
            logger.info("Saved task mapping to eval_id_map.json")
        except Exception as e:
            logger.warning(f"Failed to save task mapping: {str(e)}")
            # Continue anyway - this isn't critical

        # Return appropriate format
        if return_datasets:
            logger.info(f"Returning datasets with {len(train_dataset)} training and {len(test_dataset)} test examples")
            return train_dataset, test_dataset
        else:
            # Create DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
            logger.info("Created data loaders successfully")
            return train_loader, test_loader

    except Exception as e:
        logger.error(f"Failed to prepare data: {str(e)}")
        raise