# jarc_reactor/data/eval_data_prep.py
import os
import orjson
import json
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset
from jarc_reactor.config import config, include_synthetic_training_data, synthetic_dir
from jarc_reactor.data.context_data import ContextPair
from jarc_reactor.utils.padding_utils import pad_to_fixed_size


# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def inspect_data_structure(filename, directory=None):
    if directory is None:
        directory = config.evaluation.data_dir
    """Debug helper to examine JSON structure"""
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

            if not is_synthetic:
                train_data = data.get('train', [])[1:]
                test_data = data.get('test', [])
            else:
                train_data = data
                test_data = []

            train_results = []
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

            test_results = []
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

def prepare_data(batch_size=None, return_datasets=False, config=None):
    """Prepare evaluation data with robust error handling."""
    from jarc_reactor.config import Config
    
    # Use passed config or create new one
    if config is None:
        config = Config()
        
    logger = logging.getLogger(__name__)

    if batch_size is None:
        batch_size = config.training.batch_size

    directory = config.evaluation.data_dir
    logger.info(f"Starting data preparation with batch_size={batch_size} from directory {directory}")

    # Validate directory
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Data directory does not exist: {directory}")
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    # Initialize counters before using them
    total_files = 0
    successful_files = 0
    log_limit = 2
    
    # Get list of JSON files first
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    if not json_files:
        raise ValueError(f"No JSON files found in {directory}")
    
    # Inspect data structure for a limited number of files
    for filename in json_files:
        total_files += 1
        if successful_files < log_limit:
            if inspect_data_structure(filename, directory):
                successful_files += 1

    logger.info(f"Found {total_files} JSON files, successfully inspected {successful_files}")
    if successful_files == 0:
        raise ValueError("No valid data files found in directory")

    # Initialize data containers
    train_inputs, train_outputs, train_task_ids = [], [], []
    test_inputs, test_outputs, test_task_ids = [], [], []
    context_map = {}
    train_context_pairs, test_context_pairs = [], [], 

    try:
        # Load context pairs
        logger.info("Loading context pairs...")
        load_context_pairs(directory, context_map)
        if not context_map:
            raise ValueError("No context pairs loaded")

        # Load main dataset
        logger.info("Loading main dataset...")
        load_main_data_concurrently(
            directory=directory,
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