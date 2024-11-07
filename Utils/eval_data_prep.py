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
from config import include_synthetic_training_data, synthetic_dir
from Utils.context_data import ContextPair
from Utils.padding_utils import pad_to_fixed_size

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def inspect_data_structure(filename, directory='evaluation'):
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

            # Determine data structure based on directory
            if directory == 'evaluation':
                train_data = data.get('train', [])[1:]  # Skip the first example (used for context)
                test_data = data.get('test', [])
            else:  # 'sythtraining' assumed to have a flat list
                train_data = data[1:]  # Skip the first example if needed
                test_data = []  # Synthetic data may not have a separate test set

            # Process training data
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

            # Process test data
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

def prepare_data(batch_size=None, return_datasets=False):
    import config  # Ensure config is accessible within the function
    if batch_size is None:
        batch_size = config.batch_size  # Use the default from config if not provided
    logger.info(f"Starting data preparation with batch_size={batch_size}...")
    log_limit = 2
    successful_files = 0
    total_files = 0

    # Inspect data structure for a limited number of files
    for directory in ['evaluation']:
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                total_files += 1
                if successful_files < log_limit:
                    if inspect_data_structure(filename, directory):
                        successful_files += 1
                else:
                    break
        if successful_files >= log_limit:
            break

    logger.info(f"Successfully inspected {successful_files}/{total_files} files for structure.")

    train_inputs, train_outputs, train_task_ids = [], [], []
    test_inputs, test_outputs, test_task_ids = [], [], []
    context_map = {}
    train_context_pairs, test_context_pairs = [], []

    # Load context pairs from 'evaluation'
    load_context_pairs('evaluation', context_map)

    # Load main dataset from 'evaluation' with progress bar
    logger.info("Loading evaluation data with progress bar...")
    load_main_data_concurrently(
        directory='evaluation',
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

    # Convert lists to tensors
    train_inputs = torch.stack(train_inputs)
    train_outputs = torch.stack(train_outputs)
    test_inputs = torch.stack(test_inputs)
    test_outputs = torch.stack(test_outputs)

    
    
    # Create a sorted list of unique task_ids
    unique_task_ids = sorted(set(train_task_ids + test_task_ids))
    task_id_map = {task_id: idx for idx, task_id in enumerate(unique_task_ids)}

    # Convert task_ids to tensors
    train_task_ids_tensor = torch.tensor([task_id_map[tid] for tid in train_task_ids], dtype=torch.long)
    test_task_ids_tensor = torch.tensor([task_id_map[tid] for tid in test_task_ids], dtype=torch.long)
    unique_task_ids = sorted(set(train_task_ids + test_task_ids))
    logger.info(f"Total unique task_ids (including synthetic if any): {len(unique_task_ids)}")
    
    # Check for overlapping task_ids between training and test datasets
    if len(unique_task_ids) != len(set(train_task_ids)) + len(set(test_task_ids)):
        logger.warning("There are overlapping task_ids between training and test datasets.")

    task_id_map = {task_id: idx for idx, task_id in enumerate(unique_task_ids)}

    # Convert task_ids to tensors
    # Convert task_ids to tensors using the task_id_map
    train_task_ids_tensor = torch.tensor([task_id_map[tid] for tid in train_task_ids], dtype=torch.long)
    test_task_ids_tensor = torch.tensor([task_id_map[tid] for tid in test_task_ids], dtype=torch.long)

    # Add Diagnostic Logging
    logger.debug(f"Test Data Sizes - test_inputs: {len(test_inputs)}, "
                 f"test_outputs: {len(test_outputs)}, "
                 f"test_task_ids: {len(test_task_ids_tensor)}")

    # Assertions to ensure data consistency
    assert train_inputs.size(0) == train_outputs.size(0) == train_task_ids_tensor.size(0), "Mismatch in training data tensor sizes."
    assert test_inputs.size(0) == test_outputs.size(0) == test_task_ids_tensor.size(0), (
        f"Mismatch in testing data tensor sizes: "
        f"test_inputs={test_inputs.size(0)}, "
        f"test_outputs={test_outputs.size(0)}, "
        f"test_task_ids={test_task_ids_tensor.size(0)}"
    )

    # Convert context pairs to tensors
    train_ctx_inputs = torch.stack([pair.context_input for pair in train_context_pairs])
    train_ctx_outputs = torch.stack([pair.context_output for pair in train_context_pairs])
    test_ctx_inputs = torch.stack([pair.context_input for pair in test_context_pairs])
    test_ctx_outputs = torch.stack([pair.context_output for pair in test_context_pairs])

    # Create TensorDatasets
    train_dataset = TensorDataset(train_inputs, train_outputs, train_ctx_inputs, train_ctx_outputs, train_task_ids_tensor)
    test_dataset = TensorDataset(test_inputs, test_outputs, test_ctx_inputs, test_ctx_outputs, test_task_ids_tensor)

    # Optional: Save the task_id_map
    logger.info("Saving eval_id_map.json with the current task mappings.")
    try:
        with open('eval_id_map.json', 'w') as f:
            json.dump(task_id_map, f)
    except Exception as e:
        logger.error(f"Failed to save eval_id_map.json: {str(e)}")

    if return_datasets:
        logger.info("Returning TensorDatasets instead of DataLoaders.")
        return train_dataset, test_dataset
    else:
        # Create DataLoaders using the local batch_size variable
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)  # Set batch_size=1 for evaluation

        logger.info("Data preparation completed successfully.")
        return train_loader, val_loader
