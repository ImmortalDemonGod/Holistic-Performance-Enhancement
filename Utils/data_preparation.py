# Utils/data_preparation.py
import os
import json
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset
from config import include_sythtraining_data, synthetic_dir
from Utils.context_data import ContextPair
from Utils.padding_utils import pad_to_fixed_size

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def inspect_data_structure(filename, directory='training'):
    """Debug helper to examine JSON structure"""
    filepath = os.path.join(directory, filename)
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
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

def load_context_pairs(directory, context_map):
    """Load context pairs from the specified directory into context_map"""
    logger.info(f"Loading context pairs from '{directory}'...")
    for filename in tqdm(os.listdir(directory), desc=f"Loading context pairs from '{directory}'"):
        if not filename.endswith('.json'):
            continue
        task_id = os.path.splitext(filename)[0]
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Determine data structure based on directory
            if directory == 'training':
                examples = data.get('train', [])
            else:  # Assuming 'sythtraining' has a flat list
                examples = data
                
            if not examples:
                logger.warning(f"No examples found in {filepath}. Skipping.")
                continue
            
            # For 'training', use the first train example; for 'sythtraining', use the first example
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
            # logger.debug(f"Created context pair for task '{task_id}' from '{directory}'")
        except Exception as e:
            logger.error(f"Error loading context for task '{task_id}' from '{filepath}': {str(e)}")

def load_main_data(directory, context_map, train_inputs, train_outputs, train_task_ids, train_context_pairs,
                  test_inputs, test_outputs, test_task_ids, test_context_pairs, is_synthetic=False):
    """Load main dataset from the specified directory"""
    logger.info(f"Loading main dataset from '{directory}'{' (synthetic)' if is_synthetic else ''}...")
    for filename in os.listdir(directory):
        if not filename.endswith('.json'):
            continue
        task_id = os.path.splitext(filename)[0]
        filepath = os.path.join(directory, filename)

        if task_id in train_task_ids and not is_synthetic:
            logger.warning(f"Task ID '{task_id}' already exists in training data. Skipping to prevent overlap.")
            continue
            logger.warning(f"Skipping task '{task_id}' - no context pair available")
            continue

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Determine data structure based on directory
            if directory == 'training':
                train_data = data.get('train', [])[1:]  # Skip the first example (used for context)
                test_data = data.get('test', [])
            else:  # 'sythtraining' assumed to have a flat list
                train_data = data[1:]  # Skip the first example if needed
                test_data = []  # Synthetic data may not have a separate test set

            # Load training data
            for item in train_data:
                input_tensor = pad_to_fixed_size(
                    torch.tensor(item['input'], dtype=torch.float32),
                    target_shape=(30, 30)
                )
                output_tensor = pad_to_fixed_size(
                    torch.tensor(item['output'], dtype=torch.float32),
                    target_shape=(30, 30)
                )
                train_inputs.append(input_tensor)
                train_outputs.append(output_tensor)
                train_task_ids.append(task_id)
                train_context_pairs.append(context_map[task_id])
            
            # Load test data if available
            for item in test_data:
                input_tensor = pad_to_fixed_size(
                    torch.tensor(item['input'], dtype=torch.float32),
                    target_shape=(30, 30)
                )
                output_tensor = pad_to_fixed_size(
                    torch.tensor(item['output'], dtype=torch.float32),
                    target_shape=(30, 30)
                )
                test_inputs.append(input_tensor)
                test_outputs.append(output_tensor)
                test_task_ids.append(task_id)
                test_context_pairs.append(context_map[task_id])
        except Exception as e:
            logger.error(f"Error processing file '{filepath}': {str(e)}")

def prepare_data(batch_size=None, return_datasets=False):
    import config  # Ensure config is accessible within the function
    if batch_size is None:
        batch_size = config.batch_size  # Use the default from config if not provided
    logger.info(f"Starting data preparation with batch_size={batch_size}...")
    log_limit = 2
    successful_files = 0
    total_files = 0

    # Inspect data structure for a limited number of files
    for directory in ['training']:
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

    # Load context pairs from 'training'
    load_context_pairs('training', context_map)

    # Conditionally load context pairs from synthetic_dir
    if include_sythtraining_data:
        load_context_pairs(synthetic_dir, context_map)

    # Load main dataset from 'training' with progress bar
    logger.info("Loading training data with progress bar...")
    synthetic_data_source = os.listdir('training')
    for filename in tqdm(synthetic_data_source, desc="Loading training data"):
        if filename.endswith('.json'):
            load_main_data(
                directory='training',
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

    # Conditionally load main dataset from synthetic_dir with progress bar
    if include_sythtraining_data:
        synthetic_data_files = os.listdir(synthetic_dir)
        logger.info("Loading synthetic data with progress bar...")
        for filename in tqdm(synthetic_data_files, desc="Loading synthetic data"):
            if filename.endswith('.json'):
                load_main_data(
                    directory=synthetic_dir,
                    context_map=context_map,
                    train_inputs=train_inputs,
                    train_outputs=train_outputs,
                    train_task_ids=train_task_ids,
                    train_context_pairs=train_context_pairs,
                    test_inputs=test_inputs,
                    test_outputs=test_outputs,
                    test_task_ids=test_task_ids,
                    test_context_pairs=test_context_pairs,
                    is_synthetic=True
                )

    # Stack inputs and outputs
    train_inputs = torch.stack(train_inputs)
    train_outputs = torch.stack(train_outputs)
    test_inputs = torch.stack(test_inputs)
    test_outputs = torch.stack(test_outputs)

    # Create a sorted list of unique task_ids
    unique_task_ids = sorted(set(train_task_ids + test_task_ids))
    logger.info(f"Total unique task_ids (including synthetic if any): {len(unique_task_ids)}")
    
    # Check for overlapping task_ids between training and test datasets
    if len(unique_task_ids) != len(set(train_task_ids)) + len(set(test_task_ids)):
        logger.warning("There are overlapping task_ids between training and test datasets.")

    task_id_map = {task_id: idx for idx, task_id in enumerate(unique_task_ids)}

    # Encode task_ids as integers using the mapping
    train_task_ids_encoded = [task_id_map[tid] for tid in train_task_ids]
    test_task_ids_encoded = [task_id_map[tid] for tid in test_task_ids]

    # Convert the encoded task_ids to tensors
    train_task_ids_tensor = torch.tensor(train_task_ids_encoded, dtype=torch.long)
    test_task_ids_tensor = torch.tensor(test_task_ids_encoded, dtype=torch.long)

    # Convert ContextPair lists to tensors
    train_context_inputs = torch.stack([pair.context_input for pair in train_context_pairs])
    train_context_outputs = torch.stack([pair.context_output for pair in train_context_pairs])
    test_context_inputs = torch.stack([pair.context_input for pair in test_context_pairs])
    test_context_outputs = torch.stack([pair.context_output for pair in test_context_pairs])

    # Add Diagnostic Logging
    logger.debug(f"Test Data Sizes - test_inputs: {len(test_inputs)}, "
                 f"test_outputs: {len(test_outputs)}, "
                 f"test_context_inputs: {len(test_context_pairs)}, "
                 f"test_context_outputs: {len(test_context_pairs)}, "
                 f"test_task_ids: {len(test_task_ids)}")

    # Assertions to ensure data consistency
    assert train_inputs.size(0) == train_outputs.size(0) == train_context_inputs.size(0) == train_context_outputs.size(0) == train_task_ids_tensor.size(0), "Mismatch in training data tensor sizes."
    assert test_inputs.size(0) == test_outputs.size(0) == test_context_inputs.size(0) == test_context_outputs.size(0) == test_task_ids_tensor.size(0), (
        f"Mismatch in testing data tensor sizes: "
        f"test_inputs={test_inputs.size(0)}, "
        f"test_outputs={test_outputs.size(0)}, "
        f"test_context_inputs={test_context_inputs.size(0)}, "
        f"test_context_outputs={test_context_outputs.size(0)}, "
        f"test_task_ids={test_task_ids_tensor.size(0)}"
    )

    # Create TensorDatasets with context data
    train_dataset = TensorDataset(train_inputs, train_outputs, train_context_inputs, train_context_outputs, train_task_ids_tensor)
    test_dataset = TensorDataset(test_inputs, test_outputs, test_context_inputs, test_context_outputs, test_task_ids_tensor)

    # Optional: Save the task_id_map
    logger.info("Saving task_id_map.json with the current task mappings.")
    try:
        with open('task_id_map.json', 'w') as f:
            json.dump(task_id_map, f)
    except Exception as e:
        logger.error(f"Failed to save task_id_map.json: {str(e)}")

    if return_datasets:
        logger.info("Returning TensorDatasets instead of DataLoaders.")
        return train_dataset, test_dataset
    else:
        # Create DataLoaders using the local batch_size variable
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        logger.info("Data preparation completed successfully.")
        return train_loader, val_loader
