import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from config import include_sythtraining_data, batch_size
from Utils.context_data import ContextPair
from Utils.padding_utils import pad_to_fixed_size
import logging

# Initialize logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def inspect_data_structure(filename):
    """Debug helper to examine JSON structure"""
    try:
        with open(os.path.join('training', filename), 'r') as f:
            data = json.load(f)
            logger.debug(f"File structure for {filename}:")
            logger.debug(f"Keys in data: {list(data.keys())}")
            logger.debug(f"Number of train examples: {len(data['train'])}")
            logger.debug(f"Number of test examples: {len(data['test'])}")
            logger.debug(f"Sample train input shape: {np.array(data['train'][0]['input']).shape}")
            # Added lines to check context keys
            logger.debug(f"Sample train context_input exists: {'context_input' in data['train'][0]}")
            logger.debug(f"Sample train context_output exists: {'context_output' in data['train'][0]}")
            logger.debug(f"Sample test context_input exists: {'context_input' in data['test'][0]}")
            logger.debug(f"Sample test context_output exists: {'context_output' in data['test'][0]}")
            return True
    except Exception as e:
        logger.error(f"Error inspecting {filename}: {str(e)}")
        return False

def prepare_data():
    logger.info("Starting data preparation...")
    log_limit = 2
    log_count = 0
    successful_files = 0
    total_files = 0
    # Removed warning_log_limit and warning_log_count as they are no longer needed
    for filename in os.listdir('training'):
        if filename.endswith('.json'):
            total_files += 1
            log_count += 1
            if log_count <= log_limit and inspect_data_structure(filename):
                successful_files += 1
    logger.info(f"Successfully processed {successful_files}/{total_files} files")
    if log_count > log_limit:
        logger.debug(f"Suppressed logging for additional files after the first {log_limit}.")
        
    train_inputs, train_outputs, train_task_ids = [], [], []
    test_inputs, test_outputs, test_task_ids = [], [], []
    train_context_pairs, test_context_pairs = [], []

    # Iterate over all JSON files in the 'training' directory
    for filename in os.listdir('training'):
        if filename.endswith('.json'):
            # Extract task_id from filename (e.g., 'task_1.json' -> 'task_1')
            task_id = os.path.splitext(filename)[0]

            #logger.info(f"Loading training data for task_id: {task_id} from file: {filename}")
            with open(os.path.join('training', filename), 'r') as f:
                data = json.load(f)

            # Extract and pad training data
            for item in data['train']:
                input_tensor = pad_to_fixed_size(torch.tensor(item['input'], dtype=torch.float32), target_shape=(30, 30))
                output_tensor = pad_to_fixed_size(torch.tensor(item['output'], dtype=torch.float32), target_shape=(30, 30))
                train_inputs.append(input_tensor)
                train_outputs.append(output_tensor)

                # Create and append ContextPair using 'input' and 'output'
                context_input = input_tensor  # Using input_tensor as context_input
                context_output = output_tensor  # Using output_tensor as context_output
                context_pair = ContextPair(context_input=context_input, context_output=context_output)
                train_context_pairs.append(context_pair)
                test_context_pairs.append(context_pair)

                # Assign task_id based on filename
                train_task_ids.append(task_id)

            # Extract and pad test data
            for item in data['test']:
                input_tensor = pad_to_fixed_size(torch.tensor(item['input'], dtype=torch.float32), target_shape=(30, 30))
                output_tensor = pad_to_fixed_size(torch.tensor(item['output'], dtype=torch.float32), target_shape=(30, 30))
                test_inputs.append(input_tensor)
                test_outputs.append(output_tensor)

                # Create and append ContextPair using 'input' and 'output'
                context_input = input_tensor  # Using input_tensor as context_input
                context_output = output_tensor  # Using output_tensor as context_output
                context_pair = ContextPair(context_input=context_input, context_output=context_output)
                # Assign task_id based on filename
                test_task_ids.append(task_id)

    # Conditionally load data from the 'sythtraining' directory
    if include_sythtraining_data:
        logger.info("Including synthetic training data from 'sythtraining' directory.")
        for filename in os.listdir('sythtraining'):
            if filename.endswith('.json'):
                # Extract task_id from filename
                task_id = os.path.splitext(filename)[0]

                logger.info(f"Loading synthetic training data for task_id: {task_id}")
                with open(os.path.join('sythtraining', filename), 'r') as f:
                    data = json.load(f)

                # Extract and pad data
                for item in data:
                    input_tensor = pad_to_fixed_size(torch.tensor(item['input'], dtype=torch.float32), target_shape=(30, 30))
                    output_tensor = pad_to_fixed_size(torch.tensor(item['output'], dtype=torch.float32), target_shape=(30, 30))
                    train_inputs.append(input_tensor)
                    train_outputs.append(output_tensor)

                    # Assign task_id based on filename
                    train_task_ids.append(task_id)
    # Stack inputs, outputs, and task_ids
    train_inputs = torch.stack(train_inputs)
    train_outputs = torch.stack(train_outputs)
    test_inputs = torch.stack(test_inputs)
    test_outputs = torch.stack(test_outputs)

    # Create a sorted list of unique task_ids
    unique_task_ids = sorted(set(train_task_ids + test_task_ids))

    logger.info(f"Total unique task_ids (including synthetic if any): {len(unique_task_ids)}")
    if len(unique_task_ids) != (len(set(train_task_ids)) + len(set(test_task_ids))):
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

    # Ensure that all tensors have matching sizes
    assert train_inputs.size(0) == train_outputs.size(0) == train_context_inputs.size(0) == train_context_outputs.size(0) == train_task_ids_tensor.size(0), "Mismatch in training data tensor sizes."
    assert test_inputs.size(0) == test_outputs.size(0) == test_context_inputs.size(0) == test_context_outputs.size(0) == test_task_ids_tensor.size(0), "Mismatch in testing data tensor sizes."

    # Create TensorDatasets with context data
    train_dataset = TensorDataset(train_inputs, train_outputs, train_context_inputs, train_context_outputs, train_task_ids_tensor)
    test_dataset = TensorDataset(test_inputs, test_outputs, test_context_inputs, test_context_outputs, test_task_ids_tensor)

    # Optional: Save the task_id_map
    logger.info("Saved task_id_map.json with the current task mappings.")
    with open('task_id_map.json', 'w') as f:
        json.dump(task_id_map, f)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader
