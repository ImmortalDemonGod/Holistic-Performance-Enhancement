import json
import os

# Define the path to the evaluation_results directory
directory_path = "/workspaces/JARC-Reactor/evaluation_results/submissions"
output_file_path = "/workspaces/JARC-Reactor/submission_results_structure.txt"

# Automatically gather all JSON file paths in the specified directory
file_paths = {
    os.path.splitext(filename)[0]: os.path.join(directory_path, filename)
    for filename in os.listdir(directory_path)
    if filename.endswith(".json")
}

# Function to truncate and provide meaningful data for nested structures
def truncate_data(data, max_items=3, max_depth=10, current_depth=0):
    """
    Truncate the JSON data to provide a summarized structure.

    Parameters:
    - data: The JSON data to truncate.
    - max_items: Maximum number of items to display in lists.
    - max_depth: Maximum depth to traverse in nested structures.
    - current_depth: Current depth of recursion.

    Returns:
    - Truncated data with partial information.
    """
    if isinstance(data, dict):
        truncated_dict = {}
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                if current_depth < max_depth - 1:
                    # Recursively truncate nested structures
                    truncated_dict[key] = truncate_data(value, max_items, max_depth, current_depth + 1)
                else:
                    # At max_depth, show partial data without replacing entire structure
                    if isinstance(value, dict):
                        # Truncate nested dictionaries by showing first few key-value pairs
                        truncated_subdict = {}
                        for subkey, subvalue in list(value.items())[:max_items]:
                            if isinstance(subvalue, (dict, list)):
                                truncated_subdict[subkey] = "..."
                            elif isinstance(subvalue, str):
                                truncated_subdict[subkey] = subvalue[:100] + "..." if len(subvalue) > 100 else subvalue
                            else:
                                truncated_subdict[subkey] = subvalue
                        if len(value) > max_items:
                            truncated_subdict["..."] = f"... and {len(value) - max_items} more keys"
                        truncated_dict[key] = truncated_subdict
                    elif isinstance(value, list):
                        # Truncate nested lists by showing first few items
                        truncated_list = []
                        for item in value[:max_items]:
                            if isinstance(item, (dict, list)):
                                truncated_list.append("...")
                            elif isinstance(item, str):
                                truncated_list.append(item[:100] + "..." if len(item) > 100 else item)
                            else:
                                truncated_list.append(item)
                        if len(value) > max_items:
                            truncated_list.append(f"... and {len(value) - max_items} more items")
                        truncated_dict[key] = truncated_list
            elif isinstance(value, str):
                # Truncate long strings
                truncated_dict[key] = value[:100] + "..." if len(value) > 100 else value
            else:
                # Preserve numerical and boolean values
                truncated_dict[key] = value
        return truncated_dict

    elif isinstance(data, list):
        truncated_list = []
        for i, item in enumerate(data):
            if i < max_items:
                if isinstance(item, (dict, list)):
                    truncated_list.append(truncate_data(item, max_items, max_depth, current_depth + 1))
                elif isinstance(item, str):
                    truncated_list.append(item[:100] + "..." if len(item) > 100 else item)
                else:
                    truncated_list.append(item)
            else:
                truncated_list.append(f"... and {len(data) - max_items} more items")
                break
        return truncated_list

    elif isinstance(data, str):
        # Truncate long strings
        return data[:100] + "..." if len(data) > 100 else data

    else:
        # Return as-is for other data types (e.g., int, float, bool)
        return data

# Pretty-print function for better readability in output
def prettify_truncated_data(truncated_data, indent=2):
    """
    Convert the truncated data into a pretty-printed JSON string.

    Parameters:
    - truncated_data: The truncated JSON data.
    - indent: Indentation level for pretty-printing.

    Returns:
    - A pretty-printed JSON string.
    """
    return json.dumps(truncated_data, indent=indent, ensure_ascii=False)

# Open the output file in write mode
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    # Load and display a sample from each file
    for file_name, file_path in file_paths.items():
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError as e:
                output_file.write(f"\nStructure of '{file_name}':\n")
                output_file.write(f"Error loading JSON: {e}\n")
                continue

            # Truncate and extract meaningful data based on structure
            truncated_data = truncate_data(data)

            # Write the structured and truncated data to the output file
            output_file.write(f"\nStructure of '{file_name}':\n")
            output_file.write(prettify_truncated_data(truncated_data))
            output_file.write("\n")

print(f"Structure information has been saved to '{output_file_path}'.")