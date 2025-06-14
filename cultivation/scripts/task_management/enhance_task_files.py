import json
import os

TASKS_JSON_PATH = os.path.join(os.getcwd(), 'tasks', 'tasks.json')
TASKS_DIR = os.path.join(os.getcwd(), 'tasks')

def format_hpe_learning_meta(meta):
    """
    Formats HPE learning metadata as a structured multiline string.
    
    If the input is not a dictionary or is missing, returns "  Not specified\n". Each key-value pair is displayed on a new line, with keys capitalized and underscores replaced by spaces.
    """
    if not meta or not isinstance(meta, dict):
        return "  Not specified\n"
    formatted_meta = ""
    for key, value in meta.items():
        formatted_meta += f"    {key.replace('_', ' ').capitalize()}: {value}\n"
    return formatted_meta

def enhance_task_files():
    """
    Enhances task text files with detailed subtask information from a JSON source.
    
    Reads tasks and their subtasks from a JSON file, then updates or appends structured subtask details to corresponding task text files. Handles missing files, malformed JSON, and unexpected data structures gracefully, providing informative messages for each case.
    """
    print(f"Reading tasks from: {TASKS_JSON_PATH}")
    try:
        with open(TASKS_JSON_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {TASKS_JSON_PATH} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {TASKS_JSON_PATH}.")
        return

    tasks_list = []
    if isinstance(data, list): # tasks.json is a list of tasks
        tasks_list = data
    elif isinstance(data, dict) and 'tasks' in data and isinstance(data['tasks'], list): # tasks.json is an object with a 'tasks' list
        tasks_list = data['tasks']
    else:
        print("Error: tasks.json does not have the expected structure (a list of tasks or an object with a 'tasks' list).")
        return

    print(f"Found {len(tasks_list)} tasks in tasks.json.")

    for task in tasks_list:
        task_id = task.get('id')
        if task_id is None:
            print("Warning: Found a task without an ID. Skipping.")
            continue

        subtasks = task.get('subtasks')
        if not subtasks or not isinstance(subtasks, list):
            # print(f"Task {task_id} has no subtasks or subtasks are not a list. Skipping enhancement.")
            continue

        task_file_name = f"task_{str(task_id).zfill(3)}.txt"
        task_file_path = os.path.join(TASKS_DIR, task_file_name)

        if not os.path.exists(task_file_path):
            print(f"Warning: Task file {task_file_path} not found for task {task_id}. Skipping.")
            continue

        print(f"Enhancing {task_file_path} for task {task_id}...")

        try:
            with open(task_file_path, 'r') as f:
                original_content = f.read()
        except Exception as e:
            print(f"Error reading {task_file_path}: {e}")
            continue
        
        base_content = original_content
        subtask_details_marker = "\n# Subtask Details:\n"
        new_enhancement_prefix = "\n\n# Subtask Details:\n"

        if subtask_details_marker in original_content:
            print(f"Found existing subtask details in {task_file_path}. Replacing.")
            parts = original_content.split(subtask_details_marker, 1)
            base_content = parts[0]
            new_enhancement_prefix = "\n# Subtask Details:\n" # Use single newline if replacing from marker
        else:
            print(f"No existing subtask details found in {task_file_path}. Appending.")

        enhancement_content = new_enhancement_prefix

        for subtask in subtasks:
            sub_id = subtask.get('id')
            sub_title = subtask.get('title', 'N/A')
            enhancement_content += f"\n## Subtask {task_id}.{sub_id}: {sub_title}\n"
            
            description = subtask.get('description', 'Not specified')
            enhancement_content += f"Description: {description}\n"
            
            dependencies = subtask.get('dependencies', [])
            enhancement_content += f"Dependencies: {dependencies if dependencies else 'None'}\n"
            
            status = subtask.get('status', 'pending')
            enhancement_content += f"Status: {status}\n"
            
            risks = subtask.get('risks', 'Not specified')
            enhancement_content += f"Risks: {risks}\n"
            
            mitigation = subtask.get('mitigation', 'Not specified')
            enhancement_content += f"Mitigation: {mitigation}\n"
            
            hpe_meta = subtask.get('hpe_learning_meta')
            enhancement_content += "HPE Learning Meta:\n"
            enhancement_content += format_hpe_learning_meta(hpe_meta)

            clarification = subtask.get('clarification')
            if clarification:
                enhancement_content += f"Clarification: {clarification}\n"

            implementation_details = subtask.get('implementation_details')
            if implementation_details and isinstance(implementation_details, dict):
                steps = implementation_details.get('steps')
                if steps and isinstance(steps, list):
                    enhancement_content += "Implementation Steps:\n"
                    for step in steps:
                        enhancement_content += f"  - {step}\n"
                
                testing = implementation_details.get('testing')
                if testing and isinstance(testing, list):
                    enhancement_content += "Testing Approach:\n"
                    for test_point in testing:
                        enhancement_content += f"  - {test_point}\n"
            
            refinement = subtask.get('refinement')
            if refinement:
                enhancement_content += f"Refinement: {refinement}\n"

        final_content = base_content + enhancement_content
        try:
            with open(task_file_path, 'w') as f:
                f.write(final_content)
            print(f"Successfully updated {task_file_path}")
        except Exception as e:
            print(f"Error writing to {task_file_path}: {e}")

if __name__ == "__main__":
    enhance_task_files()
    print("Task file enhancement process complete.")
