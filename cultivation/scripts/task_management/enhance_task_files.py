import json
import os

# Get the script's directory and navigate to repository root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
TASKS_JSON_PATH = os.path.join(REPO_ROOT, '.taskmaster', 'tasks', 'tasks.json')
TASKS_DIR = os.path.join(REPO_ROOT, '.taskmaster', 'tasks')

def format_hpe_learning_meta(meta):
    if not meta or not isinstance(meta, dict):
        return "  Not specified\n"
    formatted_meta = ""
    for key, value in meta.items():
        formatted_meta += f"    {key.replace('_', ' ').capitalize()}: {value}\n"
    return formatted_meta

def enhance_task_files():
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

    all_tasks_list = []
    if isinstance(data, list):
        all_tasks_list = data
    elif isinstance(data, dict):
        # New check for task-master v0.17.0 structure with tags (e.g., "master")
        if "master" in data and \
           isinstance(data.get("master"), dict) and \
           "tasks" in data.get("master") and \
           isinstance(data.get("master").get("tasks"), list):
            all_tasks_list = data["master"]["tasks"]
        # Fallback to old structure: root dictionary with a "tasks" list
        elif 'tasks' in data and isinstance(data.get('tasks'), list):
            all_tasks_list = data['tasks']
        # Fallback to old structure: root dictionary with other top-level keys whose values are task lists
        else:
            for key in data:
                if isinstance(data.get(key), list):
                    all_tasks_list.extend(data[key])
    
    if not all_tasks_list:
        print("No tasks found in the expected structure in tasks.json.")
        return

    print(f"Found {len(all_tasks_list)} total tasks in tasks.json.")

    if not os.path.exists(TASKS_DIR):
        print(f"Creating tasks directory: {TASKS_DIR}")
        os.makedirs(TASKS_DIR, exist_ok=True)

    for task in all_tasks_list:
        task_id = task.get('id')
        if task_id is None:
            print("Warning: Found a task without an ID. Skipping.")
            continue

        task_file_name = f"task_{str(task_id).zfill(3)}.txt"
        task_file_path = os.path.join(TASKS_DIR, task_file_name)

        print(f"Processing task {task_id} -> {task_file_path}...")

        main_task_content = f"# Task ID: {task_id}\n"
        main_task_content += f"# Title: {task.get('title', 'N/A')}\n"
        main_task_content += f"# Status: {task.get('status', 'pending')}\n"
        
        dependencies = task.get('dependencies', [])
        dep_str = ', '.join(map(str, dependencies)) if dependencies else 'None'
        main_task_content += f"# Dependencies: {dep_str}\n"
        
        main_task_content += f"# Priority: {task.get('priority', 'medium')}\n\n"
        
        main_task_content += f"# Description:\n{task.get('description', 'Not specified')}\n\n"
        main_task_content += f"# Details:\n{task.get('details', 'Not specified')}\n\n"
        main_task_content += f"# Test Strategy:\n{task.get('testStrategy', 'Not specified')}\n"

        main_hpe_meta = task.get('hpe_learning_meta')
        if main_hpe_meta and isinstance(main_hpe_meta, dict) and main_hpe_meta:
            main_task_content += "\n# HPE Learning Meta (Main Task):\n"
            main_task_content += format_hpe_learning_meta(main_hpe_meta)

        enhancement_content = ""
        subtasks = task.get('subtasks')
        if subtasks and isinstance(subtasks, list) and len(subtasks) > 0:
            enhancement_content += "\n\n# Subtask Details:\n"
            for subtask in subtasks:
                sub_id = subtask.get('id')
                sub_title = subtask.get('title', 'N/A')
                enhancement_content += f"\n## Subtask {task_id}.{sub_id}: {sub_title}\n"
                enhancement_content += f"Description: {subtask.get('description', 'Not specified')}\n"

                # Add Subtask Details if present
                sub_details_content = subtask.get('details')
                if sub_details_content:
                    # Ensure consistent newlines before and after this block
                    enhancement_content += f"Subtask Details:\n{sub_details_content.strip()}\n\n" # Use strip() to clean potential leading/trailing whitespace from JSON

                sub_dependencies = subtask.get('dependencies', [])
                sub_dep_str = ', '.join(map(str, sub_dependencies)) if sub_dependencies else 'None'
                enhancement_content += f"Dependencies: {sub_dep_str}\n"
                
                enhancement_content += f"Status: {subtask.get('status', 'pending')}\n"
                enhancement_content += f"Risks: {subtask.get('risks', 'Not specified')}\n"
                enhancement_content += f"Mitigation: {subtask.get('mitigation', 'Not specified')}\n"
                
                hpe_meta = subtask.get('hpe_learning_meta')
                if hpe_meta and isinstance(hpe_meta, dict) and hpe_meta:
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
                        for step_idx, step in enumerate(steps):
                            enhancement_content += f"  {step_idx + 1}. {step}\n"
                    
                    testing = implementation_details.get('testing')
                    if testing and isinstance(testing, list):
                        enhancement_content += "Testing Approach:\n"
                        for test_idx, test_point in enumerate(testing):
                            enhancement_content += f"  - {test_point}\n"
                
                refinement = subtask.get('refinement')
                if refinement:
                    enhancement_content += f"Refinement: {refinement}\n"
        
        final_content = main_task_content.strip() + "\n" + enhancement_content.strip() + "\n"
        final_content = final_content.replace('\n\n\n', '\n\n') # Clean up excessive newlines

        try:
            with open(task_file_path, 'w') as f:
                f.write(final_content)
            print(f"Successfully generated/updated {task_file_path}")
        except Exception as e:
            print(f"Error writing to {task_file_path}: {e}")

if __name__ == "__main__":
    enhance_task_files()
    print("Task file enhancement process complete.")
