Based on the provided Task Master documentation (v0.15.0 and surrounding files), here are the key features you should be aware of to effectively set up and utilize it for managing your learning tasks from `RNA_MODELING_SKILL_MAP_CSM.md` and your daily schedule:

**I. Core Task Management & Structure:**

1.  **Centralized Task Database (`tasks.json`):**
    *   All tasks are stored in a single JSON file, typically `PROJECT_ROOT/.taskmaster/tasks.json`. This file is your primary source of truth for tasks and is intended to be version-controlled with Git.
    *   **Relevance:** You can track all your RNA learning modules, projects, and daily scheduled activities in one place.

2.  **Task Attributes (`docs/task-structure.md`):**
    *   Each task has:
        *   `id`: Unique identifier (e.g., "1", "1.1" for subtasks).
        *   `title`: Brief description.
        *   `description`: More concise detail.
        *   `status`: "pending", "in-progress", "done", "deferred", "cancelled", "blocked".
        *   `dependencies`: Array of task IDs that must be completed first.
        *   `priority`: "high", "medium", "low".
        *   `details`: In-depth implementation notes or context.
        *   `testStrategy`: How to verify task completion (more for software, but can be adapted for "mastery criteria" for learning).
        *   `subtasks`: An array for nested tasks.
    *   **Relevance:** Your `curriculum_parser.py` and `task_generator.py` scripts will need to map CSM learning units to these fields. "Mastery criteria" can go into `details` or `testStrategy`.

3.  **Individual Task Files (`task-master generate`):**
    *   Task Master can generate individual Markdown-like files for each task (e.g., `tasks/task-1.md`). These are auto-generated from `tasks.json` and are for easy reference or for AI agents to consume.
    *   **Relevance:** Could be useful if you want to have a separate file for each learning module/project that your AI tools (like Cursor) can easily focus on.

**II. Task Creation & AI-Powered Generation:**

1.  **Project Initialization (`task-master init`):**
    *   Sets up the `.taskmaster/` directory, including a basic `tasks.json` and guides you through `task-master models --setup`.
    *   **Relevance:** The first step to get Task Master running in your `Holistic-Performance-Enhancement` repo.

2.  **Parsing Product Requirements Documents (`task-master parse-prd <prd-file>`):**
    *   AI-powered command to convert a text document (your "PRD," which could be a section of your CSM or a weekly plan) into a structured list of tasks in `tasks.json`.
    *   Supports `--num-tasks` to limit generation.
    *   Supports `--append` to add to existing tasks (v0.12.0).
    *   Has a `--research` flag to use the research AI model for potentially more insightful task generation (v0.15.0).
    *   **Relevance:** Your `task_generator.py` might not need to call `task-master add-task` for every single item if you structure your parsed CSM output into a PRD-like text file that `parse-prd` can consume. This could be a powerful way to ingest large curriculum sections.

3.  **Adding Individual Tasks (`task-master add-task --prompt "..."`):**
    *   AI-powered creation of a single, well-structured task based on your prompt.
    *   Supports `--dependencies`, `--priority`.
    *   Has a `--research` flag.
    *   **Enhanced in v0.15.0:** Automatically analyzes existing tasks to provide better context to the AI for suggesting relevant dependencies. This reduces manual dependency management.
    *   **Relevance:** This is the command your `task_generator.py` will likely call for each learning unit from the CSM. The enhanced context analysis for dependencies is highly valuable.

4.  **Task Expansion (`task-master expand --id <task_id>`):**
    *   Uses AI to break down a complex parent task into a specified number of sub-tasks (`--num`, default 3 or 5 depending on version).
    *   Supports `--prompt` for additional context, `--all` to expand all pending tasks, and `--force` to regenerate.
    *   Integrates with `analyze-complexity` output (see below).
    *   Supports `--research` flag.
    *   **Relevance:** Extremely useful for taking high-level learning objectives or projects from `RNA_MODELING_SKILL_MAP_CSM.md` (e.g., "Pillar 1, Stage 2 Project") and breaking them into smaller, actionable learning/study sub-tasks.

**III. Daily Workflow & Task Tracking:**

1.  **Listing Tasks (`task-master list`):**
    *   Displays tasks with status, ID, title.
    *   Supports `--status` filter and `--with-subtasks` to show hierarchy.
    *   Dependency status is shown with glyphs (✅, ⏱️).
    *   **Relevance:** Your primary way to see your learning backlog and daily scheduled items.

2.  **Showing Next Task (`task-master next`):**
    *   Identifies the next actionable task based on dependencies (all must be "done"), priority, and task ID.
    *   Displays comprehensive details and suggested actions.
    *   **Enhanced in v0.13.0:** More subtask-aware; prioritizes subtasks within an "in-progress" parent.
    *   **Relevance:** Helps you follow your `My_Optimized_Performance_Schedule_v1.0.md` by focusing on what's ready now.

3.  **Showing Task Details (`task-master show <id>`):**
    *   Displays full details for a specific task or subtask, including description, details, test strategy, and subtask status.
    *   **Relevance:** Useful for reviewing a specific learning objective or scheduled block before starting.

4.  **Setting Task Status (`task-master set-status --id <id> --status <status>`):**
    *   Updates task status (e.g., "pending", "in-progress", "done", "deferred").
    *   If a parent task is marked "done", all its subtasks are also marked "done".
    *   Supports multiple IDs.
    *   **Enhanced in v0.15.0:** The response now suggests the next task to work on.
    *   **Relevance:** This is how you'll track your progress through the CSM and daily schedule.

**IV. Advanced Task Management & AI Features:**

1.  **Updating Tasks (`task-master update --from <id>`, `task-master update-task --id <id>`, `task-master update-subtask --id <id.subid>`):**
    *   AI-powered modification of existing tasks based on a new prompt. `update` affects a range, `update-task` a single task, `update-subtask` appends to a subtask's details.
    *   Useful for "implementation drift" or refining learning objectives.
    *   `update-subtask` (v0.13.0) gets context from parent and adjacent subtasks.
    *   **Relevance:** If a learning module in the CSM proves more complex, or your understanding evolves, you can update the task details or test/mastery criteria.

2.  **Dependency Management:**
    *   `add-dependency --id <id> --depends-on <dep_id>`
    *   `remove-dependency --id <id> --depends-on <dep_id>`
    *   `validate-dependencies`: Checks for issues like circular or non-existent dependencies.
    *   `fix-dependencies`: Attempts to automatically resolve issues found by `validate-dependencies`.
    *   **Relevance:** Crucial for structuring your CSM learning path, ensuring foundational modules are completed before advanced ones.

3.  **Task Reorganization (`task-master move --from <id> --to <id>`):**
    *   Added in v0.15.0. Allows moving tasks/subtasks within the hierarchy (e.g., making a task a subtask, reordering subtasks, moving a subtask to a different parent).
    *   Handles creating placeholder tasks if the destination doesn't exist.
    *   Useful for resolving merge conflicts in `tasks.json` if collaborating or for restructuring your learning plan.
    *   **Relevance:** As your understanding of the RNA curriculum evolves, you might want to restructure how learning projects are broken down.

4.  **Complexity Analysis & Reporting:**
    *   `task-master analyze-complexity`: Uses AI to score task complexity (1-10), recommend subtask counts, and generate tailored expansion prompts. Saves a `task-complexity-report.json`. Supports `--research` and specific task IDs (v0.15.0).
    *   `task-master complexity-report`: Displays the analysis report in a human-readable format.
    *   The `expand` command integrates with this report to use its recommendations.
    *   **Relevance:** Can help you gauge the difficulty of CSM modules and decide how to break them down effectively. The "reasoning" field in the report might offer insights.

5.  **Subtask Management (`add-subtask`, `remove-subtask`, `clear-subtasks`):**
    *   Provides fine-grained control over the sub-components of larger tasks.
    *   `add-subtask` can create a new subtask or convert an existing top-level task into a subtask.
    *   `remove-subtask` can delete a subtask or convert it back into a top-level task.
    *   `clear-subtasks --id <id>` or `--all`: Removes all subtasks from specified parent tasks.
    *   **Relevance:** Essential for managing the granular activities within each CSM learning stage or project.

**V. Configuration and AI Model Setup (`task-master models`):**

1.  **Model Roles:** Task Master defines three roles for AI models: `main`, `research`, and `fallback`.
2.  **Configuration File:** `.taskmasterconfig` (in `.taskmaster/` directory) stores your model choices and parameters (maxTokens, temperature).
3.  **Interactive Setup (`task-master models --setup`):** Guides you through selecting models for each role from a list of supported models or by providing custom IDs (e.g., for local Ollama models or specific OpenRouter models).
4.  **Ollama Integration (v0.15.0):** Includes validation against a local Ollama instance and support for custom Ollama models.
5.  **OpenRouter Support (v0.13.0):** Allows using models via OpenRouter.
6.  **API Key Management:** Keys are stored in `.taskmaster/.env` (for CLI) or MCP configuration (`.cursor/mcp.json`). The `task-master models` command will show API key status.
7.  **Token Limits (Evolving Feature - from `max-min-tokens.txt.md`):**
    *   The system is moving towards distinct `maxInputTokens` and `maxOutputTokens` for more precise control.
    *   `supported-models.json` (an internal Task Master file, not user-edited directly but informs the `models` command) will store `contextWindowTokens` (total) and `maxOutputTokens` (generation limit) for known models.
    *   Your `.taskmasterconfig` will allow you to set your preferred `maxInputTokens` and `maxOutputTokens` for each role, which should be validated against the model's absolute capabilities.
    *   The AI service runner will dynamically calculate the token budget for generation based on prompt length and these configured limits.
    *   **Relevance:** Understanding this helps you manage costs and avoid API errors. For very detailed CSM tasks or prompts, you'll need to be mindful of input token limits.

**VI. Workflow & Integration:**

1.  **AI IDE Integration (Cursor/MCP):**
    *   Task Master is designed for strong integration with AI-powered IDEs like Cursor via the Model Control Protocol (MCP).
    *   The `assets/AGENTS.md` and `context/MCP_INTEGRATION.md` files provide context and command mappings for AI agents using Task Master.
    *   **Relevance:** If you use Cursor, you can interact with Task Master using natural language.

2.  **Version Control:** `tasks.json` is Git-versioned, allowing you to track changes to your task list.
3.  **Changesets (`@changesets/cli`):** Used by the Task Master project itself for its own versioning and releases. Not directly a user-facing feature for managing *your* tasks, but good to know it's part of its development DNA.

**Key Features to Leverage for Your Use Case:**

*   **`parse-prd` or `add-task` (via `task_generator.py`):** To get your CSM learning units into Task Master.
*   **`expand` with `--research`:** To break down large CSM projects/stages into manageable sub-tasks with AI assistance.
*   **`list --with-subtasks` and `show <id>`:** To review your learning plan and daily scheduled items.
*   **`next`:** To guide your daily focus based on dependencies.
*   **`set-status`:** To track completion of learning modules and scheduled blocks.
*   **`add-dependency`:** To enforce prerequisites from your CSM.
*   **`move`:** To restructure your learning plan as it evolves.
*   **Labels (applied during `add-task` or manually):** Crucial for filtering tasks by `block:learning-active`, `domain:rna`, `pillar:1`, `type:project`, etc.

By understanding these features, you can effectively use Task Master not just as a to-do list, but as an AI-augmented system for planning, executing, and tracking your complex learning journey and daily schedule within the HPE framework.