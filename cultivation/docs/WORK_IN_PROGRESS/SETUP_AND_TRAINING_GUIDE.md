# GEORGE Project: Setup and Training Guide

This guide provides step-by-step instructions for setting up the GEORGE project environment, preparing the necessary data, and running the training script.

## 1. Prerequisites

- Python 3.9 or higher
- `pip` (Python package installer)

## 2. Environment Setup

These steps should be performed from your terminal.

### a. Navigate to the Project Directory

First, navigate into the main directory for the GEORGE system.

```bash
cd /path/to/Holistic-Performance-Enhancement/cultivation/systems/george/
```

### b. Create a Python Virtual Environment

It is highly recommended to use a virtual environment to manage project-specific dependencies.

```bash
python3 -m venv venv
```

This creates a `venv` directory within the project folder.

### c. Activate the Virtual Environment

Before installing dependencies or running scripts, you must activate the environment.

- **On macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```
- **On Windows:**
  ```bash
  .\venv\Scripts\activate
  ```

You will know the environment is active when your shell prompt is prefixed with `(venv)`.

### d. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```
*Note: If you encounter errors during training related to `tensorboard`, you may need to install it separately:*
```bash
pip install tensorboard
```

## 3. Data Setup

The GEORGE model requires the ARC-AGI dataset to be placed in the `data/` directory. The `ArcAgiDataModule` expects a specific file naming convention for training and validation (evaluation) sets.

### Required Files

Ensure the following files exist in the `data/` directory within your GEORGE project folder:

- `arc-agi_training_challenges.json`: The input-output examples for training tasks.
- `arc-agi_training_solutions.json`: The ground-truth solutions for the test portion of training tasks.
- `arc-agi_evaluation_challenges.json`: The input-output examples for validation tasks.
- `arc-agi_evaluation_solutions.json`: The ground-truth solutions for the test portion of validation tasks.

### Example Data

Below is the content for a minimal, working dataset that separates training and validation data correctly.

<details>
<summary><b><code>data/arc-agi_training_challenges.json</code></b></summary>

```json
{
  "task_identity_001": {
    "train": [
      {
        "input": [[1, 0], [0, 0]],
        "output": [[1, 0], [0, 0]]
      }
    ],
    "test": [
      {
        "input": [[0, 2], [0, 0]]
      }
    ]
  },
  "task_change_002": {
    "train": [
      {
        "input": [[3, 0], [0, 0]],
        "output": [[4, 0], [0, 0]]
      }
    ],
    "test": [
      {
        "input": [[5, 0], [0, 0]]
      }
    ]
  }
}
```
</details>

<details>
<summary><b><code>data/arc-agi_training_solutions.json</code></b></summary>

```json
{
  "task_identity_001": {
    "test": {
      "output": [[0, 2], [0, 0]]
    }
  },
  "task_change_002": {
    "test": {
      "output": [[0, 6], [0, 0]]
    }
  }
}
```
</details>

<details>
<summary><b><code>data/arc-agi_evaluation_challenges.json</code></b></summary>

```json
{
  "val_task_identity_003": {
    "train": [
      {
        "input": [[7, 7], [0, 0]],
        "output": [[7, 7], [0, 0]]
      }
    ],
    "test": [
      {
        "input": [[8, 0], [8, 0]]
      }
    ]
  },
  "val_task_change_004": {
    "train": [
      {
        "input": [[1, 1], [0, 0]],
        "output": [[2, 2], [0, 0]]
      }
    ],
    "test": [
      {
        "input": [[3, 3], [0, 0]]
      }
    ]
  }
}
```
</details>

<details>
<summary><b><code>data/arc-agi_evaluation_solutions.json</code></b></summary>

```json
{
  "val_task_identity_003": {
    "test": {
      "output": [[8, 0], [8, 0]]
    }
  },
  "val_task_change_004": {
    "test": {
      "output": [[4, 4], [0, 0]]
    }
  }
}
```
</details>

## 4. Running Training

With the environment and data set up, you can start training the model.

### Basic Training Command

To run training with the default hyperparameters defined in `config.py`, use:

```bash
python train.py
```

### Overriding Hyperparameters

You can override default settings using command-line arguments. The `train.py` script accepts several arguments.

**Common Arguments:**
- `--batch_size <int>`: The number of tasks per batch.
- `--max_epochs <int>`: The total number of epochs to train for.
- `--lr <float>`: The learning rate for the optimizer.
- `--log_every_n_steps <int>`: How often to log metrics.

### Example Training Command

This command runs training for 10 epochs with a batch size of 2 and a learning rate of 0.001. It also redirects all output to a log file.

```bash
python train.py --batch_size 2 --max_epochs 10 --lr 0.001 > training_run.log 2>&1
```

## 5. Outputs

- **Model Checkpoints**: Saved periodically to the `models/` directory.
- **Logs**: TensorBoard logs are saved to `models/george_logs/`. You can view them by running `tensorboard --logdir models/george_logs/`.
