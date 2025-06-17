# JARC-Reactor: An Integrated System for the ARC Prize

This document provides an overview of the **JARC-Reactor**, a machine learning system integrated into the `Holistic-Performance-Enhancement` project to serve as the foundational model for the Abstraction and Reasoning Corpus (ARC) Prize challenge.

---

## 1. Overview

The JARC-Reactor is a Transformer-based model built with PyTorch Lightning and configured using Hydra. It is designed to learn and solve abstract reasoning tasks presented as 2D grids of colored cells. The system includes a complete pipeline for data preparation, model training, and evaluation, with a unified logging and artifact management system.

## 2. System Architecture

The integrated system is organized as follows:

-   `cultivation/systems/arc_reactor/`: The root of the integrated system.
    -   `jarc_reactor/`: The core Python package for the model.
        -   `conf/`: Hydra configuration files (`.yaml`).
        -   `data/`: Data loading and preparation modules.
        -   `models/`: Core model components.
        -   `utils/`: Utility scripts and the main `LightningModule`.
        -   `run_model.py`: The main entry point for training and evaluation.
    -   `logs/`: **Unified directory for all logs and artifacts.**
    -   `README.md`: This file.

## 3. Getting Started

All commands should be run from the root of the `Holistic-Performance-Enhancement` project to ensure correct path resolution.

### Prerequisites

Ensure all dependencies from the root `requirements.txt` file are installed. Key dependencies include:
-   PyTorch & PyTorch Lightning
-   Hydra & OmegaConf
-   NumPy & Pandas

### Running the Model

To run a training session, execute the `run_model.py` script as a module. Any parameter from the `jarc_reactor/conf/` directory can be overridden via the command line.

**Base Command:**
```bash
python -m cultivation.systems.arc_reactor.jarc_reactor.run_model [HYDRA_OVERRIDES]
```

**Example: "First Light" Integration Test**

This command runs a quick, one-epoch cycle to verify the system is working correctly. It uses the default logging paths.

```bash
python -m cultivation.systems.arc_reactor.jarc_reactor.run_model \
    training.max_epochs=1 \
    training.batch_size=1 \
    logging.level=DEBUG
```

### Running Tests

The unit tests are located in `cultivation/systems/arc_reactor/jarc_reactor/tests/`. To run them, use `pytest`:

```bash
pytest cultivation/systems/arc_reactor/jarc_reactor/tests/
```

## 4. Unified Logging System

All outputs from the system are consolidated into the `cultivation/systems/arc_reactor/logs/` directory, organized as follows:

```
logs/
├── app/
│   ├── .hydra/             # Hydra's run-specific outputs (configs, overrides)
│   └── jarc_reactor_app.log  # Main application log file
│
└── training/
    ├── checkpoints/        # Saved model checkpoints (.ckpt)
    ├── lightning_logs/     # TensorBoard event files
    │   └── version_X/
    ├── metrics_version_X.csv # Extracted metrics per run
    └── training_plots.png    # Visualization of training metrics
```

-   **Application Logs (`logs/app`):** Contains the main application log file and all of Hydra's metadata for each run, ensuring reproducibility.
-   **Training Artifacts (`logs/training`):** Contains all outputs from PyTorch Lightning, including TensorBoard logs for visualization, model checkpoints, and extracted metric reports.

## 5. Configuration

The system uses [Hydra](https://hydra.cc/) for all configuration.

-   **Main Config**: `jarc_reactor/conf/config.yaml`
-   **Config Groups**: Stored in subdirectories within `jarc_reactor/conf/` (e.g., `training`, `logging`).
-   **Schema**: The structure of the configuration is defined in `jarc_reactor/config_schema.py`.

## 6. Continuous Integration (CI)

This system is covered by a CI workflow defined in `.github/workflows/arc-ci.yml`. This workflow automatically runs linting and unit tests on any changes pushed to the `cultivation/systems/arc_reactor/` path, ensuring code quality and stability.

## 7. Integration Details

The original `jarc-reactor` codebase was integrated into this project using `git subtree`. All import paths were refactored to align with the `cultivation` project structure, and dependencies were harmonized to create a seamless, unified development environment.
