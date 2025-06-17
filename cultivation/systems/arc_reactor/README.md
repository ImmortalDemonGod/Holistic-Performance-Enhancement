# JARC-Reactor: An Integrated System for the ARC Prize

This document provides an overview of the **JARC-Reactor**, a machine learning system integrated into the `Holistic-Performance-Enhancement` project to serve as the foundational model for the Abstraction and Reasoning Corpus (ARC) Prize challenge.

---

## Table of Contents
1.  [Overview](#overview)
2.  [Getting Started](#getting-started)
    -   [Prerequisites](#prerequisites)
    -   [Running the Model](#running-the-model)
    -   [Running Tests](#running-tests)
3.  [System Architecture](#system-architecture)
4.  [Configuration](#configuration)
5.  [Integration Details](#integration-details)

---

## 1. Overview

The JARC-Reactor is a Transformer-based model built with PyTorch Lightning and configured using Hydra. It is designed to learn and solve abstract reasoning tasks presented as 2D grids of colored cells. The system includes a complete pipeline for data preparation, model training, and evaluation.

## 2. Getting Started

All commands should be run from the root of the `Holistic-Performance-Enhancement` project.

### Prerequisites

Ensure all dependencies are installed. Key dependencies include:
-   PyTorch & PyTorch Lightning
-   Hydra & OmegaConf
-   NumPy & Pandas

*Refer to the project's main dependency files for specific versions.*

### Running the Model

To run a training session, execute the `run_model.py` script as a module to ensure correct import resolution:

```bash
PYTHONPATH=. python -m cultivation.systems.arc_reactor.jarc_reactor.run_model [HYDRA_OVERRIDES]
```

**Example: "First Light" Integration Test**

This command runs a quick, one-step training and validation cycle to verify the system is working correctly:

```bash
PYTHONPATH=. python -m cultivation.systems.arc_reactor.jarc_reactor.run_model \
    training.max_epochs=1 \
    training.batch_size=1 \
    training.fast_dev_run=true \
    logging.level=DEBUG \
    logging.log_dir='jarc_reactor/logs/first_light_test'
```

### Running Tests

The unit tests are located in `cultivation/systems/arc_reactor/jarc_reactor/tests/`. To run them, use `pytest`:

```bash
PYTHONPATH=. pytest cultivation/systems/arc_reactor/jarc_reactor/tests/
```

## 3. System Architecture

The `jarc_reactor` codebase is organized as follows:

-   `jarc_reactor/`: The root of the Python package.
    -   `conf/`: Hydra configuration files (`.yaml`) for managing all parameters.
    -   `data/`: Modules for data loading (`ARCDataset`, `ARCDataModule`) and preparation.
    -   `models/`: Core model components (`TransformerModel`, `ContextEncoderModule`).
    -   `utils/`: Utility scripts, including the `TransformerTrainer` (`LightningModule`) and logging setup.
    -   `run_model.py`: The main entry point for training and evaluation.

## 4. Configuration

The system uses [Hydra](https://hydra.cc/) for configuration.

-   **Main Config**: `jarc_reactor/conf/config.yaml`
-   **Defaults**: Stored in subdirectories within `jarc_reactor/conf/`.
-   **Overrides**: Any parameter can be overridden from the command line.

## 5. Integration Details

The original `jarc-reactor` codebase was integrated into this project using `git subtree`. All import paths were refactored to align with the `cultivation` project structure, and dependencies were harmonized to create a seamless, unified development environment.
