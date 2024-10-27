# JARC Reactor: Transformer Model Implementation

## Overview
This project implements a custom Transformer model designed to facilitate advanced sequence modeling and attention mechanisms. The Transformer architecture, originally introduced by Vaswani et al. in "Attention is All You Need," has been extended and customized in this project to enhance its utility across different applications, including 1D sequential data and 2D spatial data tasks.

### Key Features:
- **Transformer Encoder-Decoder Architecture**: Implements the original multi-head self-attention mechanism to capture relationships across sequences effectively.
- **Multi-Head Self-Attention**: Uses multiple attention heads to allow the model to focus on different positions within the input sequence, providing richer context representations.
- **Flexible Positional Encodings**: The model supports both 1D and 2D positional encodings, which help the Transformer understand the positional relationships in the input data.
- **Differential Attention Mechanism**: An optional modification to reduce noise in the attention layers by subtracting redundant context from the attention scores.
- **PyTorch Lightning Integration**: Utilizes PyTorch Lightning for modular training loops, improving efficiency and simplifying the training and validation process.

## Project Structure
- **config.py**: Configuration file to manage model hyperparameters (e.g., `d_model`, `n_layers`, etc.). Allows easy experimentation with different configurations.
- **transformer_model.py**: Core implementation of the Transformer architecture. Includes the standard attention mechanisms and supports both 1D and 2D data through custom positional encodings.
- **train.py**: Implements the PyTorch Lightning training module, handling data loading, training, and validation of the model.
- **run_model.py**: Entry-point script for starting the model training and evaluation using the configured hyperparameters.
- **positional_encoding.py**: Provides utilities for generating 1D and 2D positional encodings that are crucial for injecting sequence and spatial information into the input features.
- **padding_utils.py**: Helper functions to manage input padding, ensuring that the sequences are consistent in length for batch processing.

## Setup Instructions

### Prerequisites
- Python 3.9 or above
- PyTorch
- PyTorch Lightning

Install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

### Running the Model
1. **Configure Hyperparameters**: Edit `config.py` to set the desired hyperparameters, such as model dimension (`d_model`), number of layers (`n_layers`), learning rate (`learning_rate`), etc.
2. **Prepare Data**: Ensure that your input data is properly preprocessed and ready for training. You can adapt the data loading logic in `train.py` to suit your dataset.
3. **Train the Model**: Execute the following command to start training:
   ```bash
   python run_model.py
   ```

   This command will use the configurations from `config.py` to train the Transformer model on the provided dataset. You can monitor the training process using PyTorch Lightning loggers, such as TensorBoard.

## Model Components

### Transformer Encoder
The encoder is built using multiple layers of multi-head self-attention and feed-forward networks. Each encoder layer consists of:
- **Multi-Head Attention**: Computes multiple attention representations in parallel and concatenates them for a comprehensive understanding of different sequence segments.
- **Feed-Forward Neural Network**: A fully connected network that helps in refining the learned representations from the attention mechanism.
- **Layer Normalization and Residual Connections**: Applied after each sub-layer to stabilize training and ensure effective gradient flow.

### Positional Encoding
Since Transformer architectures do not inherently capture the order of input sequences, positional encoding is crucial. This project includes:
- **1D Positional Encoding**: Injects information about the position of tokens in sequential data (e.g., text).
- **2D Positional Encoding**: Used for spatial data (e.g., images), where positional relationships exist in two dimensions. This encoding is useful for vision transformers and tasks involving grid-like structures.

### Differential Attention (Optional)
To enhance the model's focus on significant parts of the context, a **differential attention mechanism** is optionally introduced. This mechanism helps reduce attention noise by subtracting redundant components of attention, leading to more focused attention scores.

### Training Workflow
 modify config.py, and then run run_model.py


