# Transformer Model with Custom Sigmoid Activation.

This project implements a Transformer model with a custom sigmoid activation function for processing grid-based input data. The model is designed to handle input and output grids ranging from 1x1 to 30x30, with values between 0 and 9. The data is loaded from JSON files located in the `training` folder, where each file contains training and testing examples.

## Project Structure

- **`config.py`**: Contains configuration parameters for the model, such as dimensions, learning rate, and batch size.
- **`custom_activation.py`**: Defines the custom sigmoid activation function used in the model.
- **`padding_utils.py`**: Provides a utility function to pad input and output tensors to a fixed size of 30x30.
- **`positional_encoding.py`**: Implements positional encoding for the Transformer model.
- **`transformer_model.py`**: Contains the implementation of the Transformer model.
- **`train.py`**: Defines the training module and data preparation function.
- **`run_model.py`**: Script to initiate the training process using PyTorch Lightning.
- **`training/`**: Directory containing JSON files with training and testing data.

## Requirements

- Python 3.7+
- PyTorch
- PyTorch Lightning
- Other dependencies as specified in your environment

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/leatherman55/JARC-Reactor.git
   cd <repository-directory>
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

Ensure that the `training` folder contains JSON files with the following structure:
- Each JSON file has a `train` section with at least two examples and a `test` section with one example.
- Each example consists of an `input` and `output` grid.

## Running the Model

1. Configure the model parameters in `config.py` as needed.

2. Run the model training script:
   ```bash
   python run_model.py
   ```

   This will load the data, pad it to the required size, and start the training process using the specified configuration.

## Notes

- The model uses a custom sigmoid activation function to constrain output values between -1 and 9.
- All data is padded to a 30x30 grid using a padding token of -1 to ensure consistent input sizes.
- The training process is configured to run on the CPU.

## Troubleshooting

- Ensure all JSON files in the `training` folder are correctly formatted.
- Verify that all dependencies are installed and compatible with your Python version.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
