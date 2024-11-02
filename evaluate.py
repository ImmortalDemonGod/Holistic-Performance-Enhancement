import argparse
from train import TransformerTrainer, prepare_data
from pytorch_lightning import Trainer

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate Transformer Model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the checkpoint to evaluate'
    )
    args = parser.parse_args()

    # Prepare data loaders
    train_loader, val_loader = prepare_data()
    test_loader = val_loader  # Replace with a separate test loader if available

    # Instantiate the model and load the checkpoint
    model = TransformerTrainer.load_from_checkpoint(args.checkpoint)

    # Set up the Trainer for evaluation
    trainer = Trainer(
        devices=1,              # Use a single device (CPU)
        accelerator='cpu',     # Set explicitly to CPU
        logger=False,          # Disable loggers
        enable_progress_bar=True  # Enable progress bar for visibility
    )

    # Perform evaluation
    results = trainer.test(model, dataloaders=test_loader)

    # Extract and print the test accuracy
    test_accuracy = results[0].get('test_accuracy', None)
    if test_accuracy is not None:
        print(f"Test Accuracy: {test_accuracy:.4f}")
    else:
        print("Test accuracy metric not found.")

if __name__ == '__main__':
    main()
