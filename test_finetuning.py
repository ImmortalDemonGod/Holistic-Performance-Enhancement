import torch
from validation_finetuning import ValidationFineTuner
from train import TransformerTrainer
from Utils.data_preparation import prepare_data
import logging
from validation_finetuning import log_exception
import json

def test_fine_tuning():
    try:
        # Load your pretrained model
        model = TransformerTrainer.load_from_checkpoint("your_checkpoint.ckpt")
        model.to("cpu")  # Ensure model is on CPU or adjust as needed
        
        # Create fine-tuner
        fine_tuner = ValidationFineTuner(
            base_model=model,
            checkpoint_path='fine_tuned_models',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Prepare data
        train_loader, val_loader = prepare_data()
        
        # Load task_id_map
        with open('task_id_map.json', 'r') as f:
            task_id_map = json.load(f)
        
        # Run fine-tuning
        results = fine_tuner.evaluate_all_tasks(val_loader, val_loader, task_id_map)
        
        # Print summary
        print("\nResults Summary:")
        success_count = sum(1 for r in results.values() if 'error' not in r)
        print(f"Successfully processed: {success_count}/{len(results)}")
        
        # Print failed tasks
        failed_tasks = [tid for tid, r in results.items() if 'error' in r]
        if failed_tasks:
            print(f"Failed tasks: {failed_tasks}")
        
    except Exception as e:
        log_exception(e, "test_fine_tuning")

if __name__ == "__main__":
    test_fine_tuning()
