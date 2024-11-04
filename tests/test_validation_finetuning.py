import unittest
from unittest.mock import MagicMock
from validation_finetuning import ValidationFineTuner

class TestValidationFineTuner(unittest.TestCase):
    def setUp(self):
        # Mock base_model with necessary attributes
        self.mock_model = MagicMock()
        self.mock_model.parameters.return_value = [MagicMock()]
        self.mock_model.to = MagicMock()
        self.mock_model.train = MagicMock()
        self.mock_model.eval = MagicMock()
        
        self.finetuner = ValidationFineTuner(
            base_model=self.mock_model,
            checkpoint_path='fine_tuned_models',
            device='cpu',
            patience=3,
            max_epochs=10
        )

    def test_prepare_task_data_no_data(self):
        # Mock val_loader to return no data for a task
        val_loader = MagicMock()
        val_loader.__iter__.return_value = []
        with self.assertRaises(ValueError):
            self.finetuner.prepare_task_data(val_loader, task_id=999)

    # Add more tests as needed

if __name__ == '__main__':
    unittest.main()
