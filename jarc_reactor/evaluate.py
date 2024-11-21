# jarc_reactor/evaluate.py
import logging
import os
import json
from datetime import datetime
from pathlib import Path
import torch
from tqdm import tqdm
from jarc_reactor.config import Config
from jarc_reactor.utils.model_factory import create_transformer_trainer
from jarc_reactor.data.data_preparation import prepare_data as prepare_training_data
from jarc_reactor.data.eval_data_prep import prepare_data as prepare_eval_data
from jarc_reactor.utils.task_mapping import TaskMapper
from jarc_reactor.evaluation.eval_summary import EvaluationSummary
from jarc_reactor.evaluation.batch_processor import BatchProcessor
from jarc_reactor.evaluation.metrics_calculator import MetricsCalculator
from jarc_reactor.utils.metrics import (
    TaskMetricsCollector  # Add this import
)
from jarc_reactor.evaluation.logger import EvaluationLogger


class EvaluationManager:
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        
        # Initialize result storage
        self.results_dir = Path('evaluation_results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize results dictionary
        self.results = {
            'task_summaries': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'model': vars(config.model),
                    'training': vars(config.training)
                }
            }
        }
        # Load model
        self.model = self._load_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.eval_dir = self.config.evaluation.data_dir
        if not os.path.exists(self.eval_dir):
            self.logger.error(f"Evaluation directory not found: {self.eval_dir}")
            raise FileNotFoundError(f"Evaluation directory not found: {self.eval_dir}")
        
        self.logger.info(f"Using evaluation data from: {self.eval_dir}")

        # Initialize metrics collector for each mode
        self.metrics_collectors = {
            'training_train': TaskMetricsCollector(),
            'training_validation': TaskMetricsCollector(),
            'evaluation': TaskMetricsCollector()
        }
        
        # Initialize TaskMapper
        self.task_mapper = TaskMapper(self.logger, self.config)

        # Initialize MetricsCalculator
        self.metrics_calculator = MetricsCalculator(logger=self.logger)

        # Initialize BatchProcessor with config
        self.batch_processor = BatchProcessor(
            model=self.model,
            device=self.device,
            logger=self.logger,
            metrics_calculator=self.metrics_calculator,
            config=self.config  # Pass the config instance
        )

    def setup_logging(self):
        """Setup detailed logging configuration"""
        self.logger = EvaluationLogger.setup_logging()

    def _load_model(self):
        """Load model with error handling and logging"""
        try:
            checkpoint_path = self.config.model.checkpoint_path
            self.logger.debug(f"Attempting to load model from: {checkpoint_path}")
            
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
                
            model = create_transformer_trainer(
                config=self.config,
                checkpoint_path=checkpoint_path
            )
            self.logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def _validate_shapes(self, src, tgt, ctx_input, ctx_output, outputs, mode):
        """Validate tensor shapes and log details"""
        self.logger.debug(f"\nShape validation for {mode}:")
        self.logger.debug(f"src: {src.shape}")
        self.logger.debug(f"tgt: {tgt.shape}")
        self.logger.debug(f"ctx_input: {ctx_input.shape}")
        self.logger.debug(f"ctx_output: {ctx_output.shape}")
        self.logger.debug(f"model outputs: {outputs.shape}")

    def _process_results(self, mode, collector):
        """Process and display results for a specific evaluation mode"""
        try:
            task_summaries = collector.get_task_summary()
            
            results = {
                'task_summaries': {},
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'mode': mode,
                    'config': {
                        'model': vars(self.config.model),
                        'training': vars(self.config.training)
                    }
                }
            }
            
            # Calculate overall metrics
            overall_std_acc = 0
            overall_diff_acc = 0
            total_tasks = len(task_summaries)
            
            for task_id, metrics in task_summaries.items():
                # Get metrics from the correct location in the structure
                std_acc = metrics['metrics'].get('standard_accuracy', {}).get('mean', 0.0)
                diff_acc = metrics['metrics'].get('differential_accuracy', {}).get('mean', 0.0)
                
                # Add to overall metrics
                overall_std_acc += std_acc
                overall_diff_acc += diff_acc
                
                # Create task summary with all information
                task_summary = {
                    'standard_accuracy': std_acc,
                    'differential_accuracy': diff_acc,
                    'predictions': metrics.get('predictions', []),
                    'debug_info': metrics.get('debug_info', {})
                }
                
                # Add to both results and main results storage
                results['task_summaries'][task_id] = task_summary
                if task_id not in self.results['task_summaries']:
                    self.results['task_summaries'][task_id] = {}
                self.results['task_summaries'][task_id][mode] = task_summary
            
            # Calculate averages
            if total_tasks > 0:
                overall_metrics = {
                    'standard_accuracy': overall_std_acc / total_tasks,
                    'differential_accuracy': overall_diff_acc / total_tasks
                }
            else:
                overall_metrics = {
                    'standard_accuracy': 0.0,
                    'differential_accuracy': 0.0
                }
            
            # Add overall metrics to both results
            results['overall_metrics'] = overall_metrics
            if 'overall_metrics' not in self.results:
                self.results['overall_metrics'] = {}
            self.results['overall_metrics'][mode] = overall_metrics
            
            # Save mode-specific results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_file = self.results_dir / f'{mode}_results_{timestamp}.json'
            
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Saved {mode} results to {result_file}")
            
            # Print summary
            print(f"\n{mode.upper()} RESULTS:")
            print(f"Overall Standard Accuracy: {results['overall_metrics']['standard_accuracy']:.4f}")
            print(f"Overall Differential Accuracy: {results['overall_metrics']['differential_accuracy']:.4f}")
            
            print("\nPer-Task Metrics:")
            for task_id, task_metrics in results['task_summaries'].items():
                std_acc = task_metrics['standard_accuracy']
                diff_acc = task_metrics['differential_accuracy']
                print(
                    f"Task {task_id}: "
                    f"Standard Accuracy = {std_acc:.4f}, "
                    f"Differential Accuracy = {diff_acc:.4f}"
                )
                
                if std_acc >= 0.999:
                    print(
                        f"--> Task {task_id} solved with "
                        f"{std_acc*100:.2f}% accuracy\n"
                    )
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error processing results for {mode}: {str(e)}")
            raise

    def evaluate_loader(self, loader, mode, int_to_task):
        """Evaluate all batches in a loader."""
        self.logger.info(f"\nStarting evaluation for {mode}...")
        try:
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Evaluating {mode}")):
                task_results = self.batch_processor.process_batch(batch, mode, int_to_task)
                
                for task_result in task_results:
                    task_id = task_result['task_id']
                    metrics = task_result['metrics']
                    pred_record = task_result['prediction_record']
                    
                    self.metrics_collectors[mode].add_result(
                        task_id,
                        metrics,
                        prediction_record=pred_record
                    )
                
                if (batch_idx + 1) % 10 == 0:
                    self.logger.debug(f"Processed {batch_idx + 1} batches")
                    
        except Exception as e:
            self.logger.error(f"Error in evaluate_loader for {mode}: {str(e)}")
            raise

    def _save_results(self, mode, results):
        """Save results with error handling"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = self.results_dir / f'{mode}_results_{timestamp}.json'
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Saved {mode} results to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving results for {mode}: {str(e)}")

    def generate_summary(self, all_results):
        """Generate a comprehensive evaluation summary with key findings"""
        try:
            summary_generator = EvaluationSummary(self, all_results)
            summary_file = summary_generator.generate_summary()
            return summary_file
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            raise
    
    def run_evaluation(self):
        """Main evaluation loop with error handling"""
        try:
            self.task_mapper.load_training_task_map()
            self.task_mapper.load_evaluation_task_map()
            self.task_mapper.validate_task_maps()
            all_results = {}
            
            # Training data evaluation
            if self.task_mapper.has_training_data:
                self.logger.info("\nPreparing training data...")
                try:
                    train_dataset, val_dataset = prepare_training_data(return_datasets=True)
                    
                    train_loader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=1, shuffle=False
                    )
                    val_loader = torch.utils.data.DataLoader(
                        val_dataset, batch_size=1, shuffle=False
                    )
                    
                    self.evaluate_loader(
                        train_loader, 'training_train', self.task_mapper.train_int_to_task
                    )
                    all_results['training_train'] = self._process_results(
                        'training_train', 
                        self.metrics_collectors['training_train']
                    )
                    
                    self.evaluate_loader(
                        val_loader, 'training_validation', self.task_mapper.train_int_to_task
                    )
                    all_results['training_validation'] = self._process_results(
                        'training_validation',
                        self.metrics_collectors['training_validation']
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating training data: {str(e)}")
            
            # Evaluation data
            if self.task_mapper.has_eval_data:
                self.logger.info("\nPreparing evaluation data...")
                try:
                    _, eval_dataset = prepare_eval_data(return_datasets=True)
                    eval_loader = torch.utils.data.DataLoader(
                        eval_dataset, batch_size=1, shuffle=False
                    )
                    
                    self.evaluate_loader(
                        eval_loader, 'evaluation', self.task_mapper.eval_int_to_task
                    )
                    all_results['evaluation'] = self._process_results(
                        'evaluation',
                        self.metrics_collectors['evaluation']
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating evaluation data: {str(e)}")
            
            # Generate comprehensive summary
            if all_results:
                summary_file = self.generate_summary(all_results)
                self.logger.info(f"Evaluation summary saved to {summary_file}")
            
            if self.config.evaluation.create_submission:
                self.create_submission()
        except Exception:
            self.logger.error("Evaluation failed:", exc_info=True)
            raise
  
    def create_submission(self):
        """Create submission file from evaluation results"""
        try:
            output_dir = str(self.results_dir / "submissions")
            submission = {}
            
            self.logger.debug("\nStarting submission creation...")
            
            # Process each task's predictions
            for task_id, task_data in self.results['task_summaries'].items():
                self.logger.debug(f"\nProcessing task {task_id}:")
                prediction_obj = {}
                
                # Get predictions from evaluation mode (limit to 2 attempts)
                predictions = task_data.get('evaluation', {}).get('predictions', [])[:2]
                self.logger.debug(f"Found {len(predictions)} predictions for task")
                
                for idx, pred in enumerate(predictions, 1):
                    self.logger.debug(f"\nProcessing attempt {idx}:")
                    
                    # Get predicted grid
                    grid = pred.get('predicted_grid', [])
                    self.logger.debug(f"Original grid shape: {len(grid)}x{len(grid[0]) if grid else 0}")
                    if grid:
                        self.logger.debug("First row sample: " + str(grid[0][:5]) + "...")
                    
                    # Remove padding values (10) and get actual grid dimensions
                    cleaned_grid = []
                    max_cols = 0
                    
                    # First pass - determine actual grid dimensions
                    self.logger.debug("\nFirst pass - finding dimensions:")
                    for row_idx, row in enumerate(grid):
                        valid_row = []
                        for col_idx, val in enumerate(row):
                            if val == 10:  # Found padding
                                self.logger.debug(f"Found padding at position [{row_idx}, {col_idx}]")
                                break
                            valid_row.append(val)
                        if valid_row:  # Only track non-empty rows
                            max_cols = max(max_cols, len(valid_row))
                            cleaned_grid.append(valid_row)
                            self.logger.debug(f"Row {row_idx}: found {len(valid_row)} valid columns")
                    
                    self.logger.debug("After first pass:")
                    self.logger.debug(f"Max columns: {max_cols}")
                    self.logger.debug(f"Valid rows: {len(cleaned_grid)}")
                    
                    # Second pass - ensure rectangular grid
                    final_grid = []
                    if cleaned_grid:
                        self.logger.debug("\nSecond pass - creating rectangular grid:")
                        for row_idx, row in enumerate(cleaned_grid):
                            if row:  # Only add non-empty rows
                                final_grid.append(row[:max_cols])
                                self.logger.debug(f"Row {row_idx} length: {len(row[:max_cols])}")
                    
                    self.logger.debug(f"\nFinal grid dimensions: {len(final_grid)}x{len(final_grid[0]) if final_grid else 0}")
                    if final_grid:
                        self.logger.debug("Sample of final grid:")
                        for i in range(min(3, len(final_grid))):
                            self.logger.debug(f"Row {i}: {final_grid[i][:5]}...")
                    
                    # Add attempt if we have a valid grid
                    if final_grid:
                        prediction_obj[f'attempt_{idx}'] = final_grid
                        self.logger.debug(f"Added attempt_{idx} to prediction object")
                    else:
                        self.logger.debug(f"No valid grid generated for attempt {idx} of task {task_id}.")
                
                # Only add tasks that have predictions and wrap in list
                if prediction_obj:
                    submission[task_id] = [prediction_obj]
                    self.logger.debug(f"Added task {task_id} to submission")
                else:
                    self.logger.debug(f"No predictions to add for task {task_id}.")
            
            # Save submission
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            submission_file = Path(output_dir) / f'submission_{timestamp}.json'
            submission_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Debug final submission structure
            self.logger.debug("\nFinal submission structure:")
            self.logger.debug(f"Number of tasks: {len(submission)}")
            for task_id, task_preds in submission.items():
                self.logger.debug(f"\nTask {task_id}:")
                for pred_obj in task_preds:
                    for attempt_id, grid in pred_obj.items():
                        self.logger.debug(f"{attempt_id}: {len(grid)}x{len(grid[0]) if grid else 0} grid")
            
            with open(submission_file, 'w') as f:
                json.dump(submission, f)
                
            self.logger.info(f"Created submission file: {submission_file}")
            return submission_file
        except Exception as e:
            self.logger.error(f"Failed to create submission: {str(e)}", exc_info=True)
            raise


def main():
    """Main entry point with error handling"""
    try:
        cfg = Config()
        evaluator = EvaluationManager(cfg)
        evaluator.run_evaluation()
        
    except Exception:
        logging.error("Evaluation failed:", exc_info=True)
        raise

if __name__ == "__main__":
    main()
