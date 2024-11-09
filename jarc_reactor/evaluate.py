# evaluate.py
import logging
import os
import json
from datetime import datetime
from pathlib import Path
import torch
import math
from tqdm import tqdm
from jarc_reactor.config import Config
from jarc_reactor.utils.model_factory import create_transformer_trainer
from jarc_reactor.data.data_preparation import prepare_data as prepare_training_data
from jarc_reactor.data.eval_data_prep import prepare_data as prepare_eval_data
from jarc_reactor.utils.metrics import (
    compute_standard_accuracy,
    compute_differential_accuracy,
    TaskMetricsCollector,
    PredictionRecord  # Add this import
)



class EvaluationManager:
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        
        # Initialize result storage
        self.results_dir = Path('evaluation_results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize metrics collector for each mode
        self.metrics_collectors = {
            'training_train': TaskMetricsCollector(),
            'training_validation': TaskMetricsCollector(),
            'evaluation': TaskMetricsCollector()
        }
        
        # Initialize flags for available data
        self.has_training_data = False
        self.has_eval_data = False

    def setup_logging(self):
        """Setup detailed logging configuration"""
        log_dir = Path('evaluation_logs')
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'evaluation_{timestamp}.log'
        
        self.logger = logging.getLogger('evaluation')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler for debugging
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(fh)
        
        # Console handler for info
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(ch)

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

    def _load_task_maps(self):
        """Load or create task ID mappings"""
        # Try loading training task map
        try:
            self.logger.debug("Loading task_id_map.json...")
            with open("task_id_map.json", "r") as f:
                self.train_task_map = json.load(f)
            self.train_int_to_task = {v: k for k, v in self.train_task_map.items()}
            self.has_training_data = True
            self.logger.info(f"Loaded {len(self.train_task_map)} training tasks")
        except FileNotFoundError:
            self.logger.warning("task_id_map.json not found - skipping training data evaluation")
            self.has_training_data = False
        except Exception as e:
            self.logger.error(f"Error loading training task map: {str(e)}")
            self.has_training_data = False

        # Try loading or creating evaluation task map
        try:
            self.logger.debug("Loading eval_id_map.json...")
            try:
                with open("eval_id_map.json", "r") as f:
                    self.eval_task_map = json.load(f)
                self.logger.info(f"Loaded {len(self.eval_task_map)} evaluation tasks")
            except FileNotFoundError:
                self.logger.info("eval_id_map.json not found - creating from evaluation data...")
                
                # Load evaluation data to get task IDs
                try:
                    _, eval_dataset = prepare_eval_data(return_datasets=True)
                    
                    # Extract unique task IDs
                    unique_task_ids = set()
                    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1)
                    for _, _, _, _, task_ids in eval_loader:
                        task_id = task_ids[0].item()
                        unique_task_ids.add(str(task_id))
                    
                    # Create task map
                    self.eval_task_map = {
                        task_id: idx 
                        for idx, task_id in enumerate(sorted(unique_task_ids))
                    }
                    
                    # Save the map
                    with open("eval_id_map.json", "w") as f:
                        json.dump(self.eval_task_map, f, indent=2)
                    self.logger.info(
                        f"Created and saved eval_id_map.json with {len(self.eval_task_map)} tasks"
                    )
                except Exception as e:
                    self.logger.error(f"Error creating eval_id_map.json: {str(e)}")
                    self.has_eval_data = False
                    raise
                    
            self.eval_int_to_task = {v: k for k, v in self.eval_task_map.items()}
            self.has_eval_data = True
            
        except Exception as e:
            self.logger.error(f"Error handling evaluation task map: {str(e)}")
            self.has_eval_data = False

        if not (self.has_training_data or self.has_eval_data):
            raise ValueError("No task maps found or created - cannot perform any evaluation")

    def _validate_shapes(self, src, tgt, ctx_input, ctx_output, outputs, mode):
        """Validate tensor shapes and log details"""
        self.logger.debug(f"\nShape validation for {mode}:")
        self.logger.debug(f"src: {src.shape}")
        self.logger.debug(f"tgt: {tgt.shape}")
        self.logger.debug(f"ctx_input: {ctx_input.shape}")
        self.logger.debug(f"ctx_output: {ctx_output.shape}")
        self.logger.debug(f"model outputs: {outputs.shape}")

    def _process_batch(self, batch, mode, int_to_task):
        """Process a single batch with detailed logging and prediction storage"""
        try:
            src, tgt, ctx_input, ctx_output, task_ids = [
                t.to(self.device) for t in batch
            ]
            
            # Log initial tensor info
            self.logger.debug("\nInitial tensor information:")
            self.logger.debug(f"src type: {src.dtype}, range: [{src.min():.1f}, {src.max():.1f}]")
            self.logger.debug(f"tgt type: {tgt.dtype}, range: [{tgt.min():.1f}, {tgt.max():.1f}]")
            self.logger.debug(f"src shape: {src.shape}")
            self.logger.debug(f"tgt shape: {tgt.shape}")
            
            with torch.no_grad():
                outputs = self.model(src, tgt, ctx_input, ctx_output)
            
            # Log raw model outputs
            self.logger.debug(f"\nModel outputs:")
            self.logger.debug(f"shape: {outputs.shape}")
            self.logger.debug(f"type: {outputs.dtype}")
            self.logger.debug(f"range: [{outputs.min():.1f}, {outputs.max():.1f}]")
            
            # Analyze logits distribution
            self.logger.debug("\nLogits Distribution:")
            logits_flat = outputs.view(-1, outputs.size(-1))
            for i in range(outputs.size(-1)):  # For each class
                class_logits = logits_flat[:, i]
                self.logger.debug(
                    f"Class {i} logits - "
                    f"mean: {class_logits.mean():.3f}, "
                    f"std: {class_logits.std():.3f}, "
                    f"range: [{class_logits.min():.3f}, {class_logits.max():.3f}]"
                )
            
            # Get predictions
            predictions = outputs.argmax(dim=-1)  # Shape: [batch, seq_len]
            
            # Process each example
            for idx, task_id_int in enumerate(task_ids):
                task_id = int_to_task[task_id_int.item()]
                
                # Get individual tensors - FIXED: Maintain target dimensions
                task_input = src[idx:idx + 1]  # [1, 30, 30]
                task_target = tgt[idx:idx + 1]  # [1, 30, 30] - Keeping all dimensions
                task_pred = predictions[idx:idx + 1]  # [1, 30, 30]
                task_raw_outputs = outputs[idx]  # [30, 30, 11]
                
                # Debug task tensor shapes
                self.logger.debug(f"\nTask tensor shapes:")
                self.logger.debug(f"task_input: {task_input.shape}")
                self.logger.debug(f"task_target: {task_target.shape}")
                self.logger.debug(f"task_pred: {task_pred.shape}")
                self.logger.debug(f"task_raw_outputs: {task_raw_outputs.shape}")
                
                # Convert types for consistency
                task_target = task_target.to(torch.long)
                task_pred = task_pred.to(torch.long)
                
                # Analyze output distribution for this task
                analysis = self.analyze_outputs_distribution(
                    task_raw_outputs,
                    task_pred,
                    task_target,
                    task_id
                )
                
                # Calculate probabilities and confidence
                probs = torch.softmax(task_raw_outputs, dim=-1)
                confidence, _ = probs.max(dim=-1)
                
                # Create prediction record with enhanced information
                pred_record = PredictionRecord(
                    input_grid=task_input.squeeze(0).cpu().tolist(),
                    target_grid=task_target.squeeze(0).cpu().tolist(),
                    predicted_grid=task_pred.squeeze(0).cpu().tolist(),
                    raw_logits=task_raw_outputs.cpu().tolist(),
                    position_metrics={
                        'output_probabilities': probs.cpu().tolist(),
                        'output_classes': task_pred.squeeze(0).cpu().tolist(),
                        'output_confidences': confidence.cpu().tolist(),
                        'distribution_analysis': analysis
                    }
                )
                
                # Calculate metrics
                std_acc, std_details = compute_standard_accuracy(
                    task_pred.flatten(),
                    task_target.flatten(),
                )
                
                diff_acc, diff_details = compute_differential_accuracy(
                    task_input.squeeze(0),
                    task_target.squeeze(0),
                    task_pred.squeeze(0),
                )
                
                # Enhanced metrics dictionary
                metrics_dict = {
                    'standard_accuracy': std_acc,
                    'differential_accuracy': diff_acc,
                    'std_details': std_details,
                    'diff_details': diff_details,
                    'debug_info': {
                        'target_shape': list(task_target.shape),
                        'pred_shape': list(task_pred.shape),
                        'target_unique': torch.unique(task_target).tolist(),
                        'pred_unique': torch.unique(task_pred).tolist(),
                        'output_range': [float(outputs[idx].min()), float(outputs[idx].max())],
                        'logits_stats': {
                            'mean': float(task_raw_outputs.mean()),
                            'std': float(task_raw_outputs.std()),
                            'min': float(task_raw_outputs.min()),
                            'max': float(task_raw_outputs.max())
                        },
                        'confidence_stats': {
                            'mean': float(confidence.mean()),
                            'std': float(confidence.std()),
                            'min': float(confidence.min()),
                            'max': float(confidence.max())
                        },
                        'distribution_analysis': analysis
                    }
                }
                
                # Store results
                self.metrics_collectors[mode].add_result(
                    task_id,
                    metrics_dict,
                    prediction_record=pred_record
                )
                
        except Exception as e:
            self.logger.error(f"Error processing batch: {str(e)}", exc_info=True)
            self.logger.error("Tensor shapes at failure:")
            self.logger.error(f"src: {src.shape if 'src' in locals() else 'not created'}")
            self.logger.error(f"tgt: {tgt.shape if 'tgt' in locals() else 'not created'}")
            self.logger.error(f"outputs: {outputs.shape if 'outputs' in locals() else 'not created'}")
            self.logger.error(f"predictions: {predictions.shape if 'predictions' in locals() else 'not created'}")
            raise

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
                
                # **Add 'debug_info' to the task summaries**
                results['task_summaries'][task_id] = {
                    'standard_accuracy': std_acc,
                    'differential_accuracy': diff_acc,
                    'predictions': metrics.get('predictions', []),
                    'debug_info': metrics.get('debug_info', {})  # <--- Added line
                }
            
            # Calculate averages
            if total_tasks > 0:
                results['overall_metrics'] = {
                    'standard_accuracy': overall_std_acc / total_tasks,
                    'differential_accuracy': overall_diff_acc / total_tasks
                }
            else:
                results['overall_metrics'] = {
                    'standard_accuracy': 0.0,
                    'differential_accuracy': 0.0
                }
            
            # Save results
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
        """Evaluate all batches in a loader"""
        self.logger.info(f"\nStarting evaluation for {mode}...")
        try:
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Evaluating {mode}")):
                self._process_batch(batch, mode, int_to_task)
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
            summary_file = self.results_dir / f'evaluation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            
            with open(summary_file, 'w') as f:
                def write(text):
                    print(text)  # Print to console
                    f.write(text + '\n')  # Write to file
                
                write("\n" + "="*80)
                write("                        EVALUATION SUMMARY REPORT")
                write("="*80 + "\n")
                
                # Model Information
                write("MODEL CONFIGURATION:")
                write(f"Checkpoint: {self.config.model.checkpoint_path}")
                write(f"Encoder layers: {self.config.model.encoder_layers}")
                write(f"Decoder layers: {self.config.model.decoder_layers}")
                write(f"Model dimension: {self.config.model.d_model}")
                write(f"Attention heads: {self.config.model.heads}\n")
                
                # Overall Performance
                write("OVERALL PERFORMANCE:")
                for mode, results in all_results.items():
                    if 'overall_metrics' in results:
                        metrics = results['overall_metrics']
                        write(f"\n{mode.upper()}:")
                        write(f"  Standard Accuracy: {metrics['standard_accuracy']:.4f}")
                        write(f"  Differential Accuracy: {metrics['differential_accuracy']:.4f}")
                write("")
                
                # Model Behavior Analysis
                write("MODEL BEHAVIOR ANALYSIS:")
                for mode, results in all_results.items():
                    if 'task_summaries' in results:
                        pred_values = set()
                        target_values = set()
                        total_tasks = len(results['task_summaries'])
                        perfect_tasks = 0
                        failed_tasks = 0  # Tasks with 0 accuracy
                        
                        for task_id, metrics in results['task_summaries'].items():
                            if 'debug_info' in metrics:
                                debug = metrics['debug_info']
                                if 'pred_unique' in debug:
                                    pred_values.update(debug['pred_unique'])
                                if 'target_unique' in debug:
                                    target_values.update(debug['target_unique'])
                            
                            # Count perfect and failed tasks
                            if metrics.get('standard_accuracy', 0) >= 0.999:
                                perfect_tasks += 1
                            elif metrics.get('standard_accuracy', 0) == 0:
                                failed_tasks += 1
                        
                        write(f"\n{mode.upper()} ANALYSIS:")
                        write(f"  Total tasks: {total_tasks}")
                        write(f"  Perfect solutions: {perfect_tasks} ({perfect_tasks/total_tasks*100:.1f}%)")
                        write(f"  Failed tasks: {failed_tasks} ({failed_tasks/total_tasks*100:.1f}%)")
                        write(f"  Model predictions range: {sorted(pred_values)}")
                        write(f"  Expected values range: {sorted(target_values)}")
                write("")
                
                # Task-Specific Analysis
                write("INTERESTING TASK PATTERNS:")
                for mode, results in all_results.items():
                    if 'task_summaries' in results:
                        write(f"\n{mode.upper()}:")
                        
                        # Find best and worst performing tasks
                        tasks = [(task_id, metrics.get('standard_accuracy', 0)) 
                                for task_id, metrics in results['task_summaries'].items()]
                        tasks.sort(key=lambda x: x[1], reverse=True)
                        
                        # Best tasks
                        write("\nBest performing tasks:")
                        for task_id, acc in tasks[:3]:
                            metrics = results['task_summaries'][task_id]
                            write(f"  Task {task_id}:")
                            write(f"    Standard Accuracy: {acc:.4f}")
                            write(f"    Differential Accuracy: {metrics.get('differential_accuracy', 0):.4f}")
                            if 'debug_info' in metrics:
                                debug = metrics['debug_info']
                                if 'pred_unique' in debug and 'target_unique' in debug:
                                    write(f"    Predictions: {debug['pred_unique']}")
                                    write(f"    Targets: {debug['target_unique']}")
                        
                        # Worst tasks
                        write("\nWorst performing tasks:")
                        for task_id, acc in tasks[-3:]:
                            metrics = results['task_summaries'][task_id]
                            write(f"  Task {task_id}:")
                            write(f"    Standard Accuracy: {acc:.4f}")
                            write(f"    Differential Accuracy: {metrics.get('differential_accuracy', 0):.4f}")
                            if 'debug_info' in metrics:
                                debug = metrics['debug_info']
                                if 'pred_unique' in debug and 'target_unique' in debug:
                                    write(f"    Predictions: {debug['pred_unique']}")
                                    write(f"    Targets: {debug['target_unique']}")
                
                # Key Findings
                write("\nKEY FINDINGS:")
                # Check if model predictions are clustered
                all_preds = set()
                all_targets = set()
                for results in all_results.values():
                    for metrics in results.get('task_summaries', {}).values():
                        if 'debug_info' in metrics:
                            debug = metrics['debug_info']
                            if 'pred_unique' in debug:
                                all_preds.update(debug['pred_unique'])
                            if 'target_unique' in debug:
                                all_targets.update(debug['target_unique'])
                
                write(f"\n1. Model Prediction Range:")
                write(f"   - Model predicts values in range: {sorted(all_preds)}")
                write(f"   - Expected value range: {sorted(all_targets)}")
                if len(all_preds) < len(all_targets):
                    write("   ! Model is not using full output range")
                
                # Calculate prediction bias
                pred_mean = sum(all_preds) / len(all_preds) if all_preds else 0
                target_mean = sum(all_targets) / len(all_targets) if all_targets else 0
                if abs(pred_mean - target_mean) > 1:
                    write(f"\n2. Prediction Bias:")
                    write(f"   - Average prediction: {pred_mean:.2f}")
                    write(f"   - Average target: {target_mean:.2f}")
                    write("   ! Model shows significant bias in predictions")
                
                write("\n3. Performance Pattern:")
                total_perfect = sum(1 for results in all_results.values()
                                for metrics in results.get('task_summaries', {}).values()
                                if metrics.get('standard_accuracy', 0) >= 0.999)
                total_tasks = sum(len(results.get('task_summaries', {})) 
                                for results in all_results.values())
                
                write(f"   - Perfect solutions: {total_perfect}/{total_tasks} tasks")
                write(f"   - Success rate: {total_perfect/total_tasks*100:.1f}%")
                
                # Training Recommendations
                write("\nTRAINING RECOMMENDATIONS:")
                if len(all_preds) < len(all_targets):
                    write("1. Model needs better output distribution - consider:")
                    write("   - Longer training time")
                    write("   - Adjusting loss function to encourage full output range")
                    write("   - Checking output layer configuration")
                
                if total_perfect/total_tasks < 0.5:
                    write("\n2. Low success rate suggests:")
                    write("   - Model may need more capacity (layers/dimensions)")
                    write("   - Training data might be insufficient")
                    write("   - Consider curriculum learning approach")
                
                write("\n" + "="*80)
                write(f"Report saved to: {summary_file}")
                write("="*80 + "\n")
                
            return summary_file
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            raise

    def run_evaluation(self):
        """Main evaluation loop with error handling"""
        try:
            self._load_task_maps()
            all_results = {}
            
            # Training data evaluation
            if self.has_training_data:
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
                        train_loader, 'training_train', self.train_int_to_task
                    )
                    all_results['training_train'] = self._process_results(
                        'training_train', 
                        self.metrics_collectors['training_train']
                    )
                    
                    self.evaluate_loader(
                        val_loader, 'training_validation', self.train_int_to_task
                    )
                    all_results['training_validation'] = self._process_results(
                        'training_validation',
                        self.metrics_collectors['training_validation']
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating training data: {str(e)}")
            
            # Evaluation data
            if self.has_eval_data:
                self.logger.info("\nPreparing evaluation data...")
                try:
                    _, eval_dataset = prepare_eval_data(return_datasets=True)
                    eval_loader = torch.utils.data.DataLoader(
                        eval_dataset, batch_size=1, shuffle=False
                    )
                    
                    self.evaluate_loader(
                        eval_loader, 'evaluation', self.eval_int_to_task
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
            
        except Exception as e:
            self.logger.error("Evaluation failed:", exc_info=True)
            raise
    
    def analyze_outputs_distribution(self, outputs, predictions, targets, task_id):
        """
        Analyze distribution of outputs, predictions, and targets
        
        Args:
            outputs: [batch, H, W, num_classes] or [H, W, num_classes]
            predictions: [batch, H, W] or [H, W]
            targets: [batch, H, W] or [H, W]
            task_id: Task identifier
        """
        with torch.no_grad():
            # Debug original shapes
            self.logger.debug(f"\nOriginal shapes before processing:")
            self.logger.debug(f"outputs: {outputs.shape}")
            self.logger.debug(f"predictions: {predictions.shape}")
            self.logger.debug(f"targets: {targets.shape}")
            
            # Handle batch dimension properly for all tensors
            if outputs.dim() == 4:
                outputs = outputs[0]  # [H, W, num_classes]
            if predictions.dim() == 3:
                predictions = predictions[0]  # [H, W]
                
            # Special handling for targets to ensure grid structure
            if targets.dim() == 3:
                targets = targets[0]  # Remove batch dimension if present
            if targets.dim() == 1:
                # Reshape to match grid structure using known dimensions
                H = W = int(math.sqrt(targets.size(0)))  # Should be 30
                targets = targets.view(H, W)
                
            # Extra validation
            if targets.dim() != 2:
                raise ValueError(f"Expected 2D targets after processing, got {targets.dim()}D with shape {targets.shape}")
                
            # Verify shapes after processing
            self.logger.debug(f"\nShapes after processing:")
            self.logger.debug(f"outputs: {outputs.shape}")
            self.logger.debug(f"predictions: {predictions.shape}")
            self.logger.debug(f"targets: {targets.shape}")
            
            # Assert correct dimensions
            assert outputs.dim() == 3, f"Expected outputs to be 3D after processing, got {outputs.dim()}D"
            assert predictions.dim() == 2, f"Expected predictions to be 2D after processing, got {predictions.dim()}D"
            assert targets.dim() == 2, f"Expected targets to be 2D after processing, got {targets.dim()}D"
            assert predictions.shape == targets.shape, (
                f"Predictions shape {predictions.shape} doesn't match targets shape {targets.shape}. "
                f"Original target shape was {targets.shape}"
            )

def main():
    """Main entry point with error handling"""
    try:
        cfg = Config()
        evaluator = EvaluationManager(cfg)
        evaluator.run_evaluation()
        
    except Exception as e:
        logging.error("Evaluation failed:", exc_info=True)
        raise

if __name__ == "__main__":
    main()
