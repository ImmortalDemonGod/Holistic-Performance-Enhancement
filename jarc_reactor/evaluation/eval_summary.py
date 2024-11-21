# jarc_reactor/utils/eval_summary.py

from datetime import datetime

class EvaluationSummary:
    def __init__(self, evaluator, all_results):
        self.evaluator = evaluator  # Reference to EvaluationManager for access to logger and config
        self.all_results = all_results

    def write(self, text, file_handle):
        print(text)  # Print to console
        file_handle.write(text + '\n')  # Write to file

    def write_header(self, file_handle):
        self.write("\n" + "="*80, file_handle)
        self.write("                        EVALUATION SUMMARY REPORT", file_handle)
        self.write("="*80 + "\n", file_handle)

    def write_footer(self, file_handle, summary_file):
        self.write("="*80, file_handle)
        self.write(f"Report saved to: {summary_file}", file_handle)
        self.write("="*80 + "\n", file_handle)

    def write_model_configuration(self, file_handle):
        self.write("MODEL CONFIGURATION:", file_handle)
        self.write(f"Checkpoint: {self.evaluator.config.model.checkpoint_path}", file_handle)
        self.write(f"Encoder layers: {self.evaluator.config.model.encoder_layers}", file_handle)
        self.write(f"Decoder layers: {self.evaluator.config.model.decoder_layers}", file_handle)
        self.write(f"Model dimension: {self.evaluator.config.model.d_model}", file_handle)
        self.write(f"Attention heads: {self.evaluator.config.model.heads}\n", file_handle)

    def write_overall_performance(self, file_handle):
        self.write("OVERALL PERFORMANCE:", file_handle)
        for mode, results in self.all_results.items():
            if 'overall_metrics' in results:
                metrics = results['overall_metrics']
                self.write(f"\n{mode.upper()}:", file_handle)
                self.write(f"  Standard Accuracy: {metrics['standard_accuracy']:.4f}", file_handle)
                self.write(f"  Differential Accuracy: {metrics['differential_accuracy']:.4f}", file_handle)
        self.write("", file_handle)

    def write_model_behavior_analysis(self, file_handle):
        self.write("MODEL BEHAVIOR ANALYSIS:", file_handle)
        for mode, results in self.all_results.items():
            if 'task_summaries' in results:
                analysis = self.analyze_model_behavior(mode, results)
                self.write(analysis, file_handle)
        self.write("", file_handle)

    def write_task_specific_analysis(self, file_handle):
        self.write("INTERESTING TASK PATTERNS:", file_handle)
        for mode, results in self.all_results.items():
            if 'task_summaries' in results:
                analysis = self.analyze_task_patterns(mode, results)
                self.write(analysis, file_handle)

    def write_key_findings(self, file_handle):
        self.write("KEY FINDINGS:", file_handle)
        findings = self.identify_key_findings()
        self.write(findings, file_handle)

    def write_training_recommendations(self, file_handle):
        self.write("\nTRAINING RECOMMENDATIONS:", file_handle)
        recommendations = self.generate_training_recommendations()
        self.write(recommendations, file_handle)

    def generate_summary(self):
        """Generate a comprehensive evaluation summary with key findings"""
        try:
            summary_file = self.evaluator.results_dir / f'evaluation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            
            with open(summary_file, 'w') as f:
                self.write_header(f)
                self.write_model_configuration(f)
                self.write_overall_performance(f)
                self.write_model_behavior_analysis(f)
                self.write_task_specific_analysis(f)
                self.write_key_findings(f)
                self.write_training_recommendations(f)
                self.write_footer(f, summary_file)
            
            return summary_file
        
        except Exception as e:
            self.evaluator.logger.error(f"Error generating summary: {str(e)}")
            raise

    def analyze_model_behavior(self, mode, results):
        """Analyze model behavior for a specific mode."""
        pred_values, target_values, total_tasks, perfect_tasks, failed_tasks = self._extract_model_behavior_metrics(results)
        
        analysis = f"\n{mode.upper()} ANALYSIS:"
        analysis += f"\n  Total tasks: {total_tasks}"
        analysis += f"\n  Perfect solutions: {perfect_tasks} ({(perfect_tasks/total_tasks)*100:.1f}%)"
        analysis += f"\n  Failed tasks: {failed_tasks} ({(failed_tasks/total_tasks)*100:.1f}%)"
        analysis += f"\n  Model predictions range: {sorted(pred_values)}"
        analysis += f"\n  Expected values range: {sorted(target_values)}"
        return analysis

    def _extract_model_behavior_metrics(self, results):
        """Extract metrics related to model behavior."""
        pred_values = set()
        target_values = set()
        total_tasks = len(results['task_summaries'])
        perfect_tasks = 0
        failed_tasks = 0
        
        for task_id, metrics in results['task_summaries'].items():
            if 'debug_info' in metrics:
                debug = metrics['debug_info']
                pred_values.update(debug.get('pred_unique', []))
                target_values.update(debug.get('target_unique', []))
            
            if metrics.get('standard_accuracy', 0) >= 0.999:
                perfect_tasks += 1
            elif metrics.get('standard_accuracy', 0) == 0:
                failed_tasks += 1
        
        return pred_values, target_values, total_tasks, perfect_tasks, failed_tasks

    def analyze_task_patterns(self, mode, results):
        """Analyze best and worst performing tasks."""
        tasks = [(task_id, metrics.get('standard_accuracy', 0)) 
                 for task_id, metrics in results['task_summaries'].items()]
        tasks.sort(key=lambda x: x[1], reverse=True)
        
        analysis = f"\n{mode.upper()}:"
        
        # Best tasks
        analysis += "\n\nBest performing tasks:"
        for task_id, acc in tasks[:3]:
            metrics = results['task_summaries'][task_id]
            analysis += f"\n  Task {task_id}:"
            analysis += f"\n    Standard Accuracy: {acc:.4f}"
            analysis += f"\n    Differential Accuracy: {metrics.get('differential_accuracy', 0):.4f}"
            if 'debug_info' in metrics:
                debug = metrics['debug_info']
                if 'pred_unique' in debug and 'target_unique' in debug:
                    analysis += f"\n    Predictions: {debug['pred_unique']}"
                    analysis += f"\n    Targets: {debug['target_unique']}"
        
        # Worst tasks
        analysis += "\n\nWorst performing tasks:"
        for task_id, acc in tasks[-3:]:
            metrics = results['task_summaries'][task_id]
            analysis += f"\n  Task {task_id}:"
            analysis += f"\n    Standard Accuracy: {acc:.4f}"
            analysis += f"\n    Differential Accuracy: {metrics.get('differential_accuracy', 0):.4f}"
            if 'debug_info' in metrics:
                debug = metrics['debug_info']
                if 'pred_unique' in debug and 'target_unique' in debug:
                    analysis += f"\n    Predictions: {debug['pred_unique']}"
                    analysis += f"\n    Targets: {debug['target_unique']}"
        
        return analysis

    def identify_key_findings(self):
        """Identify key findings from all results."""
        all_preds = set()
        all_targets = set()
        for results in self.all_results.values():
            for metrics in results.get('task_summaries', {}).values():
                if 'debug_info' in metrics:
                    debug = metrics['debug_info']
                    all_preds.update(debug.get('pred_unique', []))
                    all_targets.update(debug.get('target_unique', []))
        
        findings = ""
        findings += "\n1. Model Prediction Range:"
        findings += f"\n   - Model predicts values in range: {sorted(all_preds)}"
        findings += f"\n   - Expected value range: {sorted(all_targets)}"
        if len(all_preds) < len(all_targets):
            findings += "\n   ! Model is not using full output range"
        
        # Calculate prediction bias
        pred_mean = sum(all_preds) / len(all_preds) if all_preds else 0
        target_mean = sum(all_targets) / len(all_targets) if all_targets else 0
        if abs(pred_mean - target_mean) > 1:
            findings += "\n\n2. Prediction Bias:"
            findings += f"\n   - Average prediction: {pred_mean:.2f}"
            findings += f"\n   - Average target: {target_mean:.2f}"
            findings += "\n   ! Model shows significant bias in predictions"
        
        # Performance Pattern
        total_perfect = sum(
            1 for results in self.all_results.values()
            for metrics in results.get('task_summaries', {}).values()
            if metrics.get('standard_accuracy', 0) >= 0.999
        )
        total_tasks = sum(
            len(results.get('task_summaries', {})) 
            for results in self.all_results.values()
        )
        findings += "\n\n3. Performance Pattern:"
        findings += f"\n   - Perfect solutions: {total_perfect}/{total_tasks} tasks"
        findings += f"\n   - Success rate: {(total_perfect/total_tasks)*100:.1f}%"
        
        return findings

    def generate_training_recommendations(self):
        """Generate training recommendations based on key findings."""
        recommendations = ""
        # Reuse all_preds and all_targets from key findings
        all_preds = set()
        all_targets = set()
        for results in self.all_results.values():
            for metrics in results.get('task_summaries', {}).values():
                if 'debug_info' in metrics:
                    debug = metrics['debug_info']
                    all_preds.update(debug.get('pred_unique', []))
                    all_targets.update(debug.get('target_unique', []))
        
        if len(all_preds) < len(all_targets):
            recommendations += "1. Model needs better output distribution - consider:"
            recommendations += "\n   - Longer training time"
            recommendations += "\n   - Adjusting loss function to encourage full output range"
            recommendations += "\n   - Checking output layer configuration"
        
        # Calculate success rate
        total_perfect = sum(
            1 for results in self.all_results.values()
            for metrics in results.get('task_summaries', {}).values()
            if metrics.get('standard_accuracy', 0) >= 0.999
        )
        total_tasks = sum(
            len(results.get('task_summaries', {})) 
            for results in self.all_results.values()
        )
        if total_tasks > 0 and (total_perfect / total_tasks) < 0.5:
            recommendations += "\n\n2. Low success rate suggests:"
            recommendations += "\n   - Model may need more capacity (layers/dimensions)"
            recommendations += "\n   - Training data might be insufficient"
            recommendations += "\n   - Consider curriculum learning approach"
        
        return recommendations