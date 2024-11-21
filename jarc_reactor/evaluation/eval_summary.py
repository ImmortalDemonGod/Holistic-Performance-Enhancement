# jarc_reactor/utils/eval_summary.py

class SummaryWriter:
    def __init__(self, file_handle, evaluator):
        self.file = file_handle
        self.evaluator = evaluator  # Reference to EvaluationManager for access to logger and config
    
    def write(self, text):
        print(text)  # Print to console
        self.file.write(text + '\n')  # Write to file
    
    def write_header(self):
        self.write("\n" + "="*80)
        self.write("                        EVALUATION SUMMARY REPORT")
        self.write("="*80 + "\n")
    
    def write_footer(self, summary_file):
        self.write("="*80)
        self.write(f"Report saved to: {summary_file}")
        self.write("="*80 + "\n")
    
    def write_model_configuration(self):
        self.write("MODEL CONFIGURATION:")
        self.write(f"Checkpoint: {self.evaluator.config.model.checkpoint_path}")
        self.write(f"Encoder layers: {self.evaluator.config.model.encoder_layers}")
        self.write(f"Decoder layers: {self.evaluator.config.model.decoder_layers}")
        self.write(f"Model dimension: {self.evaluator.config.model.d_model}")
        self.write(f"Attention heads: {self.evaluator.config.model.heads}\n")
    
    def write_overall_performance(self, all_results):
        self.write("OVERALL PERFORMANCE:")
        for mode, results in all_results.items():
            if 'overall_metrics' in results:
                metrics = results['overall_metrics']
                self.write(f"\n{mode.upper()}:")
                self.write(f"  Standard Accuracy: {metrics['standard_accuracy']:.4f}")
                self.write(f"  Differential Accuracy: {metrics['differential_accuracy']:.4f}")
        self.write("")
    
    def write_model_behavior_analysis(self, all_results):
        self.write("MODEL BEHAVIOR ANALYSIS:")
        for mode, results in all_results.items():
            if 'task_summaries' in results:
                analysis = self.evaluator.analyze_model_behavior(mode, results)
                self.write(analysis)
        self.write("")
    
    def write_task_specific_analysis(self, all_results):
        self.write("INTERESTING TASK PATTERNS:")
        for mode, results in all_results.items():
            if 'task_summaries' in results:
                analysis = self.evaluator.analyze_task_patterns(mode, results)
                self.write(analysis)
    
    def write_key_findings(self, all_results):
        self.write("KEY FINDINGS:")
        findings = self.evaluator.identify_key_findings(all_results)
        self.write(findings)
    
    def write_training_recommendations(self, all_results):
        self.write("\nTRAINING RECOMMENDATIONS:")
        recommendations = self.evaluator.generate_training_recommendations(all_results)
        self.write(recommendations)


