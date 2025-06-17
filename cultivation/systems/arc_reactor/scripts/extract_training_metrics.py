#!/usr/bin/env python3
"""
Extract and display training metrics from TensorBoard logs using tbparse
"""
import os
from pathlib import Path
import pandas as pd
from tbparse import SummaryReader
import matplotlib.pyplot as plt

def extract_and_display_metrics(log_dir):
    """Extract metrics from TensorBoard logs and display them"""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"❌ Log directory not found: {log_dir}")
        return
    
    print(f"📊 Extracting Training Metrics from: {log_dir}")
    print("=" * 70)
    
    # Find all version directories
    version_dirs = [d for d in log_path.iterdir() if d.is_dir() and d.name.startswith('version_')]
    version_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
    
    all_metrics = {}
    
    for version_dir in version_dirs:
        version_name = version_dir.name
        print(f"\n🔍 Processing {version_name.upper()}")
        print("-" * 50)
        
        try:
            # Use tbparse to read the logs
            reader = SummaryReader(str(version_dir))
            df = reader.scalars
            
            if df.empty:
                print("   No scalar metrics found")
                continue
            
            # Store metrics for this version
            all_metrics[version_name] = df
            
            # Display summary
            print(f"   📈 Found {len(df)} metric entries")
            print(f"   📋 Metrics tracked: {df['tag'].unique().tolist()}")
            print(f"   🕐 Steps range: {df['step'].min()} - {df['step'].max()}")
            
            # Show latest values for each metric
            print("   📊 Latest values:")
            for tag in df['tag'].unique():
                tag_data = df[df['tag'] == tag]
                latest_value = tag_data.iloc[-1]['value']
                latest_step = tag_data.iloc[-1]['step']
                print(f"      {tag}: {latest_value:.6f} (step {latest_step})")
            
        except Exception as e:
            print(f"   ❌ Error reading logs: {e}")
    
    # Create visualizations if we have data
    if all_metrics:
        print(f"\n📈 Creating visualizations...")
        create_plots(all_metrics, log_path.parent / "training_plots.png")
    
    return all_metrics

def create_plots(all_metrics, output_path):
    """Create plots for the training metrics"""
    try:
        # Determine number of subplots needed
        all_tags = set()
        for df in all_metrics.values():
            all_tags.update(df['tag'].unique())
        
        if not all_tags:
            print("   No metrics to plot")
            return
        
        # Create subplots
        n_metrics = len(all_tags)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each metric
        for i, tag in enumerate(sorted(all_tags)):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            for version_name, df in all_metrics.items():
                tag_data = df[df['tag'] == tag]
                if not tag_data.empty:
                    ax.plot(tag_data['step'], tag_data['value'], 
                           label=version_name, marker='o', markersize=3)
            
            ax.set_title(f'{tag}')
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   💾 Plots saved to: {output_path}")
        
        # Also save individual CSVs
        for version_name, df in all_metrics.items():
            csv_path = output_path.parent / f"metrics_{version_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"   💾 {version_name} metrics saved to: {csv_path}")
        
    except Exception as e:
        print(f"   ❌ Error creating plots: {e}")

if __name__ == "__main__":
    log_dir = "cultivation/systems/arc_reactor/logs/training/lightning_logs"
    metrics = extract_and_display_metrics(log_dir)
