#!/usr/bin/env python3
"""
Simple script to view PyTorch Lightning training logs without TensorBoard
"""
import os
import json
from pathlib import Path

def read_lightning_logs(log_dir):
    """
    Reads and displays a summary of PyTorch Lightning training logs from a specified directory.
    
    For each versioned log directory, prints hyperparameters, metrics summary, TensorBoard event files, and available model checkpoints. Handles missing files and read errors gracefully.
    """
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"Log directory not found: {log_dir}")
        return
    
    print(f"📊 Training Logs from: {log_dir}")
    print("=" * 60)
    
    # Find all version directories
    version_dirs = [d for d in log_path.iterdir() if d.is_dir() and d.name.startswith('version_')]
    version_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
    
    for version_dir in version_dirs:
        print(f"\n🔍 {version_dir.name.upper()}")
        print("-" * 40)
        
        # Check for hparams.yaml
        hparams_file = version_dir / "hparams.yaml"
        if hparams_file.exists():
            print(f"📋 Hyperparameters: {hparams_file}")
            try:
                with open(hparams_file, 'r') as f:
                    content = f.read().strip()
                    if content and content != '{}':
                        print(f"   {content}")
                    else:
                        print("   No hyperparameters recorded")
            except Exception as e:
                print(f"   Error reading hparams: {e}")
        
        # Check for metrics.csv (if exists)
        metrics_file = version_dir / "metrics.csv"
        if metrics_file.exists():
            print(f"📈 Metrics file found: {metrics_file}")
            try:
                with open(metrics_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        print(f"   Headers: {lines[0].strip()}")
                        print(f"   Last entry: {lines[-1].strip()}")
                        print(f"   Total entries: {len(lines) - 1}")
            except Exception as e:
                print(f"   Error reading metrics: {e}")
        
        # List event files
        event_files = list(version_dir.glob("events.out.tfevents.*"))
        if event_files:
            print(f"📊 TensorBoard event files:")
            for event_file in event_files:
                size_kb = event_file.stat().st_size / 1024
                print(f"   {event_file.name} ({size_kb:.1f} KB)")
        
        # Check for checkpoints directory
        checkpoints_dir = version_dir / "checkpoints"
        if checkpoints_dir.exists():
            checkpoints = list(checkpoints_dir.glob("*.ckpt"))
            if checkpoints:
                print(f"💾 Model checkpoints ({len(checkpoints)} files):")
                for ckpt in checkpoints[:3]:  # Show first 3
                    size_mb = ckpt.stat().st_size / (1024 * 1024)
                    print(f"   {ckpt.name} ({size_mb:.1f} MB)")
                if len(checkpoints) > 3:
                    print(f"   ... and {len(checkpoints) - 3} more")

if __name__ == "__main__":
    log_dir = "cultivation/systems/arc_reactor/logs/training/lightning_logs"
    read_lightning_logs(log_dir)
