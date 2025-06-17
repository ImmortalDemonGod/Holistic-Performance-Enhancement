#!/bin/bash

# Quick TensorBoard launcher script
echo "🚀 Starting TensorBoard..."

# Check if tensorboard environment exists
if [ ! -d ".tensorboard_env" ]; then
    echo "❌ TensorBoard environment not found. Run ./setup_tensorboard_env.sh first"
    exit 1
fi

# Activate environment and run TensorBoard
source .tensorboard_env/bin/activate
echo "📊 Launching TensorBoard on http://localhost:6006"
echo "Press Ctrl+C to stop"
tensorboard --logdir=cultivation/systems/arc_reactor/logs/training/lightning_logs --port=6006 --host=127.0.0.1
