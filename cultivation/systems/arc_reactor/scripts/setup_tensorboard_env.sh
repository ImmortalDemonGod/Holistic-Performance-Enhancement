#!/bin/bash

# Setup TensorBoard Environment Script
echo "🔧 Setting up dedicated TensorBoard environment..."

# Check if we have Python 3.11 or 3.12 available
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo "✅ Found Python 3.11"
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    echo "✅ Found Python 3.12"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo "✅ Found Python 3.10"
else
    echo "❌ No compatible Python version found. Installing Python 3.11..."
    brew install python@3.11
    PYTHON_CMD="python3.11"
fi

# Create virtual environment
echo "📦 Creating virtual environment with $PYTHON_CMD..."
$PYTHON_CMD -m venv .tensorboard_env

# Activate and install TensorBoard
echo "🚀 Installing TensorBoard..."
source .tensorboard_env/bin/activate
pip install --upgrade pip
pip install tensorboard tensorflow

echo "✅ TensorBoard environment setup complete!"
echo ""
echo "To use TensorBoard:"
echo "1. Activate the environment: source .tensorboard_env/bin/activate"
echo "2. Run TensorBoard: tensorboard --logdir=cultivation/systems/arc_reactor/logs/training/lightning_logs --port=6006"
echo "3. Open browser: http://localhost:6006"
echo ""
echo "To deactivate: deactivate"
