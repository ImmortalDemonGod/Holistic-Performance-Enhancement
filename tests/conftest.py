"""
Configure pytest to add the project root to the Python path.
This allows tests to import modules from the project without installing it.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
