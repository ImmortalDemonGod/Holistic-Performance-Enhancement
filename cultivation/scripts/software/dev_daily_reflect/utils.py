"""
utils.py -- Path and repo utilities for DevDailyReflect
"""
import pathlib

def get_repo_root(start_path=None):
    """Ascend from start_path (or this file) until a .git directory is found."""
    p = pathlib.Path(start_path or __file__).resolve()
    while not (p / '.git').exists() and p != p.parent:
        p = p.parent
    if (p / '.git').exists():
        return p
    raise RuntimeError('Could not find .git directory in any parent')
