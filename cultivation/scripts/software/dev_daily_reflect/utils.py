"""
utils.py -- Path and repo utilities for DevDailyReflect
"""
import pathlib

import os

def get_repo_root(start_path=None):
    """
    Returns the root directory of the Git repository.
    
    If the CULTIVATION_REPO_ROOT_OVERRIDE environment variable is set, its resolved path is returned. Otherwise, ascends from the given start_path (or the current file's location) until a directory containing a .git folder is found. Raises RuntimeError if no such directory exists.

    """Ascend from start_path (or this file) until a .git directory is found, or use CULTIVATION_REPO_ROOT_OVERRIDE env var if set."""
    repo_env = os.environ.get("CULTIVATION_REPO_ROOT_OVERRIDE")

    if repo_env:
        return pathlib.Path(repo_env).resolve()
    p = pathlib.Path(start_path or __file__).resolve()
    while not (p / '.git').exists() and p != p.parent:
        p = p.parent
    if (p / '.git').exists():
        return p
    raise RuntimeError('Could not find .git directory in any parent')
