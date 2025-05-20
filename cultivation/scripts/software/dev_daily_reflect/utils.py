"""
utils.py -- Path and repo utilities for DevDailyReflect
"""
import pathlib

import os

def get_repo_root(start_path=None):
    """
    Returns the root directory of the Git repository.
    
    If the DEV_DAILY_REFLECT_REPO_ROOT environment variable is set, its resolved path is returned. Otherwise, ascends from the given start_path (or the current file's location) until a directory containing a .git folder is found. Raises RuntimeError if no such directory exists.
    """
    repo_env = os.environ.get("DEV_DAILY_REFLECT_REPO_ROOT")
    if repo_env:
        return pathlib.Path(repo_env).resolve()
    p = pathlib.Path(start_path or __file__).resolve()
    while not (p / '.git').exists() and p != p.parent:
        p = p.parent
    if (p / '.git').exists():
        return p
    raise RuntimeError('Could not find .git directory in any parent')
