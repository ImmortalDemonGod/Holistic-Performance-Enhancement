"""
commit_processor.py -- Enrich commit data with code quality metrics (Radon, Ruff)
"""
import os
import tempfile
import subprocess
import json
from pathlib import Path
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from typing import List, Dict

try:
    from git import Repo
except ImportError:
    raise ImportError("GitPython must be installed: pip install GitPython")

def analyze_commits_code_quality(repo_path: str, commits: List[Dict]) -> List[Dict]:
    """
    Enriches commit dictionaries with code quality metrics for changed Python files.
    
    For each commit, analyzes all changed `.py` files to compute total cyclomatic complexity, average maintainability index, total Ruff linter errors, and the count of changed Python files. Returns the list of commit dictionaries with these metrics added.
    """
    repo = Repo(repo_path)
    enriched_commits = []
    for commit in commits:
        sha = commit['sha']
        py_files = []
        total_cc = 0
        mi_scores = []
        total_ruff = 0
        files_changed = set()
        try:
            c = repo.commit(sha)
            for diff in c.stats.files:
                fname = diff
                if fname.endswith('.py'):
                    files_changed.add(fname)
        except Exception as e:
            commit['cc_error'] = str(e)
            enriched_commits.append(commit)
            continue
        for fname in files_changed:
            try:
                blob = repo.git.show(f'{sha}:{fname}')
            except Exception as e:
                continue
            # Cyclomatic Complexity
            try:
                cc_scores = cc_visit(blob)
                file_cc = sum([c.complexity for c in cc_scores])
            except Exception:
                file_cc = 0
            total_cc += file_cc
            # Maintainability Index
            try:
                mi = mi_visit(blob, True)
                mi_scores.append(mi)
            except Exception as e:
                print(f"[WARN] Could not compute MI for {fname} in commit {sha}: {e}")
            # Ruff errors
            try:
                with tempfile.NamedTemporaryFile('w', delete=False, suffix='.py') as tf:
                    tf.write(blob)
                    temp_path = tf.name
                result = subprocess.run(['ruff', temp_path, '--format', 'json'], capture_output=True, text=True)
                ruff_json = json.loads(result.stdout) if result.stdout else []
                total_ruff += len(ruff_json)
                os.unlink(temp_path)
            except Exception as e:
                print(f"[WARN] Could not run Ruff for {fname} in commit {sha}: {e}")
        commit['py_files_changed_count'] = len(files_changed)
        commit['total_cc'] = total_cc
        commit['avg_mi'] = sum(mi_scores) / len(mi_scores) if mi_scores else 0
        commit['ruff_errors'] = total_ruff
        enriched_commits.append(commit)
    return enriched_commits
