import json
import pathlib
import tempfile
import sys
import pytest

# Ensure cultivation is on sys.path for import
sys.path.insert(0, str(pathlib.Path(__file__).parents[0]))
from cultivation.scripts.software.dev_daily_reflect.metrics import commit_processor

def test_analyze_commits_code_quality_handles_basic_commit(monkeypatch):
    # Setup: create a fake commit list with a single commit and a .py file
    fake_commits = [
        {
            'sha': 'deadbeef',
            'author': 'testuser',
            'timestamp': '2025-05-16 12:34:56 +0000',
            'message': 'Initial commit',
            'files': ['foo.py'],
            'py_files_changed_count': 1,
            'added': 10,
            'deleted': 2
        }
    ]

    # Patch GitPython Repo object to return a fake blob
    class FakeBlob:
        def data_stream(self):
            import io
            return io.BytesIO(b'def foo():\n    return 42\n')

    class FakeTree:
        def __getitem__(self, name):
            return FakeBlob()
    class FakeCommit:
        tree = FakeTree()
        class stats:
            files = {'foo.py': {}}
    class FakeGit:
        def show(self, ref):
            return 'def foo():\n    return 42\n'
    class FakeRepo:
        def __init__(self, *a, **k):
            self.git = FakeGit()
        def commit(self, sha):
            return FakeCommit()
    monkeypatch.setattr(commit_processor, 'Repo', FakeRepo)

    # Patch subprocess to simulate Ruff output
    def fake_run(cmd, input=None, text=None, capture_output=None):
        class Result:
            stdout = '[]'  # no errors
        return Result()
    monkeypatch.setattr(commit_processor.subprocess, 'run', fake_run)

    # Patch radon metrics
    monkeypatch.setattr(commit_processor, 'cc_visit', lambda code: [])
    monkeypatch.setattr(commit_processor, 'mi_visit', lambda code, _: 80.0)

    # Run
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        enriched = commit_processor.analyze_commits_code_quality(tmpdir, fake_commits)
        assert enriched[0]['total_cc'] == 0
        assert enriched[0]['avg_mi'] == 80.0
        assert enriched[0]['ruff_errors'] == 0
        assert enriched[0]['py_files_changed_count'] == 1
        assert enriched[0]['author'] == 'testuser'


def test_analyze_commits_code_quality_no_py_files(monkeypatch):
    """Edge case: commit contains no .py files; metrics should be zero, no crash."""
    fake_commits = [
        {
            'sha': 'deadbeef',
            'author': 'testuser',
            'timestamp': '2025-05-16 12:34:56 +0000',
            'message': 'No python here',
            'files': ['README.md'],
            'py_files_changed_count': 0,
            'added': 5,
            'deleted': 1
        }
    ]

    class FakeCommit:
        class stats:
            files = {'README.md': {}}
    class FakeGit:
        def show(self, ref):
            return '# readme'
    class FakeRepo:
        def __init__(self, *a, **k):
            self.git = FakeGit()
        def commit(self, sha):
            return FakeCommit()
    monkeypatch.setattr(commit_processor, 'Repo', FakeRepo)
    monkeypatch.setattr(commit_processor.subprocess, 'run', lambda *a, **k: type('Result', (), {'stdout': '[]'})())
    monkeypatch.setattr(commit_processor, 'cc_visit', lambda code: [])
    monkeypatch.setattr(commit_processor, 'mi_visit', lambda code, _: 80.0)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        enriched = commit_processor.analyze_commits_code_quality(tmpdir, fake_commits)
        assert enriched[0]['total_cc'] == 0
        assert enriched[0]['avg_mi'] == 0
        assert enriched[0]['ruff_errors'] == 0
        assert enriched[0]['py_files_changed_count'] == 0
        assert enriched[0]['author'] == 'testuser'

    # Setup: create a fake commit list with a single commit and a .py file
    fake_commits = [
        {
            'sha': 'deadbeef',
            'author': 'testuser',
            'timestamp': '2025-05-16 12:34:56 +0000',
            'message': 'Initial commit',
            'files': ['foo.py'],
            'py_files_changed_count': 1,
            'added': 10,
            'deleted': 2
        }
    ]

    # Patch GitPython Repo object to return a fake blob
    class FakeBlob:
        def data_stream(self):
            import io
            return io.BytesIO(b'def foo():\n    return 42\n')

    class FakeTree:
        def __getitem__(self, name):
            return FakeBlob()
    class FakeCommit:
        tree = FakeTree()
        class stats:
            files = {'foo.py': {}}
    class FakeGit:
        def show(self, ref):
            return 'def foo():\n    return 42\n'
    class FakeRepo:
        def __init__(self, *a, **k):
            self.git = FakeGit()
        def commit(self, sha):
            return FakeCommit()
    monkeypatch.setattr(commit_processor, 'Repo', FakeRepo)

    # Patch subprocess to simulate Ruff output
    def fake_run(cmd, input=None, text=None, capture_output=None):
        class Result:
            stdout = '[]'  # no errors
        return Result()
    monkeypatch.setattr(commit_processor.subprocess, 'run', fake_run)

    # Patch radon metrics
    monkeypatch.setattr(commit_processor, 'cc_visit', lambda code: [])
    monkeypatch.setattr(commit_processor, 'mi_visit', lambda code, _: 80.0)

    # Run
    with tempfile.TemporaryDirectory() as tmpdir:
        enriched = commit_processor.analyze_commits_code_quality(tmpdir, fake_commits)
        assert enriched[0]['total_cc'] == 0
        assert enriched[0]['avg_mi'] == 80.0
        assert enriched[0]['ruff_errors'] == 0
        assert enriched[0]['py_files_changed_count'] == 1
        assert enriched[0]['author'] == 'testuser'
