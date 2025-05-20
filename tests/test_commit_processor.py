import json
import pathlib
import tempfile
import sys
import pytest

# Ensure cultivation is on sys.path for import
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from cultivation.scripts.software.dev_daily_reflect.metrics import commit_processor

def test_analyze_commits_code_quality_handles_basic_commit(monkeypatch):
    # Setup: create a fake commit list with a single commit and a .py file
    """
    Tests that analyze_commits_code_quality correctly processes a commit with a Python file.
    
    Simulates a repository with a single commit modifying one Python file, mocks external dependencies,
    and verifies that the enriched commit data includes expected code quality metrics.
    """
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
            """
            Returns a byte stream containing a sample Python function definition.
            
            This can be used to simulate file-like input for testing or processing purposes.
            """
            import io
            return io.BytesIO(b'def foo():\n    return 42\n')

    class FakeTree:
        def __getitem__(self, name):
            """
            Returns a new FakeBlob instance for the given name.
            
            This method simulates dictionary-like access to retrieve a blob object, typically used in mocking repository tree behavior during tests.
            """
            return FakeBlob()
    class FakeCommit:
        tree = FakeTree()
        class stats:
            files = {'foo.py': {}}
    class FakeGit:
        def show(self, ref):
            """
            Returns the contents of a file at the given reference as a string.
            
            Args:
                ref: The reference (e.g., commit hash or branch name) to retrieve the file from.
            
            Returns:
                The file contents as a string.
            """
            return 'def foo():\n    return 42\n'
    class FakeRepo:
        def __init__(self, *a, **k):
            """
            Initializes the object with a fake Git interface for testing purposes.
            """
            self.git = FakeGit()
        def commit(self, sha):
            """
            Returns a fake commit object for the given SHA.
            
            This method is used to simulate retrieving a commit in test scenarios.
            """
            return FakeCommit()
    monkeypatch.setattr(commit_processor, 'Repo', FakeRepo)

    # Patch subprocess to simulate Ruff output
    def fake_run(cmd, input=None, text=None, capture_output=None):
        """
        Mocks subprocess.run to simulate Ruff linter output with no errors.
        
        Args:
            cmd: The command to execute.
            input: Ignored.
            text: Ignored.
            capture_output: Ignored.
        
        Returns:
            An object with a 'stdout' attribute set to '[]', representing no Ruff errors.
        """
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
    """
    Tests analyze_commits_code_quality for commits with and without Python files.
    
    Verifies that the function returns zeroed metrics when no Python files are present and correct metrics when a Python file is included, ensuring robustness across edge cases.
    """
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
            """
            Returns a placeholder README content string for the given reference.
            """
            return '# readme'
    class FakeRepo:
        def __init__(self, *a, **k):
            """
            Initializes the object with a fake Git interface for testing purposes.
            """
            self.git = FakeGit()
        def commit(self, sha):
            """
            Returns a fake commit object for the given SHA.
            
            This method is used to simulate retrieving a commit in test scenarios.
            """
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
            """
            Returns a byte stream containing a sample Python function definition.
            
            The stream provides the bytes for a simple function that returns the integer 42.
            """
            import io
            return io.BytesIO(b'def foo():\n    return 42\n')

    class FakeTree:
        def __getitem__(self, name):
            """
            Returns a new FakeBlob instance for the given name.
            
            This method simulates dictionary-like access to retrieve a blob object, typically used in mocking repository tree behavior during tests.
            """
            return FakeBlob()
    class FakeCommit:
        tree = FakeTree()
        class stats:
            files = {'foo.py': {}}
    class FakeGit:
        def show(self, ref):
            """
            Returns the source code for the given reference as a string.
            
            Args:
                ref: The reference identifier for the code object.
            
            Returns:
                The source code corresponding to the provided reference.
            """
            return 'def foo():\n    return 42\n'
    class FakeRepo:
        def __init__(self, *a, **k):
            """
            Initializes the object with a fake Git interface for testing purposes.
            """
            self.git = FakeGit()
        def commit(self, sha):
            """
            Returns a fake commit object for the given SHA.
            
            This method is used to mock repository commit retrieval in tests.
            """
            return FakeCommit()
    monkeypatch.setattr(commit_processor, 'Repo', FakeRepo)

    # Patch subprocess to simulate Ruff output
    def fake_run(cmd, input=None, text=None, capture_output=None):
        """
        Mocks subprocess.run to simulate Ruff linter output with no errors.
        
        Args:
            cmd: The command to execute.
            input: Ignored.
            text: Ignored.
            capture_output: Ignored.
        
        Returns:
            An object with a 'stdout' attribute set to '[]', indicating no Ruff errors.
        """
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
