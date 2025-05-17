import tempfile
import pathlib
import sys
import json
import subprocess
import shutil
import os
import types
import pytest

# Patch sys.path to allow absolute imports
sys.path.insert(0, str(pathlib.Path(__file__).parents[0]))

# Path to the script under test
SCRIPT = pathlib.Path(__file__).parents[0] / "../cultivation/scripts/software/dev_daily_reflect/ingest_git.py"
SCRIPT = SCRIPT.resolve()

def create_fake_git_repo(tmpdir):
    repo_path = tmpdir / "repo"
    repo_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo_path, check=True)
    # Add a Python file and commit
    (repo_path / "foo.py").write_text("print('hello world')\n")
    subprocess.run(["git", "add", "foo.py"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True)
    return repo_path

@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_ingest_git_basic(monkeypatch, tmp_path):
    """Integration: ingest_git.py produces expected JSON output from a minimal repo."""
    # Find the project root (parent of 'cultivation')
    project_root = pathlib.Path(__file__).parent.parent.resolve()
    # Create the fake repo inside the project root's temp subdir
    repo_path = project_root / "test_fake_repo"
    if repo_path.exists():
        shutil.rmtree(repo_path)
    repo_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo_path, check=True)
    (repo_path / "foo.py").write_text("print('hello world')\n")
    subprocess.run(["git", "add", "foo.py"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True)
    # Monkeypatch get_repo_root to return our temp repo
    sys.modules.pop('cultivation.scripts.software.dev_daily_reflect.utils', None)
    import importlib.util
    utils_path = SCRIPT.parent / "utils.py"
    spec = importlib.util.spec_from_file_location("cultivation.scripts.software.dev_daily_reflect.utils", utils_path)
    utils_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils_mod)
    monkeypatch.setattr(utils_mod, "get_repo_root", lambda start_path=None: repo_path)
    sys.modules["cultivation.scripts.software.dev_daily_reflect.utils"] = utils_mod
    # Set env and CWD to project root
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    result = subprocess.run([sys.executable, str(SCRIPT)], cwd=project_root, capture_output=True, text=True, env=env)
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    assert result.returncode == 0
    # Check that output JSON exists and is valid
    raw_files = list((repo_path / "cultivation/outputs/software/dev_daily_reflect/raw").glob("git_commits_*.json"))
    assert raw_files, "No output JSON found"
    with open(raw_files[0]) as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert data and "sha" in data[0] and "author" in data[0]
    # Cleanup
    shutil.rmtree(repo_path)
