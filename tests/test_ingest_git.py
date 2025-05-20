import tempfile
import pathlib
import sys
import json
import subprocess
import shutil
import os
import pytest
import re # Ensure re is imported

# Patch sys.path to allow absolute imports
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # Parent of tests dir = project root

# Path to the script under test
# Corrected SCRIPT path definition
SCRIPT_REL_PATH = pathlib.Path('../cultivation/scripts/software/dev_daily_reflect/ingest_git.py')
SCRIPT = (pathlib.Path(__file__).parent / SCRIPT_REL_PATH).resolve()


def create_fake_git_repo(tmpdir):
    """
    Creates a temporary Git repository with an initial Python file and commit.
    
    Args:
        tmpdir: The temporary directory in which to create the repository.
    
    Returns:
        The path to the created Git repository.
    """
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
    """Integration: ingest_git.py produces expected JSON output from a minimal repo using default date (lookback)."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir(exist_ok=True)
    subprocess.run(["git", "init"], cwd=repo_path, check=True)
    (repo_path / "foo.py").write_text("print('hello world basic')\n")
    subprocess.run(["git", "add", "foo.py"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "Initial basic commit"], cwd=repo_path, check=True)

    env = os.environ.copy()
    project_root = pathlib.Path(__file__).parents[1].resolve()
    env["PYTHONPATH"] = str(project_root)
    env['CULTIVATION_REPO_ROOT_OVERRIDE'] = str(repo_path)

    script_absolute_path = SCRIPT.resolve()
    result = subprocess.run(
        [sys.executable, str(script_absolute_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env
    )
    print("STDOUT (basic):\n", result.stdout)
    print("STDERR (basic):\n", result.stderr)
    assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"

    output_json_dir = repo_path / "cultivation" / "outputs" / "software" / "dev_daily_reflect" / "raw"
    
    date_tag_from_stdout = None
    for line in result.stdout.splitlines():
        if "current date tag:" in line:
            match = re.search(r"current date tag: (\d{4}-\d{2}-\d{2})", line)
            if match:
                date_tag_from_stdout = match.group(1)
                break
    assert date_tag_from_stdout, "Could not determine date_tag from script stdout for basic test"

    expected_raw_file = output_json_dir / f"git_commits_{date_tag_from_stdout}.json"
    assert expected_raw_file.exists(), f"Expected raw output file {expected_raw_file} not found in {output_json_dir}. Files: {list(output_json_dir.glob('*'))}"

    with open(expected_raw_file) as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]['message'] == "Initial basic commit"
    # shutil.rmtree(repo_path) # tmp_path handles cleanup


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_ingest_git_date_arg(monkeypatch, tmp_path):
    """Integration: ingest_git.py produces expected JSON for a specific --date argument."""
    repo_path = tmp_path / "repo_date_arg" # Use a different subdir to avoid conflicts
    repo_path.mkdir(exist_ok=True)
    
    # Create a commit with a specific, old date
    commit_date_str = "2024-01-15T12:00:00"
    env_for_commit = os.environ.copy()
    env_for_commit['GIT_AUTHOR_DATE'] = commit_date_str
    env_for_commit['GIT_COMMITTER_DATE'] = commit_date_str

    subprocess.run(["git", "init"], cwd=repo_path, check=True)
    (repo_path / "bar.py").write_text("print('hello from past')\n")
    subprocess.run(["git", "add", "bar.py"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "Commit for specific date"], cwd=repo_path, check=True, env=env_for_commit)

    target_date_arg = "2024-01-15" # The date we want the script to process

    env_for_script = os.environ.copy()
    project_root = pathlib.Path(__file__).parents[1].resolve()
    env_for_script["PYTHONPATH"] = str(project_root)
    env_for_script['CULTIVATION_REPO_ROOT_OVERRIDE'] = str(repo_path)

    script_absolute_path = SCRIPT.resolve()
    result = subprocess.run(
        [sys.executable, str(script_absolute_path), f"--date={target_date_arg}"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env_for_script
    )
    print("STDOUT (date_arg):\n", result.stdout)
    print("STDERR (date_arg):\n", result.stderr)
    assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"

    output_json_dir = repo_path / "cultivation" / "outputs" / "software" / "dev_daily_reflect" / "raw"
    
    # For --date arg, the date_tag in filename should match the argument
    expected_raw_file = output_json_dir / f"git_commits_{target_date_arg}.json"
    assert expected_raw_file.exists(), f"Expected raw output file {expected_raw_file} not found in {output_json_dir}. Files: {list(output_json_dir.glob('*'))}"
    
    # Verify the enriched file for the specific date also exists
    expected_enriched_file = output_json_dir / f"git_commits_enriched_{target_date_arg}.json"
    assert expected_enriched_file.exists(), f"Expected enriched file {expected_enriched_file} not found."

    with open(expected_raw_file) as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) >= 1, "No commits found in output for specific date"
    assert data[0]['message'] == "Commit for specific date"
    assert target_date_arg in data[0]['timestamp'] # Check if commit date matches target date
    # shutil.rmtree(repo_path) # tmp_path handles cleanup
