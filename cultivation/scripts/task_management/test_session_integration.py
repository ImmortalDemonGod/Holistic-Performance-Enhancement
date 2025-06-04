import subprocess
import tempfile
import os
import pathlib
from datetime import datetime, timedelta

def make_logfile():
    tmpdir = tempfile.TemporaryDirectory()
    log_file = os.path.join(tmpdir.name, "log.txt")
    return tmpdir, log_file

def run_timer_script(args, env=None):
    script = str(pathlib.Path(__file__).parent / "session_timer.py")
    result = subprocess.run([
        os.environ.get("PYTHON", "python3"), script
    ] + args, input="", capture_output=True, text=True, env=env)
    return result

def test_recovery_flow(tmp_path):
    # Setup: simulate a START entry in the log file
    log_file = tmp_path / "log.txt"
    task_id = "1.1"
    start = datetime(2025, 6, 3, 13, 0, 0)
    with open(log_file, "w") as f:
        f.write(f"Task: {task_id}\tStart: {start.isoformat()}\tEnd: {start.isoformat()}\tDuration: 0:00:00\tNote: START: test\tStatus: START\n")
    # Set env to use this log file
    env = os.environ.copy()
    env["SESSION_LOG"] = str(log_file)
    env["LOG_FILE"] = str(log_file)
    # Run script, simulate user pressing enter to accept default end time
    result = subprocess.run([
        os.environ.get("PYTHON", "python3"), str(pathlib.Path(__file__).parent / "session_timer.py")
    ], input="\n", capture_output=True, text=True, env=env)
    # Check output
    assert "Session Recovery Needed" in result.stdout
    assert "closed. Duration:" in result.stdout
    # Check that END entry was appended
    with open(log_file) as f:
        lines = f.readlines()
    assert any("Status: END" in l for l in lines)
    assert sum("Status: START" in l for l in lines) == 1
    assert sum("Status: END" in l for l in lines) == 1

def test_normal_session(tmp_path):
    log_file = tmp_path / "log.txt"
    env = os.environ.copy()
    env["SESSION_LOG"] = str(log_file)
    env["LOG_FILE"] = str(log_file)
    # Run a short session (1s)
    args = ["--parent", "1", "--sub", "1", "--minutes", "0"]
    # Simulate user input for prompts (accept all defaults)
    result = subprocess.run([
        os.environ.get("PYTHON", "python3"), str(pathlib.Path(__file__).parent / "session_timer.py")
    ] + args, input="\n\n\n", capture_output=True, text=True, env=env, timeout=5)
    # Check that both START and END are in the log
    with open(log_file) as f:
        lines = f.readlines()
    assert any("Status: START" in l for l in lines)
    assert any("Status: END" in l for l in lines)
