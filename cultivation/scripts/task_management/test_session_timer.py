"""Unit tests for session timer logging functionality."""

from datetime import datetime, timedelta
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
from session_timer import log_session


def test_log_session_creates_entry(tmp_path):
    """Test that log_session writes a correct entry to a log file."""
    log_file = tmp_path / "test_session_log.txt"
    task_id = "1.1"
    start = datetime(2025, 6, 3, 22, 17, 43)
    end = datetime(2025, 6, 3, 22, 43, 18)
    duration = end - start
    note = "TEST: Logging session"

    log_session(task_id, start, end, duration, note, "TEST", log_file=str(log_file))

    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert any("Task: 1.1" in line and "TEST: Logging session" in line for line in lines)
    assert any("Start: 2025-06-03T22:17:43" in line and "End: 2025-06-03T22:43:18" in line for line in lines)


def test_log_session_zero_duration(tmp_path):
    """Test logging when start and end times are the same (zero duration)."""
    log_file = tmp_path / "test_session_log.txt"
    task_id = "1.1"
    start = datetime(2025, 6, 3, 22, 17, 43)
    end = start
    duration = timedelta(0)
    note = "TEST: Zero duration"

    log_session(task_id, start, end, duration, note, "ZERO", log_file=str(log_file))

    with open(log_file, "r", encoding="utf-8") as f:
        log_lines = f.readlines()
    assert any("Duration: 0:00:00" in line for line in log_lines)
    assert any("TEST: Zero duration" in line for line in log_lines)


def test_log_session_interrupted(tmp_path):
    """Test logging an interrupted session."""
    log_file = tmp_path / "test_session_log.txt"
    task_id = "1.1"
    start = datetime(2025, 6, 3, 22, 17, 43)
    end = datetime(2025, 6, 3, 22, 20, 0)
    duration = end - start
    note = "INTERRUPTED: Simulated interruption"

    log_session(task_id, start, end, duration, note, "INTERRUPTED", log_file=str(log_file))

    with open(log_file, "r", encoding="utf-8") as f:
        log_lines = f.readlines()
    assert any("INTERRUPTED" in line for line in log_lines)
    assert any("Duration: 0:02:17" in line for line in log_lines)


def test_log_session_multiple_entries(tmp_path):
    """Test multiple log entries are appended, not overwritten."""
    log_file = tmp_path / "test_session_log.txt"
    task_id = "1.1"
    start = datetime(2025, 6, 3, 22, 17, 43)
    end = datetime(2025, 6, 3, 22, 43, 18)
    duration = end - start
    note1 = "START: First session"
    note2 = "COMPLETED: Second session"

    log_session(task_id, start, end, duration, note1, "COMPLETED", log_file=str(log_file))
    log_session(task_id, start, end, duration, note2, "COMPLETED", log_file=str(log_file))

    with open(log_file, "r", encoding="utf-8") as f:
        log_lines = f.readlines()
    assert any("START: First session" in line for line in log_lines)
    assert any("COMPLETED: Second session" in line for line in log_lines)
