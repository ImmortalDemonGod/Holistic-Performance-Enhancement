import sys
import pathlib
from datetime import datetime, timedelta
import tempfile
import os
import shutil
import pytest

# Ensure import from script directory
sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
from session_timer import log_session, parse_log_for_unclosed

# --- Helper for simulating log lines ---
def write_log_lines(log_file, lines):
    with open(log_file, "w") as f:
        for line in lines:
            f.write(line + "\n")


def test_detects_unclosed_session(tmp_path):
    log_file = tmp_path / "log.txt"
    # Simulate a session with only a START entry
    start_time = datetime(2025, 6, 3, 13, 0, 0)
    log_lines = [
        f"Task: 1.1\tStart: {start_time.isoformat()}\tEnd: {start_time.isoformat()}\tDuration: 0:00:00\tNote: START: test\tStatus: START"
    ]
    write_log_lines(log_file, log_lines)
    unclosed = parse_log_for_unclosed(log_file)
    assert len(unclosed) == 1
    assert unclosed[0][0] == "1.1"
    assert unclosed[0][1] == start_time.isoformat()

def test_ignores_closed_sessions(tmp_path):
    log_file = tmp_path / "log.txt"
    start_time = datetime(2025, 6, 3, 13, 0, 0)
    end_time = datetime(2025, 6, 3, 13, 30, 0)
    log_lines = [
        f"Task: 1.1\tStart: {start_time.isoformat()}\tEnd: {start_time.isoformat()}\tDuration: 0:00:00\tNote: START: test\tStatus: START",
        f"Task: 1.1\tStart: {start_time.isoformat()}\tEnd: {end_time.isoformat()}\tDuration: 0:30:00\tNote: COMPLETED: test\tStatus: END"
    ]
    write_log_lines(log_file, log_lines)
    unclosed = parse_log_for_unclosed(log_file)
    assert unclosed == []

def test_multiple_unclosed_sessions(tmp_path):
    log_file = tmp_path / "log.txt"
    s1 = datetime(2025, 6, 3, 13, 0, 0)
    s2 = datetime(2025, 6, 3, 14, 0, 0)
    log_lines = [
        f"Task: 1.1\tStart: {s1.isoformat()}\tEnd: {s1.isoformat()}\tDuration: 0:00:00\tNote: START: test1\tStatus: START",
        f"Task: 2.1\tStart: {s2.isoformat()}\tEnd: {s2.isoformat()}\tDuration: 0:00:00\tNote: START: test2\tStatus: START"
    ]
    write_log_lines(log_file, log_lines)
    unclosed = parse_log_for_unclosed(log_file)
    assert len(unclosed) == 2
    assert set(u[0] for u in unclosed) == {"1.1", "2.1"}

def test_recovery_entry_closes_session(tmp_path):
    log_file = tmp_path / "log.txt"
    s1 = datetime(2025, 6, 3, 13, 0, 0)
    e1 = datetime(2025, 6, 3, 13, 30, 0)
    # Write initial unclosed session
    log_lines = [
        f"Task: 1.1\tStart: {s1.isoformat()}\tEnd: {s1.isoformat()}\tDuration: 0:00:00\tNote: START: test\tStatus: START"
    ]
    write_log_lines(log_file, log_lines)
    # Now simulate recovery: log END
    log_session("1.1", s1, e1, e1-s1, "RECOVERED: test", log_file=str(log_file))
    # Manually add Status: END for this test
    with open(log_file, "a") as f:
        f.write(f"Task: 1.1\tStart: {s1.isoformat()}\tEnd: {e1.isoformat()}\tDuration: 0:30:00\tNote: RECOVERED: test\tStatus: END\n")
    unclosed = parse_log_for_unclosed(log_file)
    assert unclosed == []
