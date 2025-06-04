#!/usr/bin/env python3
"""
Session Timer Script for Holistic-Performance-Enhancement
- Logs session start/end/duration to .taskmaster/session_log.txt
- Optionally updates Task Master with status/comment
- Mac notification at session end
- Uses .venv/bin/python
"""
import os
import sys
import time
import subprocess
from datetime import datetime, timedelta

# --- CONFIG ---
# --- LOG FILE CONFIG ---
# The session log is now stored inside the repo for version control and collaboration.
# Default: cultivation/logs/session_log.txt (relative to project root)
# Can override with LOG_FILE env variable for testing or custom setups.
REPO_LOG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/session_log.txt'))
DEFAULT_LOG_FILE = REPO_LOG_FILE
TASK_MASTER = "task-master"
PYTHON_EXEC = os.path.abspath(".venv/bin/python")

print(f"DEBUG: LOG_FILE (module scope): {os.environ.get('LOG_FILE', DEFAULT_LOG_FILE)}", flush=True)

# --- Prompt for session info ---
def prompt(msg, default=None):
    if default is not None:
        return input(f"{msg} [{default}]: ") or default
    return input(f"{msg}: ")

import os

# The log file is now stored in the repo by default; override with LOG_FILE env var for testing or custom setups.
LOG_FILE = os.environ.get("LOG_FILE", DEFAULT_LOG_FILE)

import json

def log_session(task_id_display, start_time, end_time, duration_td, note_str, status, log_file=None):
    """
    Log a session as a JSON object per line for robust parsing and encoding safety.
    Args:
        task_id_display (str): Human-readable task ID or label.
        start_time (datetime): Session start time.
        end_time (datetime): Session end time.
        duration_td (timedelta): Session duration.
        note_str (str): Notes (may contain tabs or special chars).
        status (str): Task/session status.
        log_file (str, optional): Path to log file. Defaults to LOG_FILE env or DEFAULT_LOG_FILE.
    """
    if log_file is None:
        log_file = os.environ.get("LOG_FILE", DEFAULT_LOG_FILE)
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_entry = {
        "task": task_id_display,
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "duration": str(duration_td),
        "note": note_str,
        "status": status
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def mac_notify(title, msg):
    subprocess.run([
        "osascript", "-e", f'display notification "{msg}" with title "{title}"'
    ])

def update_task_master(task_id_for_tm, comment_text, status_to_set=None):
    # Add a comment
    try:
        subprocess.run([
            TASK_MASTER, "comment", "--id", str(task_id_for_tm), "--text", comment_text
        ], check=True)
        print(f"Successfully added comment to Task Master for {task_id_for_tm}.")
    except subprocess.CalledProcessError as e:
        print(f"Error adding comment to Task Master for {task_id_for_tm}: {e}")
    except FileNotFoundError:
        print(f"Error: '{TASK_MASTER}' command not found. Is it installed and in PATH?")
    # Optionally set status
    if status_to_set:
        try:
            subprocess.run([
                TASK_MASTER, "set-status", f"--id={task_id_for_tm}", f"--status={status_to_set}"
            ], check=True)
            print(f"Successfully set status to '{status_to_set}' for {task_id_for_tm} in Task Master.")
        except subprocess.CalledProcessError as e:
            print(f"Error setting status for {task_id_for_tm} in Task Master: {e}")
        except FileNotFoundError:
            print(f"Error: '{TASK_MASTER}' command not found.")

import argparse

def parse_log_for_unclosed(log_file=None):
    if log_file is None:
        log_file = os.environ.get("LOG_FILE", DEFAULT_LOG_FILE)
    starts = {}
    ends = set()
    if not os.path.exists(log_file):
        return []
    with open(log_file) as f:
        for line in f:
            if not line.startswith("Task:"):
                continue  # Skip notes, summaries, blank lines, etc.
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue  # Skip malformed or incomplete log lines
            if "Status: START" in line:
                try:
                    task = parts[0].split(": ")[1]
                    start = parts[1].split(": ")[1]
                    note = parts[-2].split(": ")[1].strip()
                    starts[(task, start)] = note
                except Exception:
                    continue  # Skip lines that don't parse cleanly
            elif "Status: END" in line:
                try:
                    task = parts[0].split(": ")[1]
                    start = parts[1].split(": ")[1]
                    ends.add((task, start))
                except Exception:
                    continue
    return [(task, start, note) for (task, start), note in starts.items() if (task, start) not in ends]

def handle_recovery(log_file):
    """Handle recovery of any unclosed sessions in the log."""
    unclosed = parse_log_for_unclosed(log_file=log_file)
    if unclosed:
        print("\n=== Session Recovery Needed ===")
        for idx, (task, start, note) in enumerate(unclosed, 1):
            print(f"[{idx}] Task: {task}, Started: {start}, Note: {note}")
        for task, start, note in unclosed:
            print(f"Recovering session for Task {task} started at {start}...")
            end_time_str = prompt(
                f"Enter end time for Task {task} (YYYY-MM-DDTHH:MM, default=now)",
                default=datetime.now().strftime("%Y-%m-%dT%H:%M"),
            )
            try:
                end_time_dt = datetime.strptime(end_time_str, "%Y-%m-%dT%H:%M")
            except ValueError:
                print("Invalid format, using current time.")
                end_time_dt = datetime.now()
            start_time_dt = datetime.fromisoformat(start)
            duration_td = end_time_dt - start_time_dt
            log_session(
                task,
                start_time_dt,
                end_time_dt,
                duration_td,
                f"RECOVERED: {note}",
                status="END",
                log_file=log_file,
            )
            print(f"Session for Task {task} closed. Duration: {duration_td}")
        print("=== All unclosed sessions recovered. ===\n")


def parse_args_and_task_ids():
    """Parse CLI arguments and compose Task Master IDs for parent or subtask."""
    parser = argparse.ArgumentParser(
        description="Session Timer for RNA Curriculum (Task Master integration)"
    )
    parser.add_argument('--parent', type=str, default=None,
                        help='Parent Task Master ID (e.g., 1, 23)')
    parser.add_argument('--sub', type=str, default=None,
                        help='Subtask Local ID (e.g., 1, 2, or a, b). Leave blank if timing parent task')
    parser.add_argument('--minutes', type=int, default=None,
                        help='Session duration in minutes')
    parser.add_argument('--note', type=str, default=None,
                        help='Session note (optional)')
    parser.add_argument('--manual', type=str, choices=['start','stop'],
                        help='Manual punch in/out mode: start or stop')
    args = parser.parse_args()
    parent_task_id_str = args.parent or prompt(
        "Parent Task Master ID (e.g., 1, 23)", default="1"
    ).strip()
    sub_task_local_id_str = args.sub or prompt(
        "Subtask Local ID (e.g., 1, 2, or a, b if your Task Master uses that. Leave blank if timing parent task)",
        default=""
    ).strip()
    if sub_task_local_id_str:
        task_id_for_tm = f"{parent_task_id_str}.{sub_task_local_id_str}"
        task_id_display = task_id_for_tm
    else:
        task_id_for_tm = parent_task_id_str
        task_id_display = parent_task_id_str
    return args, task_id_for_tm, task_id_display


def manual_start(task_id_display, note, log_file):
    """Handle manual 'start' punch-in flow."""
    existing = [u for u in parse_log_for_unclosed(log_file=log_file)
                if u[0] == task_id_display]
    if existing:
        print(
            f"WARNING: Unclosed session for Task {task_id_display} "
            f"started at {existing[0][1]}."
        )
        sys.exit(1)
    session_note = note if note is not None else prompt(
        "Session note (optional)", default=""
    )
    start_time = datetime.now()
    log_session(
        task_id_display, start_time, start_time, timedelta(0),
        f"START: {session_note}", status="START",
        log_file=log_file,
    )
    print(
        f"Manual session START logged for Task {task_id_display} "
        f"at {start_time.isoformat()}."
    )


def manual_stop(task_id_for_tm, task_id_display, note, log_file):
    """Handle manual 'stop' punch-out flow and optional Task Master updates."""
    unclosed = [u for u in parse_log_for_unclosed(log_file=log_file)
                if u[0] == task_id_display]
    if not unclosed:
        print(f"No unclosed session found for Task {task_id_display}.")
        sys.exit(1)
    _, start_str, start_note = unclosed[0]
    start_time = datetime.fromisoformat(start_str)
    end_time = datetime.now()
    duration = end_time - start_time
    session_note = note if note is not None else prompt(
        "Session note for END (optional)", default=start_note
    )
    log_session(
        task_id_display, start_time, end_time, duration,
        f"MANUAL END: {session_note}", status="END",
        log_file=log_file,
    )
    print(f"Manual session END logged for Task {task_id_display}. Duration: {duration}")
    if prompt(
        f"Update Task Master for {task_id_for_tm} with comment? (y/n)",
        default="y"
    ).lower().startswith("y"):
        comment = (
            f"Manual timed session: {start_time.strftime('%Y-%m-%d %H:%M')} - "
            f"{end_time.strftime('%H:%M')} (Actual: {int(duration.total_seconds()/60)} min). "
            f"Note: {session_note}"
        )
        update_task_master(task_id_for_tm, comment)
    if prompt(
        f"Set task {task_id_for_tm} as done in Task Master? (y/n)",
        default="n"
    ).lower().startswith("y"):
        done_c = f"Session completed, marked as done. Note: {session_note}"
        update_task_master(task_id_for_tm, done_c, status_to_set="done")
    print(
        f"Session for Task {task_id_display} logged and "
        f"(optionally) Task Master updated."
    )


def timed_session(args, task_id_for_tm, task_id_display, log_file):
    """Handle fixed-duration session flow with timing, logging, notifications."""
    minutes = args.minutes
    if minutes is None:
        minutes_str = prompt(
            f"Session duration for Task {task_id_display} (minutes)",
            default="60"
        )
        try:
            minutes = int(minutes_str)
        except ValueError:
            print("Invalid duration. Please enter a number.")
            sys.exit(1)
    try:
        minutes = int(minutes)
    except (TypeError, ValueError):
        print("Invalid duration. Please enter a number.")
        sys.exit(1)
    if minutes <= 0:
        print("Duration must be a positive number of minutes.")
        sys.exit(1)
    note = args.note or prompt("Session note (optional)", default="")
    print(f"Starting timer for Task {task_id_display} ({minutes} min)...")
    start_time = datetime.now()
    log_session(
        task_id_display, start_time, start_time, timedelta(0),
        f"START: {note}", status="START",
        log_file=log_file,
    )
    try:
        time.sleep(minutes * 60)
        reason = "COMPLETED"
    except KeyboardInterrupt:
        print(f"\nSession for Task {task_id_display} interrupted.")
        reason = "INTERRUPTED"
    end_time = datetime.now()
    actual = end_time - start_time
    mins_actual = int(actual.total_seconds() / 60)
    final_note = f"{reason}: {note}"
    log_session(
        task_id_display, start_time, end_time, actual,
        final_note, status="END",
        log_file=log_file,
    )
    if reason == "COMPLETED":
        mac_notify(
            "Session Complete",
            f"Task {task_id_display} complete! Duration: {mins_actual} min"
        )
    else:
        mac_notify(
            "Session Interrupted",
            f"Task {task_id_display} interrupted at {end_time.strftime('%H:%M')}. "
            f"Duration: {mins_actual} min"
        )
    if prompt(
        f"Update Task Master for {task_id_for_tm} with comment? (y/n)",
        default="y"
    ).lower().startswith("y"):
        comment = (
            f"Timed session: {start_time.strftime('%Y-%m-%d %H:%M')} - "
            f"{end_time.strftime('%H:%M')} (Actual: {mins_actual} min, Planned: {minutes} min). "
            f"Outcome: {reason}. Note: {note}"
        )
        update_task_master(task_id_for_tm, comment)
    if reason == "COMPLETED" and prompt(
        f"Set task {task_id_for_tm} as done in Task Master? (y/n)",
        default="n"
    ).lower().startswith("y"):
        done_c = f"Session completed, marked as done. Initial note: {note}"
        update_task_master(task_id_for_tm, done_c, status_to_set="done")
    print(
        f"Session for Task {task_id_display} logged and "
        f"(optionally) Task Master updated."
    )

# --- Simplified main orchestration ---

def main():
    log_file = os.environ.get("LOG_FILE", DEFAULT_LOG_FILE)
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    handle_recovery(log_file)
    args, task_id_for_tm, task_id_display = parse_args_and_task_ids()
    if args.manual == 'start':
        manual_start(task_id_display, args.note, log_file)
        return
    if args.manual == 'stop':
        manual_stop(task_id_for_tm, task_id_display, args.note, log_file)
        return
    timed_session(args, task_id_for_tm, task_id_display, log_file)

if __name__ == "__main__":
    main()
