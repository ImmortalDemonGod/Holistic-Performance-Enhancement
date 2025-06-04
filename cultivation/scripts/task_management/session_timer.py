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

def log_session(task_id_display, start_time, end_time, duration_td, note_str, status, log_file=None):
    if log_file is None:
        log_file = os.environ.get("LOG_FILE", DEFAULT_LOG_FILE)
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_line = (
        f"Task: {task_id_display}\t"
        f"Start: {start_time.isoformat()}\t"
        f"End: {end_time.isoformat()}\t"
        f"Duration: {str(duration_td)}\t"
        f"Note: {note_str}\t"
        f"Status: {status}\n"
    )
    with open(log_file, "a") as f:
        f.write(log_line)

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

def main():
    LOG_FILE = os.environ.get("LOG_FILE", DEFAULT_LOG_FILE)
    print(f"DEBUG: LOG_FILE (main): {LOG_FILE}", flush=True)
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # --- RECOVERY LOGIC ---
    unclosed = parse_log_for_unclosed(log_file=LOG_FILE)
    if unclosed:
        print("\n=== Session Recovery Needed ===")
        for idx, (task, start, note) in enumerate(unclosed, 1):
            print(f"[{idx}] Task: {task}, Started: {start}, Note: {note}")
        for task, start, note in unclosed:
            print(f"Recovering session for Task {task} started at {start}...")
            end_time_str = prompt(f"Enter end time for Task {task} (YYYY-MM-DDTHH:MM, default=now)", default=datetime.now().strftime("%Y-%m-%dT%H:%M"))
            try:
                end_time_dt = datetime.strptime(end_time_str, "%Y-%m-%dT%H:%M")
            except Exception:
                print("Invalid format, using current time.")
                end_time_dt = datetime.now()
            start_time_dt = datetime.fromisoformat(start)
            duration_td = end_time_dt - start_time_dt
            log_session(task, start_time_dt, end_time_dt, duration_td, f"RECOVERED: {note}", status="END", log_file=LOG_FILE)
            print(f"Session for Task {task} closed. Duration: {duration_td}")
        print("=== All unclosed sessions recovered. ===\n")

    parser = argparse.ArgumentParser(description="Session Timer for RNA Curriculum (Task Master integration)")
    parser.add_argument('--parent', type=str, default=None, help='Parent Task Master ID (e.g., 1, 23)')
    parser.add_argument('--sub', type=str, default=None, help='Subtask Local ID (e.g., 1, 2, or a, b). Leave blank if timing parent task')
    parser.add_argument('--minutes', type=int, default=None, help='Session duration in minutes')
    parser.add_argument('--note', type=str, default=None, help='Session note (optional)')
    parser.add_argument('--manual', type=str, choices=['start','stop'], help='Manual punch in/out mode: start or stop')
    args = parser.parse_args()

    parent_task_id_str = args.parent if args.parent is not None else prompt("Parent Task Master ID (e.g., 1, 23)", default="1").strip()
    sub_task_local_id_str = args.sub if args.sub is not None else prompt("Subtask Local ID (e.g., 1, 2, or a, b if your Task Master uses that. Leave blank if timing parent task)", default="").strip()
    if sub_task_local_id_str:
        task_id_for_tm = f"{parent_task_id_str}.{sub_task_local_id_str}"
        task_id_display = task_id_for_tm
    else:
        task_id_for_tm = parent_task_id_str
        task_id_display = parent_task_id_str

    # --- MANUAL MODE (Punch In/Out) ---
    if args.manual:
        LOG_FILE = os.environ.get("LOG_FILE", DEFAULT_LOG_FILE)
        if args.manual == 'start':
            # Check for existing unclosed session
            unclosed = [u for u in parse_log_for_unclosed(log_file=LOG_FILE) if u[0] == task_id_display]
            if unclosed:
                print(f"WARNING: There is already an unclosed session for Task {task_id_display} started at {unclosed[0][1]}.")
                sys.exit(1)
            session_note_str = args.note if args.note is not None else prompt("Session note (optional)", default="")
            start_time_dt = datetime.now()
            log_session(task_id_display, start_time_dt, start_time_dt, timedelta(0), f"START: {session_note_str}", status="START")
            print(f"Manual session START logged for Task {task_id_display} at {start_time_dt.isoformat()}.")
            return
        elif args.manual == 'stop':
            # Find most recent unclosed session for this task
            unclosed = [u for u in parse_log_for_unclosed(log_file=LOG_FILE) if u[0] == task_id_display]
            if not unclosed:
                print(f"No unclosed session found for Task {task_id_display}.")
                sys.exit(1)
            task, start_str, start_note = unclosed[0]
            start_time_dt = datetime.fromisoformat(start_str)
            end_time_dt = datetime.now()
            duration_td = end_time_dt - start_time_dt
            session_note_str = args.note if args.note is not None else prompt("Session note for END (optional)", default=start_note)
            log_session(task_id_display, start_time_dt, end_time_dt, duration_td, f"MANUAL END: {session_note_str}", status="END")
            print(f"Manual session END logged for Task {task_id_display}. Duration: {duration_td}")
            # Optionally update Task Master
            update_tm_choice = prompt(f"Update Task Master for {task_id_for_tm} with comment? (y/n)", default="y").lower()
            if update_tm_choice.startswith("y"):
                comment_for_tm = (
                    f"Manual timed session: {start_time_dt.strftime('%Y-%m-%d %H:%M')} - {end_time_dt.strftime('%H:%M')} "
                    f"(Actual: {int(duration_td.total_seconds()/60)} min). Note: {session_note_str}"
                )
                update_task_master(task_id_for_tm, comment_for_tm)
            set_done_choice = prompt(f"Set task {task_id_for_tm} as done in Task Master? (y/n)", default="n").lower()
            if set_done_choice.startswith("y"):
                done_comment = f"Session completed, marked as done. Note: {session_note_str}"
                update_task_master(task_id_for_tm, done_comment, status_to_set="done")
            print(f"Session for Task {task_id_display} logged and (optionally) Task Master updated.")
            return

    # --- FIXED DURATION MODE (Default) ---
    minutes = args.minutes if args.minutes is not None else None
    if minutes is None:
        minutes_str = prompt(f"Session duration for Task {task_id_display} (minutes)", default="60")
        try:
            minutes = int(minutes_str)
            if minutes <= 0:
                print("Duration must be a positive number of minutes.")
                sys.exit(1)
        except ValueError:
            print("Invalid duration. Please enter a number.")
            sys.exit(1)
    session_note_str = args.note if args.note is not None else prompt("Session note (optional)", default="")
    print(f"Starting timer for Task {task_id_display} ({minutes} min)...")
    start_time_dt = datetime.now()
    log_session(task_id_display, start_time_dt, start_time_dt, timedelta(0), f"START: {session_note_str}", status="START")
    try:
        time.sleep(minutes * 60)
        end_reason = "COMPLETED"
    except KeyboardInterrupt:
        print(f"\nSession for Task {task_id_display} interrupted.")
        end_reason = "INTERRUPTED"
    end_time_dt = datetime.now()
    actual_duration_td = end_time_dt - start_time_dt
    actual_minutes_int = int(actual_duration_td.total_seconds() / 60)
    final_note = f"{end_reason}: {session_note_str}"
    log_session(task_id_display, start_time_dt, end_time_dt, actual_duration_td, final_note, status="END")
    if end_reason == "COMPLETED":
        mac_notify("Session Complete", f"Task {task_id_display} complete! Duration: {actual_minutes_int} min")
    else:
        mac_notify("Session Interrupted", f"Task {task_id_display} interrupted at {end_time_dt.strftime('%H:%M')}. Duration: {actual_minutes_int} min")
    update_tm_choice = prompt(f"Update Task Master for {task_id_for_tm} with comment? (y/n)", default="y").lower()
    if update_tm_choice.startswith("y"):
        comment_for_tm = (
            f"Timed session: {start_time_dt.strftime('%Y-%m-%d %H:%M')} - {end_time_dt.strftime('%H:%M')} "
            f"(Actual: {actual_minutes_int} min, Planned: {minutes} min). "
            f"Outcome: {end_reason}. Note: {session_note_str}"
        )
        update_task_master(task_id_for_tm, comment_for_tm)
    if end_reason == "COMPLETED":
        set_done_choice = prompt(f"Set task {task_id_for_tm} as done in Task Master? (y/n)", default="n").lower()
        if set_done_choice.startswith("y"):
            done_comment = f"Session completed, marked as done. Initial note: {session_note_str}"
            update_task_master(task_id_for_tm, done_comment, status_to_set="done")
    print(f"Session for Task {task_id_display} logged and (optionally) Task Master updated.")

if __name__ == "__main__":
    main()
