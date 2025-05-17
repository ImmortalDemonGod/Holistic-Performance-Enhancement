#!/usr/bin/env python3
"""
reading_session.py - Instrumented reading session CLI for cultivation

Commands:
  start-reading <arxiv_id>   Start a new reading session
  end-reading <session_id>    End a reading session and log summary metrics
"""
import argparse
import sqlite3
import json
from pathlib import Path
from datetime import datetime
import sys

def get_db_path():
    """
    Returns the file path to the SQLite database, ensuring the 'literature' directory exists.
    
    The database file 'db.sqlite' is located in a 'literature' directory two levels above the script's location. The directory is created if it does not already exist.
    
    Returns:
        Path to the SQLite database file.
    """
    base = Path(__file__).parent.parent / 'literature'
    base.mkdir(parents=True, exist_ok=True)
    return base / 'db.sqlite'

def init_db(conn):
    """
    Initializes the SQLite database schema for reading sessions and event logs.
    
    Creates the 'sessions' and 'events' tables if they do not already exist, enabling
    storage of session metadata and associated event records.
    """
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            payload TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )
    ''')
    conn.commit()

def start_reading(args):
    """
    Starts a new reading session for a specified arXiv paper ID.
    
    Creates a new session entry in the database with the provided arXiv ID and the current UTC timestamp as the start time. Prints the session ID and start time upon successful creation.
    """
    db = get_db_path()
    conn = sqlite3.connect(db)
    init_db(conn)
    now = datetime.utcnow().isoformat() + 'Z'
    c = conn.cursor()
    c.execute('INSERT INTO sessions(paper_id, started_at) VALUES (?, ?)',
              (args.arxiv_id, now))
    conn.commit()
    session_id = c.lastrowid
    print(f"Started session {session_id} for paper {args.arxiv_id} at {now}")
    conn.close()


def end_reading(args):
    """
    Ends a reading session and records user feedback metrics.
    
    Marks the specified reading session as finished by updating its completion timestamp. Prompts the user for self-rated metrics (comprehension, relevance, personal novelty, and time spent), validates the input, and logs these metrics as a summary event linked to the session. Exits with an error if the session is not active or if input collection fails.
    """
    db = get_db_path()
    conn = sqlite3.connect(db)
    init_db(conn)
    c = conn.cursor()
    # update session finished_at
    now = datetime.utcnow().isoformat() + 'Z'
    c.execute(
        'UPDATE sessions SET finished_at = ? WHERE session_id = ? AND finished_at IS NULL',
        (now, args.session_id),
    )
    if c.rowcount == 0:
        print(f"No active session with id {args.session_id}", file=sys.stderr)
        conn.close()
        sys.exit(1)
    conn.commit()
    print(f"Session {args.session_id} marked finished at {now}")
    print(f"Session {args.session_id} marked finished at {now}")
    # prompt for self-rated metrics
    metrics = {}
    try:
        def ask_int(prompt, lo, hi):
            """
            Prompts the user to enter an integer within a specified range.
            
            Continuously requests input until the user provides an integer value between the given lower and upper bounds (inclusive).
            
            Args:
                prompt: The message displayed to the user.
                lo: The minimum acceptable integer value.
                hi: The maximum acceptable integer value.
            
            Returns:
                The validated integer entered by the user.
            """
            while True:
                try:
                    val = int(input(prompt))
                    if lo <= val <= hi:
                        return val
                except ValueError:
                    pass
                print(f"Enter an integer between {lo} and {hi}.")
        def ask_float(prompt, lo, hi):
            """
            Prompts the user to enter a floating-point number within a specified range.
            
            Continuously requests input until the user provides a valid float between the given lower and upper bounds (inclusive).
            
            Args:
                prompt: The message displayed to the user.
                lo: The minimum acceptable value.
                hi: The maximum acceptable value.
            
            Returns:
                The validated floating-point number entered by the user.
            """
            while True:
                try:
                    val = float(input(prompt))
                    if lo <= val <= hi:
                        return val
                except ValueError:
                    pass
                print(f"Enter a number between {lo} and {hi}.")
        metrics['self_rated_comprehension'] = ask_int('Self-rated comprehension (0-5): ', 0, 5)
        metrics['self_rated_relevance'] = ask_int('Self-rated relevance (0-5): ', 0, 5)
        metrics['self_rated_novelty_personal'] = ask_float('Self-rated novelty personal (0-1): ', 0.0, 1.0)
        metrics['actual_time_spent_minutes'] = ask_float('Actual time spent (minutes): ', 0.0, 10000.0)
    except Exception as e:
        print(f"Error reading input: {e}")
        conn.close()
        sys.exit(1)
    # log summary event
    payload = json.dumps(metrics)
    c.execute(
        'INSERT INTO events(session_id, event_type, timestamp, payload) VALUES (?, ?, ?, ?)',
        (args.session_id, 'session_summary_user', now, payload)
    )
    conn.commit()
    print(f"Logged session_summary_user for session {args.session_id}")
    conn.close()


def main():
    """
    Parses command-line arguments and dispatches to the appropriate reading session command.
    
    Defines and handles the 'start-reading' and 'end-reading' subcommands for managing reading sessions. Exits with status 1 if no command is provided.
    """
    parser = argparse.ArgumentParser(prog='cultivation literature reading-session')
    sub = parser.add_subparsers(dest='command')

    p_start = sub.add_parser('start-reading', help='Start a reading session')
    p_start.add_argument('arxiv_id', type=str, help='arXiv ID of the paper')
    p_start.set_defaults(func=start_reading)

    p_end = sub.add_parser('end-reading', help='End a reading session')
    p_end.add_argument('session_id', type=int, help='ID of the session to end')
    p_end.set_defaults(func=end_reading)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)

if __name__ == '__main__':
    main()
