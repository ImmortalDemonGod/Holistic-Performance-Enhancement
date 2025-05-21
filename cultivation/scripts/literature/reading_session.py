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
from datetime import datetime, UTC
import sys

def get_db_path():
    # Always use the canonical backend DB path, regardless of invocation location
    db_path = Path(__file__).parent.parent.parent / 'literature' / 'db.sqlite'
    print(f"DEBUG: CLI using DB path: {db_path.resolve()}")
    return db_path

def init_db(conn):
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
    db = get_db_path()
    conn = sqlite3.connect(db)
    init_db(conn)
    now = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
    c = conn.cursor()
    c.execute('INSERT INTO sessions(paper_id, started_at) VALUES (?, ?)',
              (args.arxiv_id, now))
    conn.commit()
    session_id = c.lastrowid
    print(f"Started session {session_id} for paper {args.arxiv_id} at {now}")
    conn.close()


def list_open(args=None, return_rows=False):
    db = get_db_path()
    conn = sqlite3.connect(db)
    init_db(conn)
    c = conn.cursor()
    c.execute('SELECT session_id, paper_id, started_at FROM sessions WHERE finished_at IS NULL ORDER BY started_at DESC')
    rows = c.fetchall()
    print("DEBUG: Raw rows returned from DB:", rows)
    if not rows:
        print("No open sessions.")
        conn.close()
        return [] if return_rows else None
    print("Open sessions:")
    print("{:<10} {:<20} {:<25}".format("session_id", "paper_id", "started_at"))
    for sid, pid, start in rows:
        print(f"{sid:<10} {pid:<20} {start:<25}")
    conn.close()
    return rows if return_rows else None

def end_reading_interactive(args):
    # Allow session_id to be optional; if missing, prompt user to select from open sessions
    session_id = getattr(args, 'session_id', None)
    if session_id is None:
        rows = list_open(return_rows=True)
        if not rows:
            sys.exit(0)
        try:
            session_id = int(input("Enter session_id to end: "))
        except Exception:
            print("Invalid input.")
            sys.exit(1)
    else:
        session_id = args.session_id
    # Now proceed as in original end_reading
    db = get_db_path()
    conn = sqlite3.connect(db)
    init_db(conn)
    c = conn.cursor()
    now = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
    c.execute(
        'UPDATE sessions SET finished_at = ? WHERE session_id = ? AND finished_at IS NULL',
        (now, session_id),
    )
    if c.rowcount == 0:
        print(f"No active session with id {session_id}", file=sys.stderr)
        conn.close()
        sys.exit(1)
    conn.commit()
    print(f"Session {session_id} marked finished at {now}")
    # prompt for self-rated metrics
    metrics = {}
    try:
        def ask_int(prompt, lo, hi):
            while True:
                try:
                    val = int(input(prompt))
                    if lo <= val <= hi:
                        return val
                except ValueError:
                    pass
                print(f"Enter an integer between {lo} and {hi}.")
        def ask_float(prompt, lo, hi):
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
        (session_id, 'session_summary_user', now, payload)
    )
    conn.commit()
    print(f"Logged session_summary_user for session {session_id}")
    conn.close()


def main():
    parser = argparse.ArgumentParser(prog='cultivation literature reading-session')
    sub = parser.add_subparsers(dest='command')

    p_start = sub.add_parser('start-reading', help='Start a reading session')
    p_start.add_argument('arxiv_id', type=str, help='arXiv ID of the paper')
    p_start.set_defaults(func=start_reading)

    p_list = sub.add_parser('list-open', help='List all open (unfinished) reading sessions')
    p_list.set_defaults(func=list_open)

    p_end = sub.add_parser('end-reading', help='End a reading session')
    p_end.add_argument('session_id', type=int, nargs='?', help='ID of the session to end (optional, will prompt if omitted)')
    p_end.set_defaults(func=end_reading_interactive)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)

if __name__ == '__main__':
    main()
