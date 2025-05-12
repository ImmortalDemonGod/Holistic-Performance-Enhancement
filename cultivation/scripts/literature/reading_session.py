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
    base = Path(__file__).parent.parent / 'literature'
    base.mkdir(parents=True, exist_ok=True)
    return base / 'db.sqlite'

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
    now = datetime.utcnow().isoformat() + 'Z'
    c = conn.cursor()
    c.execute('INSERT INTO sessions(paper_id, started_at) VALUES (?, ?)',
              (args.arxiv_id, now))
    conn.commit()
    session_id = c.lastrowid
    print(f"Started session {session_id} for paper {args.arxiv_id} at {now}")
    conn.close()


def end_reading(args):
    db = get_db_path()
    conn = sqlite3.connect(db)
    init_db(conn)
    c = conn.cursor()
    # update session finished_at
    now = datetime.utcnow().isoformat() + 'Z'
    c.execute('UPDATE sessions SET finished_at = ? WHERE session_id = ?', (now, args.session_id))
    conn.commit()
    print(f"Session {args.session_id} marked finished at {now}")
    # prompt for self-rated metrics
    metrics = {}
    try:
        metrics['self_rated_comprehension'] = int(input('Self-rated comprehension (0-5): '))
        metrics['self_rated_relevance'] = int(input('Self-rated relevance (0-5): '))
        metrics['self_rated_novelty_personal'] = float(input('Self-rated novelty personal (0-1): '))
        metrics['actual_time_spent_minutes'] = float(input('Actual time spent (minutes): '))
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
