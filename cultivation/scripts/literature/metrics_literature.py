#!/usr/bin/env python3
"""
metrics_literature.py - Aggregate literature metadata and instrumented reading session metrics into reading_stats.parquet
"""
import json
from pathlib import Path
import argparse
import logging
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema
import sqlite3


def setup_logging(level=logging.INFO):
    """
    Configures logging with a timestamped format at the specified log level.
    
    Args:
        level: The logging level to use (e.g., logging.INFO, logging.DEBUG).
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_metadata(md_dir: Path) -> pd.DataFrame:
    """
    Loads metadata from JSON files in a directory and extracts arXiv IDs and novelty scores.
    
    Each JSON file is expected to contain 'arxiv_id' and 'docinsight_novelty' fields. Files missing these fields or containing invalid JSON are skipped with a warning. Returns a DataFrame with columns 'arxiv_id' and 'docinsight_novelty_corpus'.
    """
    records = []
    for fp in md_dir.glob('*.json'):
        try:
            data = json.loads(fp.read_text())
            arxiv_id = data.get('arxiv_id')
            novelty = data.get('docinsight_novelty')
            if not arxiv_id or novelty is None:
                logging.warning(f"Skipping {fp.name}: missing arxiv_id or docinsight_novelty.")
                continue
            records.append({
                'arxiv_id': arxiv_id,
                'docinsight_novelty_corpus': float(novelty)
            })
        except Exception:
            logging.warning(f"Skipping {fp.name}: invalid JSON or data.")
            continue
    return pd.DataFrame(records)


def load_sessions(db_path: Path) -> pd.DataFrame:
    """
    Loads reading session data from a SQLite database.
    
    Connects to the specified database and retrieves session IDs, associated arXiv IDs, and session start and finish timestamps from the `sessions` table. If a database error occurs, returns an empty DataFrame with the expected columns.
    
    Args:
        db_path: Path to the SQLite database file.
    
    Returns:
        A DataFrame with columns: 'session_id', 'arxiv_id', 'started_at', and 'finished_at'.
    """
    try:
        conn = sqlite3.connect(str(db_path))
        df = pd.read_sql_query(
            'SELECT session_id, paper_id AS arxiv_id, started_at, finished_at FROM sessions',
            conn, parse_dates=['started_at', 'finished_at']
        )
        conn.close()
        return df
    except sqlite3.Error as e:
        logging.error(f"Database error in load_sessions: {e}")
        return pd.DataFrame(columns=['session_id', 'arxiv_id', 'started_at', 'finished_at'])


def load_events(db_path: Path) -> pd.DataFrame:
    """
    Loads event data from the specified SQLite database.
    
    Connects to the database at the given path and retrieves all records from the `events` table, including session ID, event type, timestamp, and payload. If a database error occurs, returns an empty DataFrame with the expected columns.
    
    Args:
        db_path: Path to the SQLite database file.
    
    Returns:
        A DataFrame containing event records with columns: 'session_id', 'event_type', 'timestamp', and 'payload'.
    """
    try:
        conn = sqlite3.connect(str(db_path))
        df = pd.read_sql_query(
            'SELECT session_id, event_type, timestamp, payload FROM events',
            conn, parse_dates=['timestamp']
        )
        conn.close()
        return df
    except sqlite3.Error as e:
        logging.error(f"Database error in load_events: {e}")
        return pd.DataFrame(columns=['session_id', 'event_type', 'timestamp', 'payload'])


def get_schema() -> DataFrameSchema:
    """
    Defines and returns the Pandera schema for the aggregated reading statistics DataFrame.
    
    The schema specifies required columns, data types, and nullability for fields such as arXiv ID, reading date, ISO week, session metrics, and various counts.
    """
    return DataFrameSchema({
        'arxiv_id': Column(pa.String, nullable=False),
        'date_read': Column(pa.DateTime, nullable=False),
        'iso_week_read': Column(pa.String, nullable=False),
        'time_spent_minutes_actual': Column(pa.Float, nullable=True),
        'docinsight_novelty_corpus': Column(pa.Float, nullable=False),
        'self_rated_novelty_personal': Column(pa.Float, nullable=True),
        'self_rated_comprehension': Column(pa.Int, nullable=True),
        'self_rated_relevance': Column(pa.Int, nullable=True),
        'flashcards_created': Column(pa.Int, nullable=True),
        'docinsight_queries_during_session': Column(pa.Int, nullable=True),
        'notes_length_chars_final': Column(pa.Int, nullable=True),
        'pages_viewed_count': Column(pa.Int, nullable=True),
        'highlights_made_count': Column(pa.Int, nullable=True)
    }, coerce=True)


import argparse

def main():
    """
    Aggregates literature metadata and reading session metrics into a Parquet file.
    
    Parses command-line arguments for metadata directory, SQLite database path, and output file location. Loads metadata, session, and event data, then processes each completed reading session to extract reading dates, ISO weeks, user metrics, and event counts. Combines these with metadata novelty scores and writes the validated results to a Parquet file. Logs progress and errors throughout execution.
    """
    parser = argparse.ArgumentParser(description='Aggregate literature metadata and reading sessions')
    parser.add_argument(
        '--metadata-dir',
        type=str,
        default=str(Path(__file__).parent.parent.parent / 'literature' / 'metadata'),
        help='Directory containing metadata JSON files'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=str(Path(__file__).parent.parent.parent / 'literature' / 'db.sqlite'),
        help='Path to SQLite database with session data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(Path(__file__).parent.parent.parent / 'literature' / 'reading_stats.parquet'),
        help='Output path for Parquet file'
    )
    args = parser.parse_args()

    md_dir  = Path(args.metadata_dir)
    db_path = Path(args.db_path)
    out     = Path(args.output)
    logging.info('Loading metadata and session data')
    metadata_df = load_metadata(md_dir)
    sessions_df = load_sessions(db_path)
    events_df = load_events(db_path)
    records = []
    for _, row in sessions_df.iterrows():
        sess_id = row['session_id']
        if pd.isna(row['finished_at']):
            continue
        finished = row['finished_at']
        date_read = finished.date()
        iso_week = finished.strftime('%Y-W%V')
        # parse session_summary_user
        evt = events_df[events_df['session_id']==sess_id]
        summary = evt[evt['event_type']=='session_summary_user']
        metrics = {}
        if not summary.empty:
            try:
                metrics = json.loads(summary.iloc[-1]['payload'])
            except Exception:
                metrics = {}
        records.append({
            'arxiv_id': row['arxiv_id'],
            'date_read': pd.to_datetime(date_read),
            'iso_week_read': iso_week,
            'time_spent_minutes_actual': metrics.get('actual_time_spent_minutes'),
            'docinsight_novelty_corpus': arxiv_to_novelty.get(row['arxiv_id']),
            'self_rated_novelty_personal': metrics.get('self_rated_novelty_personal'),
            'self_rated_comprehension': metrics.get('self_rated_comprehension'),
            'self_rated_relevance': metrics.get('self_rated_relevance'),
            'flashcards_created': len(evt[evt['event_type']=='flashcard_staged']),
            'docinsight_queries_during_session': len(evt[evt['event_type']=='docinsight_query']),
            'notes_length_chars_final': None,
            'pages_viewed_count': None,
            'highlights_made_count': None
        })
    df = pd.DataFrame(records)
    schema = get_schema()
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        logging.error(f"Schema validation error: {e}")
        raise
    df.to_parquet(out, index=False)
    logging.info(f"Wrote enriched reading stats to {out}")
    print(f"Wrote reading stats to {out}")


if __name__ == '__main__':
    main()
