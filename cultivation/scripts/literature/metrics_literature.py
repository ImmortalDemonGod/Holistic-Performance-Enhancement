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
    """Configure logging with consistent format."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_metadata(md_dir: Path) -> pd.DataFrame:
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


def aggregate(metadata_df: pd.DataFrame, sessions_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates metadata, session, and event data into a structured DataFrame."""
    
    schema_cols = list(get_schema().columns.keys())

    if sessions_df.empty:
        return pd.DataFrame(columns=schema_cols)

    arxiv_to_novelty = metadata_df.set_index('arxiv_id')['docinsight_novelty_corpus'].to_dict()
    records = []
    for _, row in sessions_df.iterrows():
        sess_id = row['session_id']
        arxiv_id = row['arxiv_id']

        # Skip session if its arxiv_id is not in metadata_df (novelty would be None)
        if arxiv_id not in arxiv_to_novelty:
            logging.debug(f"Skipping session {sess_id} for arxiv_id {arxiv_id} as it's not in metadata.")
            continue

        if pd.isna(row['finished_at']):
            logging.debug(f"Skipping session {sess_id} due to missing finished_at time.")
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
                payload_str = summary.iloc[-1]['payload']
                if isinstance(payload_str, str):
                    metrics = json.loads(payload_str)
                elif isinstance(payload_str, dict): # Already a dict, no need to parse
                    metrics = payload_str
                else:
                    logging.warning(f"Session {sess_id} summary payload is not a string or dict: {type(payload_str)}")
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse session_summary_user payload for session {sess_id}: {e}. Payload: {summary.iloc[-1]['payload']}")
            except Exception as e:
                logging.error(f"Unexpected error parsing summary for session {sess_id}: {e}")

        fc_from_payload = metrics.get('flashcards_created')
        flashcards_count = fc_from_payload if fc_from_payload is not None else len(evt[evt['event_type']=='flashcard_staged'])

        records.append({
            'arxiv_id': row['arxiv_id'],
            'date_read': pd.to_datetime(date_read),
            'iso_week_read': iso_week,
            'time_spent_minutes_actual': (finished - row['started_at']).total_seconds() / 60.0 if pd.notna(row['started_at']) and pd.notna(finished) else None,
            'docinsight_novelty_corpus': arxiv_to_novelty.get(row['arxiv_id']),
            'self_rated_novelty_personal': metrics.get('self_rated_novelty_personal'),
            'self_rated_comprehension': metrics.get('self_rated_comprehension'),
            'self_rated_relevance': metrics.get('self_rated_relevance'),
            'flashcards_created': flashcards_count,
            'docinsight_queries_during_session': len(evt[evt['event_type']=='docinsight_query']),
            'notes_length_chars_final': metrics.get('notes_length_chars_final'), 
            'pages_viewed_count': metrics.get('pages_read_count'), 
            'highlights_made_count': len(evt[evt['event_type']=='highlight_created']) 
        })
    
    if not records:
        return pd.DataFrame(columns=schema_cols)
        
    return pd.DataFrame(records)


# Constants for output columns - consider defining these via the schema directly if possible
def main():
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
    
    # Call the new aggregate function
    df = aggregate(metadata_df, sessions_df, events_df)

    # Validate and write the DataFrame
    if df.empty:
        logging.info("No data to process after aggregation. Output file will not be created.")
        print("No data to process after aggregation. Output file will not be created.")
        # Still, we might want to ensure an empty parquet with schema is written if that's desired
        # For now, exiting if no records post-aggregation. Or, handle schema for empty df.
        # Create an empty DataFrame with the correct columns if df is empty
        # to ensure schema validation doesn't fail and an empty parquet is written as expected.
        empty_cols = get_schema().columns.keys()
        df = pd.DataFrame(columns=empty_cols) 
        # Ensure dtypes for empty df match schema to prevent validation issues
        for col, col_obj in get_schema().columns.items():
            if col in df.columns:
                 # Attempt to cast to the pandera type's corresponding pandas dtype
                try:
                    if str(col_obj.dtype) == 'datetime64[ns]': # Pandera uses 'datetime64[ns]' for DateTime
                        df[col] = pd.to_datetime(df[col])
                    elif str(col_obj.dtype) == 'string': # For Pandera String
                        df[col] = df[col].astype(pd.StringDtype())
                    else: # For float, int etc.
                        df[col] = df[col].astype(str(col_obj.dtype).lower())
                except Exception as e:
                    logging.debug(f"Could not cast empty column {col} to {col_obj.dtype}: {e}")
                    pass # Keep it as object if casting fails for empty

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
    setup_logging() # Add setup_logging call
    main()
