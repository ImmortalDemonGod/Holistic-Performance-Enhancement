#!/usr/bin/env python3
"""
metrics_literature.py - Aggregate literature metadata into reading_stats.parquet
"""
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema
import logging


def load_metadata(md_dir: Path) -> pd.DataFrame:
    records = []
    for fp in md_dir.glob('*.json'):
        try:
            data = json.loads(fp.read_text())
            imported = data.get('imported_at')
            novelty = data.get('docinsight_novelty')
            if imported is None or novelty is None:
                logging.warning(f"Skipping {fp.name}: missing imported_at or docinsight_novelty.")
                continue
            dt = datetime.fromisoformat(imported)
            week = dt.strftime('%Y-W%V')
            records.append({'iso_week': week, 'novelty': float(novelty)})
        except Exception:
            logging.warning(f"Skipping {fp.name}: invalid JSON or data.")
            continue
    return pd.DataFrame(records)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['iso_week', 'papers_read', 'avg_novelty'])
    grouped = df.groupby('iso_week').agg(
        papers_read=pd.NamedAgg(column='novelty', aggfunc='count'),
        avg_novelty=pd.NamedAgg(column='novelty', aggfunc='mean')
    )
    return grouped.reset_index()


def get_schema() -> DataFrameSchema:
    return DataFrameSchema({
        "iso_week": Column(pa.String),
        "papers_read": Column(pa.Int, nullable=False),
        "avg_novelty": Column(pa.Float, nullable=True),
        "minutes_spent": Column(pa.Float, nullable=True)
    }, coerce=True)


def main():
    base = Path(__file__).parent.parent / 'literature' / 'metadata'
    out = Path(__file__).parent.parent / 'literature' / 'reading_stats.parquet'
    df = load_metadata(base)
    stats = aggregate(df)

    schema = get_schema()
    try:
        schema.validate(stats, lazy=True)
    except pa.errors.SchemaErrors as e:
        logging.error(f"Schema validation error on reading_stats.parquet: {e}")
        raise

    stats.to_parquet(out, index=False)
    logging.info("INFO: 'minutes_spent' in reading_stats.parquet is currently a placeholder and not derived from actual reading time.")
    print(f"Wrote reading stats to {out}")


if __name__ == '__main__':
    main()
