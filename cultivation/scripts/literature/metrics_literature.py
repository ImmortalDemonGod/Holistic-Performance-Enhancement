#!/usr/bin/env python3
"""
metrics_literature.py - Aggregate literature metadata into reading_stats.parquet
"""
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


def load_metadata(md_dir: Path) -> pd.DataFrame:
    records = []
    for fp in md_dir.glob('*.json'):
        try:
            data = json.loads(fp.read_text())
            imported = data.get('imported_at')
            novelty = data.get('docinsight_novelty')
            if imported is None or novelty is None:
                continue
            dt = datetime.fromisoformat(imported)
            week = dt.strftime('%Y-W%V')
            records.append({'iso_week': week, 'novelty': float(novelty)})
        except Exception:
            continue
    return pd.DataFrame(records)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['iso_week', 'papers_read', 'avg_novelty'])
    grouped = df.groupby('iso_week')['novelty']
    return grouped.agg(papers_read='count', avg_novelty='mean').reset_index()


def main():
    base = Path(__file__).parent.parent / 'literature' / 'metadata'
    out = Path(__file__).parent.parent / 'literature' / 'reading_stats.parquet'
    df = load_metadata(base)
    stats = aggregate(df)
    stats.to_parquet(out, index=False)
    print(f"Wrote reading stats to {out}")


if __name__ == '__main__':
    main()
