import json
from datetime import datetime
import pandas as pd
import pytest

from cultivation.scripts.literature.metrics_literature import load_metadata, aggregate

def test_load_metadata(tmp_path):
    # Prepare sample metadata files
    md_dir = tmp_path
    records = [
        {'imported_at': '2025-05-01T12:00:00', 'docinsight_novelty': 0.5},
        {'imported_at': '2025-05-05T08:30:00', 'docinsight_novelty': 0.9},
        # invalid record missing fields
        {'imported_at': None, 'docinsight_novelty': 0.2},
    ]
    for i, rec in enumerate(records):
        p = md_dir / f"{i}.json"
        p.write_text(json.dumps(rec))
 
    df = load_metadata(md_dir)
    assert isinstance(df, pd.DataFrame)
    # Only two valid records
    assert len(df) == 2
    assert set(df.columns) == {'iso_week', 'novelty'}
    # Weeks computed correctly
    expected_weeks = set(
        datetime.fromisoformat(rec['imported_at']).strftime('%Y-W%V')
        for rec in records[:2]
    )
    assert set(df['iso_week']) == expected_weeks

def test_aggregate_empty():
    df = pd.DataFrame(columns=['iso_week', 'novelty'])
    out = aggregate(df)
    assert isinstance(out, pd.DataFrame)
    assert out.empty
    assert list(out.columns) == ['iso_week', 'papers_read', 'avg_novelty']

def test_aggregate_non_empty():
    df = pd.DataFrame([
        {'iso_week': '2025-W18', 'novelty': 0.5},
        {'iso_week': '2025-W18', 'novelty': 0.7},
        {'iso_week': '2025-W19', 'novelty': 0.9},
    ])
    out = aggregate(df)
    assert len(out) == 2
    row18 = out[out['iso_week'] == '2025-W18'].iloc[0]
    assert row18['papers_read'] == 2
    assert pytest.approx(row18['avg_novelty'], rel=1e-3) == 0.6
    row19 = out[out['iso_week'] == '2025-W19'].iloc[0]
    assert row19['papers_read'] == 1
    assert pytest.approx(row19['avg_novelty'], rel=1e-3) == 0.9
