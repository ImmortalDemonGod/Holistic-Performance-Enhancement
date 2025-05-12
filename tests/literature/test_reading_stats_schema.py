import pandas as pd
import pytest
import pandera as pa
from cultivation.scripts.literature.metrics_literature import get_schema

def test_valid_stats_df_passes():
    df = pd.DataFrame([
        {"iso_week": "2025-W19", "papers_read": 3, "avg_novelty": 0.85, "minutes_spent": None},
        {"iso_week": "2025-W20", "papers_read": 2, "avg_novelty": 0.92, "minutes_spent": 10}
    ])
    schema = get_schema()
    schema.validate(df)

def test_missing_column_fails():
    df = pd.DataFrame([
        {"iso_week": "2025-W19", "papers_read": 3, "avg_novelty": 0.85} # missing minutes_spent
    ])
    schema = get_schema()
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(df)

def test_wrong_type_fails():
    df = pd.DataFrame([
        {"iso_week": "2025-W19", "papers_read": "three", "avg_novelty": 0.85, "minutes_spent": None}
    ])
    schema = get_schema()
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(df)
