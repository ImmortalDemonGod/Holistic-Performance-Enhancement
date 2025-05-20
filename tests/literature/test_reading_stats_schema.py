import pandas as pd
import pytest
import pandera as pa

def get_weekly_reading_stats_schema():
    """Defines the Pandera schema for weekly aggregated reading statistics."""
    return pa.DataFrameSchema({
        "iso_week": pa.Column(str, nullable=False),
        "papers_read": pa.Column(int, checks=pa.Check.ge(0), nullable=False),
        "avg_novelty": pa.Column(float, checks=[pa.Check.ge(0), pa.Check.le(1)], nullable=True),
        "minutes_spent": pa.Column("Int64", checks=pa.Check.ge(0), nullable=True, coerce=True)
    }, strict=True, ordered=True, coerce=True)

def test_valid_stats_df_passes():
    df = pd.DataFrame([
        {"iso_week": "2025-W19", "papers_read": 3, "avg_novelty": 0.85, "minutes_spent": None},
        {"iso_week": "2025-W20", "papers_read": 2, "avg_novelty": 0.92, "minutes_spent": 10}
    ])
    schema = get_weekly_reading_stats_schema()
    schema.validate(df)

def test_missing_column_fails():
    df = pd.DataFrame([
        {"iso_week": "2025-W19", "papers_read": 3, "avg_novelty": 0.85} # missing minutes_spent
    ])
    schema = get_weekly_reading_stats_schema()
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(df)

def test_wrong_type_fails():
    df = pd.DataFrame([
        {"iso_week": "2025-W19", "papers_read": "three", "avg_novelty": 0.85, "minutes_spent": None}
    ])
    schema = get_weekly_reading_stats_schema()
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(df)
