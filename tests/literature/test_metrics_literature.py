import json
import pandas as pd

from cultivation.scripts.literature.metrics_literature import load_metadata, aggregate, get_schema

def test_load_metadata(tmp_path):
    # Create sample metadata files with arxiv_id and docinsight_novelty
    records = [
        {"arxiv_id": "1234.5678", "docinsight_novelty": 0.8},
        {"arxiv_id": "2345.6789", "docinsight_novelty": 0.6},
    ]
    for i, rec in enumerate(records):
        with open(tmp_path / f"{i}.json", "w") as f:
            json.dump(rec, f)
    df = load_metadata(tmp_path)
    assert not df.empty
    # load_metadata only produces these two columns
    assert set(df.columns) == {"arxiv_id", "docinsight_novelty_corpus"}

def test_aggregate_empty():
    metadata_df = pd.DataFrame(columns=['arxiv_id', 'docinsight_novelty_corpus'])
    sessions_df = pd.DataFrame(columns=['session_id', 'arxiv_id', 'started_at', 'finished_at'])
    events_df = pd.DataFrame(columns=['session_id', 'event_type', 'timestamp', 'payload'])

    out_df = aggregate(metadata_df, sessions_df, events_df)

    expected_cols = set(get_schema().columns.keys())
    assert out_df.empty
    assert set(out_df.columns) == expected_cols

def test_aggregate_non_empty():
    metadata_df = pd.DataFrame([
        {'arxiv_id': '0001.0001', 'docinsight_novelty_corpus': 0.5},
        {'arxiv_id': '0001.0002', 'docinsight_novelty_corpus': 0.7},
        {'arxiv_id': '0002.0001', 'docinsight_novelty_corpus': 0.9},
    ])

    sessions_df = pd.DataFrame([
        {
            'session_id': 'sess1',
            'arxiv_id': '0001.0001',
            'started_at': pd.Timestamp('2025-05-01 10:00:00'),
            'finished_at': pd.Timestamp('2025-05-01 10:30:00') # Week 18
        },
        {
            'session_id': 'sess2',
            'arxiv_id': '0001.0002',
            'started_at': pd.Timestamp('2025-05-02 11:00:00'),
            'finished_at': pd.Timestamp('2025-05-02 11:45:00') # Week 18
        },
        {
            'session_id': 'sess3',
            'arxiv_id': '0002.0001',
            'started_at': pd.Timestamp('2025-05-08 14:00:00'),
            'finished_at': pd.Timestamp('2025-05-08 15:00:00') # Week 19
        },
        {
            'session_id': 'sess4', # Session with no corresponding metadata
            'arxiv_id': '9999.9999',
            'started_at': pd.Timestamp('2025-05-09 10:00:00'),
            'finished_at': pd.Timestamp('2025-05-09 10:30:00')
        },
        {
            'session_id': 'sess5', # Session with no finish time
            'arxiv_id': '0001.0001',
            'started_at': pd.Timestamp('2025-05-10 10:00:00'),
            'finished_at': pd.NaT
        }
    ])

    events_df = pd.DataFrame([
        {
            'session_id': 'sess1',
            'event_type': 'session_summary_user',
            'timestamp': pd.Timestamp('2025-05-01 10:30:00'),
            'payload': json.dumps({'self_rated_novelty_personal': 3, 'self_rated_comprehension': 4, 'flashcards_created': 0})
        },
        # sess2 has no summary event
        {
            'session_id': 'sess3',
            'event_type': 'session_summary_user',
            'timestamp': pd.Timestamp('2025-05-08 15:00:00'),
            'payload': json.dumps({'self_rated_novelty_personal': 5, 'self_rated_comprehension': 5, 'flashcards_created': 2, 'notes_length_chars_final': 500})
        },
        {
            'session_id': 'sess3', # Test with multiple summary events for one session (should pick last)
            'event_type': 'session_summary_user',
            'timestamp': pd.Timestamp('2025-05-08 15:01:00'), # Later timestamp
            'payload': json.dumps({'self_rated_novelty_personal': 4, 'self_rated_comprehension': 4, 'flashcards_created': 3})
        }
    ])

    out_df = aggregate(metadata_df, sessions_df, events_df)

    expected_cols = set(get_schema().columns.keys())
    assert set(out_df.columns) == expected_cols
    
    # sess5 (no finish time) and sess4 (no metadata) should be excluded
    assert len(out_df) == 3 

    out_df = out_df.sort_values(by='arxiv_id').reset_index(drop=True)

    # Check sess1 (arxiv_id: 0001.0001)
    assert out_df.loc[0, 'arxiv_id'] == '0001.0001'
    assert out_df.loc[0, 'iso_week_read'] == '2025-W18'
    assert out_df.loc[0, 'docinsight_novelty_corpus'] == 0.5
    assert out_df.loc[0, 'time_spent_minutes_actual'] == 30.0
    assert out_df.loc[0, 'self_rated_novelty_personal'] == 3
    assert out_df.loc[0, 'self_rated_comprehension'] == 4
    assert out_df.loc[0, 'flashcards_created'] == 0
    assert pd.isna(out_df.loc[0, 'notes_length_chars_final'])

    # Check sess2 (arxiv_id: 0001.0002)
    assert out_df.loc[1, 'arxiv_id'] == '0001.0002'
    assert out_df.loc[1, 'iso_week_read'] == '2025-W18'
    assert out_df.loc[1, 'docinsight_novelty_corpus'] == 0.7
    assert out_df.loc[1, 'time_spent_minutes_actual'] == 45.0
    assert pd.isna(out_df.loc[1, 'self_rated_novelty_personal'])
    assert pd.isna(out_df.loc[1, 'self_rated_comprehension'])
    assert out_df.loc[1, 'flashcards_created'] == 0

    # Check sess3 (arxiv_id: 0002.0001) - should use the latest event payload
    assert out_df.loc[2, 'arxiv_id'] == '0002.0001'
    assert out_df.loc[2, 'iso_week_read'] == '2025-W19'
    assert out_df.loc[2, 'docinsight_novelty_corpus'] == 0.9
    assert out_df.loc[2, 'time_spent_minutes_actual'] == 60.0
    assert out_df.loc[2, 'self_rated_novelty_personal'] == 4 # From the last event
    assert out_df.loc[2, 'self_rated_comprehension'] == 4 # From the last event
    assert out_df.loc[2, 'flashcards_created'] == 3 # From the last event
    assert pd.isna(out_df.loc[2, 'notes_length_chars_final']) # Original event for sess3 had it, but the latest one didn't

# Test the main function and schema validation (basic)
