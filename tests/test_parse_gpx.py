import os
import pytest
from cultivation.scripts.running import parse_run_files

def test_parse_gpx_handles_missing_hr_cadence():
    gpx_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_missing_hr_cadence.gpx')
    df = parse_run_files.parse_gpx(gpx_path)
    # Should not raise, and missing HR/cadence should be NaN
    assert 'hr' in df.columns
    assert 'cadence' in df.columns
    assert df['hr'].isnull().any() or df['cadence'].isnull().any()
