def test_run_metrics_sample():
    import pandas as pd
    from cultivation.scripts.running.metrics import run_metrics
    df = pd.read_csv('tests/data/sample_run.csv')
    res = run_metrics(df)
    assert 'efficiency_factor' in res
    assert res['decoupling_%'] >= 0

def test_zone_mapping_sanity():
    import pandas as pd
    from cultivation.scripts.running.metrics import compute_training_zones, load_personal_zones
    zones = load_personal_zones()
    # Edge cases: exact boundaries and NaN
    hr_cases = [158, 166, 175, 184, 201, float('nan')]
    pace_cases = [9.4, 9.0, 8.5, 8.1, 8.0, float('nan')]
    hr_series = pd.Series(hr_cases)
    pace_series = pd.Series(pace_cases)
    zone_hr, zone_pace, zone_effective = compute_training_zones(hr_series, pace_series, zones)
    # Canonical names from YAML
    expected_hr = ["Z1 (Recovery)", "Z2 (Aerobic)", "Z3 (Tempo)", "Z4 (Threshold)", "Z5 (VO2max)", None]
    expected_pace = ["Z1 (Recovery)", "Z2 (Aerobic)", "Z3 (Tempo)", "Z4 (Threshold)", "Z5 (VO2max)", None]
    assert list(zone_hr) == expected_hr
    assert list(zone_pace) == expected_pace

def test_decoupling_zero_for_constant():
    import pandas as pd
    from cultivation.scripts.running.metrics import run_metrics
    import numpy as np
    # Synthetic: constant HR and speed
    df = pd.DataFrame({
        'dist': np.linspace(0, 1000, 61),
        'dt': np.full(61, 10),
        'hr': np.full(61, 150),
        'pace_sec_km': np.full(61, 400),
        'time_delta_s': np.full(61, 10),
        'latitude': np.full(61, 40.0),
        'longitude': np.full(61, -105.0),
    }, index=pd.date_range("2025-01-01", periods=61, freq="10s"))
    res = run_metrics(df)
    assert abs(res['decoupling_%']) < 1e-6