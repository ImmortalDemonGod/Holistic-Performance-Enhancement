def test_run_metrics_sample():
    import pandas as pd
    from cultivation.scripts.running.metrics import run_metrics
    df = pd.read_csv('tests/data/sample_run.csv')
    res = run_metrics(df)
    assert 'efficiency_factor' in res
    assert res['decoupling_%'] >= 0

def test_decoupling_zero_for_constant():
    import pandas as pd
    from cultivation.scripts.running.metrics import run_metrics
    import numpy as np
    # Synthetic: constant HR and speed
    df = pd.DataFrame({
        'dist': np.linspace(0, 1000, 61),
        'dt': np.full(61, 10),
        'hr': np.full(61, 150),
        'pace_sec_km': np.full(61, 360),
        'time_delta_s': np.full(61, 10),
        'latitude': np.linspace(40.0, 40.01, 61),
        'longitude': np.linspace(-105.0, -104.99, 61),
    }, index=pd.date_range("2025-01-01", periods=61, freq="10s"))
    res = run_metrics(df)
    assert abs(res['decoupling_%']) < 1e-6

def test_zone_mapping_sanity():
    import pandas as pd
    from cultivation.scripts.running.metrics import compute_training_zones, load_personal_zones
    import numpy as np
    # Create HR and pace arrays spanning the range
    zones = load_personal_zones()
    zone_names = set(zones.keys())

    # use Series, not ndarray
    hr_s = pd.Series(np.linspace(100, 190, 10))
    pace_s = pd.Series(np.linspace(4.0, 8.0, 10) * 60)  # sec / km

    zone_hr, zone_pace, zone_eff = compute_training_zones(hr_s, pace_s, zones)

    assert set(z for z in zone_hr.dropna()) <= zone_names
    # Effective zones should be in the union
    all_zones = zone_names
    assert all(z in all_zones or z is None for z in zone_eff)