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
    }, index=pd.date_range("2025-01-01", periods=61, freq="10s"))
    res = run_metrics(df)
    assert abs(res['decoupling_%']) < 1e-6