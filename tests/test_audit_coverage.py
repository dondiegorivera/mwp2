import pandas as pd
from scripts.audit_outliers import compute_metrics


def test_coverage_uses_true_column_when_present(tmp_path, monkeypatch):
    # Fake single-horizon predictions with explicit truth
    df = pd.DataFrame(
        {
            "1d@h1": [0.0, 0.1, -0.1],
            "1d_lower@h1": [-0.05, -0.05, -0.15],
            "1d_upper@h1": [0.05, 0.15, -0.05],
            "1d@h1_true": [0.0, 0.12, -0.12],
        }
    )
    m = compute_metrics(df, "1d")
    # second is inside, third is inside, first is inside => 3/3
    assert abs(m["coverage_90"] - 1.0) < 1e-6
