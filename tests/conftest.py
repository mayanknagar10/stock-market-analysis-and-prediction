import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

@pytest.fixture
def candles():
    rng = np.random.default_rng(17)
    idx = pd.bdate_range('2020-01-01', periods=420)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.014, len(idx))))
    open_ = close * np.exp(rng.normal(0, 0.004, len(idx)))
    return pd.DataFrame({'Open': open_, 'High': np.maximum(open_, close) * 1.01,
                         'Low': np.minimum(open_, close) * .99, 'Close': close,
                         'Volume': rng.integers(10000, 1000000, len(idx))}, index=idx)


@pytest.fixture(autouse=True)
def isolate_legacy_outcome_test_reports(request,tmp_path,monkeypatch):
    """Legacy resolver fixtures must not write synthetic evidence into real reports."""
    if request.node.name.startswith("test_resolver_keeps_bad_or_premature_endpoints_pending"):
        import research.post_v5.outcomes as outcomes
        original=outcomes.ROOT
        target=tmp_path/"report-root"
        universe=target/"config/research"/outcomes.STUDY/"UNIVERSE.json"
        universe.parent.mkdir(parents=True,exist_ok=True)
        universe.write_bytes((original/"config/research"/outcomes.STUDY/"UNIVERSE.json").read_bytes())
        monkeypatch.setattr(outcomes,"ROOT",target)
    yield
