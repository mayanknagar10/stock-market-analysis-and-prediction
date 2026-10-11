import pandas as pd
import pytest
from core.forecasting.panel import build_panel

def test_panel_rejects_missing_sessions_instead_of_shortening_horizon(candles):
    frames={'A.NS':candles.drop(candles.index[300]),'B.NS':candles.copy()}
    manifest={'assets':{t:{'family':'india_equity','sector':'IT'} for t in frames}}
    with pytest.raises(ValueError,match='session'): build_panel(frames,manifest,1,'india_equity')
