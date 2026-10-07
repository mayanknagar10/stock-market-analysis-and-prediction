import numpy as np
import pandas as pd
from core.indicators import build_ml_features, build_ml_features_batch, generate_signals
from core.risk_metrics import drawdown_analysis, var_historical, cvar
from core.screen_backtest import _evaluate_technical_filters_at

def test_features_do_not_change_when_future_candles_are_appended(candles):
    full = build_ml_features(candles)
    prefix = build_ml_features(candles.iloc[:310])
    pd.testing.assert_frame_equal(full.iloc[:310], prefix)

def test_universal_features_are_price_scale_invariant(candles):
    scaled = candles.copy()
    scaled[['Open', 'High', 'Low', 'Close']] *= 1000
    np.testing.assert_allclose(build_ml_features(candles), build_ml_features(scaled),
                               rtol=1e-8, atol=1e-7, equal_nan=True)

def test_batch_features_match_single_path(candles):
    single = build_ml_features(candles)
    batch = build_ml_features_batch(*[candles[[c]].rename(columns={c: 'path'})
                                      for c in ['Close','Open','High','Low','Volume']])
    for name in single.columns:
        np.testing.assert_allclose(single[name], batch[name]['path'], rtol=1e-8, atol=1e-7, equal_nan=True)

def test_drawdown_and_historical_tail_risk():
    _, loss, duration = drawdown_analysis(pd.Series([100., 120., 90., 108., 130.]))
    assert loss == -.25
    assert duration == 2
    returns = pd.Series([-.1, -.05, 0., .01, .02])
    assert np.isclose(var_historical(returns, .8), .06)
    assert np.isclose(cvar(returns, .8), .1)

def test_screen_filter_excludes_later_price_changes(candles):
    changed = candles.copy()
    changed.iloc[301:, changed.columns.get_loc('Close')] *= 10
    args = (300, 0, 100, 'Price > EMA20', 0)
    assert _evaluate_technical_filters_at(candles, *args) == _evaluate_technical_filters_at(changed, *args)

def test_signals_remain_complete(candles):
    result = generate_signals(candles)
    assert result['composite'] in {'BUY','SELL','NEUTRAL','STRONG BUY','STRONG SELL','HOLD'}
    assert len(result['indicators']) == 8
    assert 0 <= result['buy_count'] <= 8
