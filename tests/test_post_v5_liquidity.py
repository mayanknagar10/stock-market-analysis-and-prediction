import pytest
from test_post_v5_archive import source
from core.data.contracts import DataSnapshot

def test_liquidity_diagnostics_preserve_zero_volume_and_missing_price_flags():
    from research.post_v5.diagnostics import liquidity_diagnostics
    snapshot=source();frame=snapshot.frame;frame.iloc[-1,frame.columns.get_loc('Volume')]=0.
    result=liquidity_diagnostics(DataSnapshot(frame,snapshot.metadata))
    assert result['n_observed_sessions']==2
    assert result['positive_volume_fraction']==pytest.approx(.5)
    assert result['median_positive_volume_turnover_proxy']==pytest.approx(10000.)
    assert result['n_usable_price_volume_sessions']==1
    assert result['selection_effect']=='none; frozen membership retained'
