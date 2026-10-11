import pandas as pd
import pytest
from core.forecasting.context import research_feature_frame

def test_research_context_excludes_same_date_later_market_data(candles):
    reference=candles.copy()
    reference.index=reference.index.tz_localize('UTC')
    reference['Adj Close']=reference.Close
    features,meta=research_feature_frame(candles,'TEST.NS','india_equity',{'nasdaq':reference},{},research_only=True)
    changed=reference.copy()
    changed.iloc[-1,changed.columns.get_loc('Adj Close')]*=2
    newer,_=research_feature_frame(candles,'TEST.NS','india_equity',{'nasdaq':changed},{},research_only=True)
    assert features.nasdaq_ret_1d.iloc[-1]==newer.nasdaq_ret_1d.iloc[-1]
    assert meta['point_in_time_verified'] is False

def test_context_reconstruction_cannot_enter_verified_mode(candles):
    with pytest.raises(ValueError,match='research'):
        research_feature_frame(candles,'TEST.NS','india_equity',{}, {},research_only=False)

def test_unknown_sector_is_explicit_without_invented_sector_movement(candles):
    features,meta=research_feature_frame(candles,'TEST.NS','india_equity',{}, {},research_only=True)
    assert meta['sector_status']=='UNAVAILABLE'
    assert not any(c.startswith('sector_') for c in features)
