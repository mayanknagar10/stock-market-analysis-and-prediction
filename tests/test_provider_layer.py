import pandas as pd
import pytest
from core.data.providers import YahooMarketDataProvider, ProviderUnavailable

def test_provider_failure_is_explicit_and_never_substitutes_raw_prices():
    def fail(*args,**kwargs): raise OSError('feed down')
    with pytest.raises(ProviderUnavailable): YahooMarketDataProvider(transport=fail).capture_history('TEST.NS')

def test_provider_preserves_actions_timezone_and_unverified_vintage(candles):
    frame=candles.copy()
    frame.index=frame.index.tz_localize('Asia/Kolkata')
    frame['Adj Close']=frame.Close
    frame['Dividends']=0.
    frame['Stock Splits']=0.
    provider=YahooMarketDataProvider(transport=lambda *a,**kw:frame)
    data=provider.capture_history('TEST.NS')
    assert data.metadata.provider_id=='yahoo/yfinance'
    assert data.metadata.interval=='1d'
    assert data.metadata.availability_verified is False
    assert data.metadata.revision_history_verified is False
    assert data.frame.index.tz is not None
    assert {'Close','Adj Close','Dividends','Stock Splits'}.issubset(data.frame)

def test_missing_adjusted_close_does_not_become_a_raw_label(candles):
    frame=candles.copy()
    frame.index=frame.index.tz_localize('UTC')
    with pytest.raises(ProviderUnavailable):
        YahooMarketDataProvider(transport=lambda *a,**kw:frame).capture_history('TEST.NS')


def test_cached_partial_bar_cannot_become_completed_after_fetch():
    from core.data.contracts import SourceMetadata,DataSnapshot
    from core.forecasting.service import adjusted_frame
    dates=pd.DatetimeIndex(['2023-12-29','2024-01-02'],tz='America/New_York')
    frame=pd.DataFrame({'Open':[100.,100.],'High':[101.,110.],'Low':[99.,99.],'Close':[100.,105.],
        'Adj Close':[100.,105.],'Volume':[100.,200.],'Dividends':0.,'Stock Splits':0.},index=dates.tz_convert('UTC'))
    metadata=SourceMetadata(provider_id='fixture',provider_version='1',symbol='TEST',family='us_equity',
        exchange_timezone='America/New_York',interval='1d',adjustment_policy='fixture',
        source_timestamp='2024-01-02T05:00Z',available_at='2024-01-02T20:00Z',
        fetched_at='2024-01-02T20:00Z',vintage_at='2024-01-02T20:00Z')
    raw,adjusted=adjusted_frame(DataSnapshot(frame,metadata),'2024-01-02T22:00Z')
    assert len(raw)==1
    assert adjusted.index[-1]==pd.Timestamp('2023-12-29')
