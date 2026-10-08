import numpy as np
import pandas as pd
import pytest
from core.data.contracts import SourceMetadata,DataSnapshot
from core.forecasting.relative import relative_targets

def test_relative_targets_use_session_identity_not_delayed_cutoff_date():
    dates=pd.DatetimeIndex(['2024-11-03','2024-11-04'],tz='Asia/Kolkata')
    frame=pd.DataFrame({'Adj Close':[100.,105.]},index=dates.tz_convert('UTC'))
    metadata=SourceMetadata(provider_id='fixture',provider_version='1',symbol='INDEX',family='research_reference',
        exchange_timezone='Asia/Kolkata',interval='1d',adjustment_policy='fixture',
        source_timestamp=frame.index[-1].isoformat(),available_at='2024-11-05T00:00Z',fetched_at='2024-11-05T00:00Z',vintage_at='2024-11-05T00:00Z')
    meta=pd.DataFrame({'ticker':['TEST.NS'],'feature_time':pd.to_datetime(['2024-11-04T00:20Z'],utc=True),
        'label_end':pd.to_datetime(['2024-11-04T10:20Z'],utc=True),
        'origin_session':['2024-11-03'],'maturity_session':['2024-11-04']},index=[40])
    result=relative_targets(meta,pd.Series([np.log(1.1)]),{'market_india':DataSnapshot(frame,metadata)}, {},'india_equity')
    assert result['market'][0]==pytest.approx(np.log(1.1)-np.log(1.05))
