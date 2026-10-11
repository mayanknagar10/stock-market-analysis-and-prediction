import pandas as pd
from core.data.cleaning import research_bar_cleanup

def test_flat_zero_volume_non_session_is_quarantined_with_evidence():
    dates=pd.DatetimeIndex(['2024-01-01','2024-01-02','2024-01-03'])
    bars=pd.DataFrame({'Open':[100.,100.,101.],'High':[101.,100.,102.],'Low':[99.,100.,100.],
        'Close':[100.,100.,101.],'Volume':[100.,0.,200.]},index=dates)
    cleaned,report=research_bar_cleanup(bars,dates[[0,2]])
    assert cleaned.index.tolist()==dates[[0,2]].tolist()
    assert report['quarantined_non_session_dates']==['2024-01-02']
    assert report['point_in_time_verified'] is False

def test_zero_volume_on_expected_session_stays_unresolved():
    dates=pd.DatetimeIndex(['2024-01-01','2024-01-02'])
    bars=pd.DataFrame({'Open':[100.,100.],'High':[101.,100.],'Low':[99.,100.],
        'Close':[100.,100.],'Volume':[100.,0.]},index=dates)
    cleaned,report=research_bar_cleanup(bars,dates)
    assert len(cleaned)==2
    assert report['unresolved_session_dates']==['2024-01-02']
