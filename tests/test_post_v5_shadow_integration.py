from datetime import datetime,timezone
import numpy as np
import pandas as pd
from core.data.contracts import DataSnapshot,SourceMetadata
from core.forecasting.ledger import PredictionLedger
from research.post_v5.archive import RunArchive

def test_complete_captured_inputs_generate_real_frozen_shadow_ledger(tmp_path,monkeypatch):
    from research.post_v5.shadow import forecast_from_run,STUDY
    import research.post_v5.shadow as shadow
    import research.post_v5.archive as archive_module
    class StudyClock:
        @classmethod
        def now(cls,tz=None):return pd.Timestamp('2026-10-08T20:10:00Z').to_pydatetime()
    monkeypatch.setattr(shadow,'datetime',StudyClock)
    monkeypatch.setattr(archive_module,'datetime',StudyClock)
    from core.forecasting.service import active_research,load_research_model
    dates=pd.bdate_range('2024-01-01','2026-10-08',tz='Asia/Kolkata')
    close=100*np.exp(np.arange(len(dates))*.0001+np.sin(np.arange(len(dates))*.12)*.02)
    frame=pd.DataFrame({'Open':close,'High':close*1.01,'Low':close*.99,'Close':close,'Adj Close':close,
        'Volume':np.full(len(dates),100000.),'Dividends':0.,'Stock Splits':0.},index=dates.tz_convert('UTC'))
    fetched='2026-10-08T20:00:00+00:00'
    def snapshot(symbol,family):
        return DataSnapshot(frame,SourceMetadata(provider_id='synthetic-test-fixture',provider_version='1',symbol=symbol,
            family=family,exchange_timezone='Asia/Kolkata',interval='1d',adjustment_policy='synthetic fixture',
            source_timestamp=frame.index[-1].isoformat(),available_at=fetched,fetched_at=fetched,vintage_at=fetched,
            availability_verified=False,revision_history_verified=False))
    sources={'TEST.NS':snapshot('TEST.NS','india_equity'),'PEER.NS':snapshot('PEER.NS','india_equity')}
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    settings=json.loads((root/'config/context_sources.json').read_text())
    sources.update({key:snapshot(symbol,'research_reference') for key,symbol in settings['sources'].items()})
    archive=RunArchive(tmp_path);run=archive.save_run(sources,{},fetched,STUDY)
    pointer=active_research();bundles={h:load_research_model('india_equity',h,pointer) for h in [1,5,10,20]}
    result=forecast_from_run(run,'TEST.NS',directory=tmp_path,bundles=bundles)
    records=PredictionLedger(tmp_path/'shadow_predictions.sqlite').rows()
    assert len(records)==4 and len(result['forecasts'])==4
    assert all(r['abstain'] and r['confidence']=='LOW' for r in records)
    assert all(r['input_archive_observed_at_prediction'] and not r['point_in_time_verified'] for r in records)
    assert all(r['feature_values'] and r['input_run_id']==run for r in records)
    assert {r['horizon'] for r in records}=={1,5,10,20}


def test_stale_ticker_tail_is_rejected_against_later_captured_peer_session(tmp_path,monkeypatch):
    import pytest
    from research.post_v5.shadow import forecast_from_run,STUDY
    import research.post_v5.shadow as shadow
    import research.post_v5.archive as archive_module
    class Clock:
        @classmethod
        def now(cls,tz=None):return pd.Timestamp('2026-10-08T20:10:00Z').to_pydatetime()
    monkeypatch.setattr(shadow,'datetime',Clock);monkeypatch.setattr(archive_module,'datetime',Clock)
    def make(symbol,end):
        dates=pd.bdate_range('2024-01-01',end,tz='Asia/Kolkata').tz_convert('UTC')
        frame=pd.DataFrame({'Open':100.,'High':101.,'Low':99.,'Close':100.,'Adj Close':100.,'Volume':100000.,'Dividends':0.,'Stock Splits':0.},index=dates)
        fetched='2026-10-08T20:00:00Z'
        return DataSnapshot(frame,SourceMetadata(provider_id='synthetic-test-fixture',provider_version='1',symbol=symbol,family='india_equity',exchange_timezone='Asia/Kolkata',interval='1d',adjustment_policy='fixture',source_timestamp=dates[-1].isoformat(),available_at=fetched,fetched_at=fetched,vintage_at=fetched,availability_verified=False,revision_history_verified=False))
    archive=RunArchive(tmp_path)
    run=archive.save_run({'TEST.NS':make('TEST.NS','2026-10-07'),'PEER.NS':make('PEER.NS','2026-10-08')},{},'2026-10-08T20:00:00Z',STUDY)
    with pytest.raises(ValueError,match='price data'):
        forecast_from_run(run,'TEST.NS',directory=tmp_path)
    assert not (tmp_path/'shadow_predictions.sqlite').exists()


@__import__('pytest').mark.parametrize('last_source_session',['2026-10-08','2026-10-09'])
def test_preclose_capture_cannot_issue_expired_one_session_forecast_after_close(tmp_path,monkeypatch,last_source_session):
    import pytest
    import research.post_v5.shadow as shadow
    import research.post_v5.archive as archive_module
    from research.post_v5.shadow import forecast_from_run,STUDY
    class Clock:
        @classmethod
        def now(cls,tz=None):return pd.Timestamp('2026-10-09T11:00:00Z').to_pydatetime()
    monkeypatch.setattr(shadow,'datetime',Clock);monkeypatch.setattr(archive_module,'datetime',Clock)
    def make(symbol):
        dates=pd.bdate_range('2024-01-01',last_source_session,tz='Asia/Kolkata').tz_convert('UTC')
        frame=pd.DataFrame({'Open':100.,'High':101.,'Low':99.,'Close':100.,'Adj Close':100.,'Volume':100000.,'Dividends':0.,'Stock Splits':0.},index=dates)
        fetched='2026-10-09T09:00:00Z'
        return DataSnapshot(frame,SourceMetadata(provider_id='synthetic-test-fixture',provider_version='1',symbol=symbol,family='india_equity',exchange_timezone='Asia/Kolkata',interval='1d',adjustment_policy='fixture',source_timestamp=dates[-1].isoformat(),available_at=fetched,fetched_at=fetched,vintage_at=fetched,availability_verified=False,revision_history_verified=False))
    archive=RunArchive(tmp_path);run=archive.save_run({'TEST.NS':make('TEST.NS'),'PEER.NS':make('PEER.NS')},{},'2026-10-09T09:01:00Z',STUDY)
    with pytest.raises(ValueError,match='session already closed'):
        forecast_from_run(run,'TEST.NS',directory=tmp_path)
    assert not (tmp_path/'shadow_predictions.sqlite').exists()
