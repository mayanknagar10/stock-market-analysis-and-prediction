import numpy as np
import pandas as pd
import pytest

def test_benchmark_endpoints_use_exact_sessions_never_shift_missing_bar():
    from research.post_v5.outcomes import endpoint_return
    values=pd.Series([100.,110.],index=pd.to_datetime(['2026-10-01','2026-10-05']))
    assert endpoint_return(values,'2026-10-01','2026-10-05')==pytest.approx(np.log(1.1))
    assert endpoint_return(values,'2026-10-01','2026-10-02') is None

def test_fixed_sector_proxy_cannot_silently_drop_missing_members():
    from research.post_v5.outcomes import peer_sector_return
    values=pd.Series([100.,110.],index=pd.to_datetime(['2026-10-01','2026-10-05']))
    members=[{'ticker':t,'industry':'IT'} for t in ['A.NS','B.NS','C.NS']]
    assert peer_sector_return('A.NS','IT',members,{'B.NS':values},'2026-10-01','2026-10-05')['log_return'] is None
    result=peer_sector_return('A.NS','IT',members,{'B.NS':values,'C.NS':values},'2026-10-01','2026-10-05')
    assert result['log_return']==pytest.approx(np.log(1.1))
    assert result['n_fixed_peers']==2


def endpoint_snapshot(fetched='2026-10-08T10:10:00Z',family='research_reference',volume=100.):
    from core.data.contracts import DataSnapshot,SourceMetadata
    dates=pd.date_range('2026-10-06',periods=3,tz='Asia/Kolkata').tz_convert('UTC')
    frame=pd.DataFrame({'Open':100.,'High':101.,'Low':99.,'Close':100.,'Adj Close':100.,'Volume':volume,'Dividends':0.,'Stock Splits':0.},index=dates)
    meta=SourceMetadata(provider_id='fixture',provider_version='1',symbol='ABC.NS' if family=='india_equity' else '^NSEI',family=family,
        exchange_timezone='Asia/Kolkata',interval='1d',adjustment_policy='fixture',source_timestamp=dates[-1].isoformat(),
        available_at=fetched,fetched_at=fetched,vintage_at=fetched,availability_verified=False,revision_history_verified=False)
    return DataSnapshot(frame,meta)

def test_outcome_reference_maturity_bar_cannot_complete_after_original_retrieval():
    from research.post_v5.outcomes import prices_for
    snapshot=endpoint_snapshot()
    values=prices_for(snapshot,asof='2026-10-08T11:00:00Z')
    assert endpoint_return_for_test(values,'2026-10-06','2026-10-08') is None
    assert pd.Timestamp('2026-10-07') in values.index

def endpoint_return_for_test(values,origin,end):
    from research.post_v5.outcomes import endpoint_return
    return endpoint_return(values,origin,end)

def test_suspended_equity_endpoint_is_unusable_but_index_zero_volume_is_allowed():
    from research.post_v5.outcomes import prices_for
    stock=endpoint_snapshot(fetched='2026-10-08T11:00:00Z',family='india_equity',volume=0.)
    reference=endpoint_snapshot(fetched='2026-10-08T11:00:00Z',volume=0.)
    assert endpoint_return_for_test(prices_for(stock),'2026-10-06','2026-10-08') is None
    assert endpoint_return_for_test(prices_for(reference),'2026-10-06','2026-10-08')==pytest.approx(0.)

def test_outcome_revision_diagnostics_compare_original_known_prices_and_actions():
    from core.data.contracts import DataSnapshot
    from research.post_v5.outcomes import revision_diagnostics
    original=endpoint_snapshot(fetched='2026-10-08T11:00:00Z',family='india_equity')
    frame=original.frame;frame.loc[frame.index[-1],'Adj Close']=99.
    changed=DataSnapshot(frame,original.metadata)
    result=revision_diagnostics(original,changed,'2026-10-08')
    assert result['fields_changed']==['Adj Close']
    assert result['origin_values']['Adj Close']=={'at_prediction':100.,'at_outcome':99.}


@pytest.mark.parametrize('defect',['missing_adjusted_price','suspended_volume','reference_captured_before_close'])
def test_resolver_keeps_bad_or_premature_endpoints_pending(tmp_path,defect):
    from dataclasses import replace
    from core.data.contracts import DataSnapshot
    from core.forecasting.ledger import PredictionLedger
    from research.post_v5.archive import RunArchive
    from research.post_v5.shadow import record_prediction,STUDY
    from research.post_v5.outcomes import resolve_run
    stock=endpoint_snapshot(fetched='2026-10-08T11:00:00Z',family='india_equity')
    original_frame=stock.frame.iloc[:2]
    meta=replace(stock.metadata,source_timestamp=original_frame.index[-1].isoformat(),fetched_at='2026-10-07T11:00:00Z',available_at='2026-10-07T11:00:00Z',vintage_at='2026-10-07T11:00:00Z')
    original=DataSnapshot(original_frame,meta)
    archive=RunArchive(tmp_path);first=archive.save_run({'ABC.NS':original},{},'2026-10-07T11:01:00Z',STUDY)
    result={'central_log_return':.02,'central_price':102.02013400267558,'quantiles':[-.1,-.05,.02,.05,.1],'p_positive':.52,'p_positive_calibrated':None,'trust':{'confidence':'LOW','abstain':True,'agreement':.8,'reasons':['RESEARCH_ONLY']},'ood_score':.8}
    model={'model_version':'research-20261008T190859','training_cutoff':'2025-09-01T00:00:00Z','feature_version':'fixture','feature_schema_hash':'schema','code_version':'code'}
    record=record_prediction(original,result,model,{'ret_1d':.01},1,'2026-10-07T11:05:00Z','2026-10-07',first,archive.load_run(first)['snapshots'],'unknown',{'fixture':'1'},{'eligible':True})
    record['generated_at']='2026-10-07T11:05:00Z';record['benchmark_source_key']='market_india'
    record['sector_source_key']=None;record['frozen_universe_member']=False
    ledger=PredictionLedger(tmp_path/'shadow_predictions.sqlite');ledger.append(record)
    changed=stock.frame
    if defect=='missing_adjusted_price':changed.iloc[-1,changed.columns.get_loc('Adj Close')]=np.nan
    if defect=='suspended_volume':changed.iloc[-1,changed.columns.get_loc('Volume')]=0.
    stock=DataSnapshot(changed,stock.metadata)
    peer=DataSnapshot(endpoint_snapshot(fetched='2026-10-08T11:00:00Z',family='india_equity').frame,replace(stock.metadata,symbol='PEER.NS'))
    market=endpoint_snapshot(fetched='2026-10-08T10:10:00Z' if defect=='reference_captured_before_close' else '2026-10-08T11:00:00Z')
    run=archive.save_run({'ABC.NS':stock,'PEER.NS':peer,'market_india':market},{},'2026-10-08T11:01:00Z',STUDY)
    summary=resolve_run(run,directory=tmp_path)
    assert summary['matured']==0
    assert ledger.rows()[0]['outcome'] is None
    assert ledger.rows()[0]['central_log_return']==.02
