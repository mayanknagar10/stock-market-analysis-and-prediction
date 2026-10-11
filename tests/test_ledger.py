import json
import sqlite3
import pandas as pd
import pytest
from core.forecasting.ledger import PredictionLedger,resolve_outcome

def prediction():
    return {'prediction_id':'id1','ticker':'TEST.NS','market':'india_equity','forecast_as_of':'2024-01-01T10:20Z',
        'generated_at':'2024-01-01T10:21Z','information_cutoff':'2024-01-01T10:20Z','timezone':'UTC',
        'horizon':5,'current_price':100.,'central_log_return':.05,'central_price':105.1271096,
        'quantiles':[-.05,-.02,.05,.1,.15],'p_positive':.6,'confidence':'LOW','abstain':True,
        'model_agreement':1.,'regime':'sideways_low_vol','data_quality':{'eligible':True},'ood_score':0.,
        'model_version':'research-test','feature_version':'v1','training_cutoff':'2023-12-01T00:00Z',
        'provider_versions':{'yahoo':'test'},'snapshot_ids':['snapshot'],'feature_schema_hash':'hash',
        'code_version':'code','scope':'research_only','point_in_time_verified':False}

def test_prediction_and_outcome_are_append_only(tmp_path):
    ledger=PredictionLedger(tmp_path/'ledger.sqlite')
    ledger.append(prediction())
    with pytest.raises(ValueError): ledger.append(prediction())
    outcome={'status':'matured','actual_return':.03,'actual_log_return':float(__import__('numpy').log(1.03)), 'actual_future_price':103.,'interval_80_hit':True,'interval_50_hit':True,'direction_correct':True,'prediction_error_log':float(__import__('numpy').log(1.03)-.05),'maturity_timestamp':'2024-01-08T10:20Z','resolved_at':'2024-01-09T00:00Z','outcome_snapshot_id':'outcome1'}
    ledger.attach_outcome('id1',outcome)
    with pytest.raises(ValueError): ledger.attach_outcome('id1',outcome)
    rows=ledger.rows()
    assert rows[0]['prediction_id']=='id1'
    assert rows[0]['outcome']['actual_return']==.03
    with sqlite3.connect(tmp_path/'ledger.sqlite') as connection:
        with pytest.raises(sqlite3.IntegrityError): connection.execute('DELETE FROM forecasts')

def test_future_outcome_and_too_few_sessions_do_not_mature():
    dates=pd.bdate_range('2024-01-01',periods=6,tz='UTC')
    bars=pd.DataFrame({'Close':[100.,101.,102.,103.,104.,105.],'Adj Close':[100.,101.,102.,103.,104.,105.]},index=dates)
    assert resolve_outcome(prediction(),bars.iloc[:5],as_of='2024-01-20T00:00Z',expected_sessions=dates) is None
    assert resolve_outcome(prediction(),bars,as_of='2024-01-03T00:00Z',expected_sessions=dates) is None
    result=resolve_outcome(prediction(),bars,as_of='2024-01-20T00:00Z',expected_sessions=dates)
    assert result['actual_return']==pytest.approx(.05)
    assert result['actual_future_price']==105.

def test_incomplete_provenance_is_rejected(tmp_path):
    record=prediction()
    del record['snapshot_ids']
    with pytest.raises(ValueError): PredictionLedger(tmp_path/'ledger.sqlite').append(record)


def test_ledger_rejects_inconsistent_price_and_incomplete_outcomes(tmp_path):
    ledger=PredictionLedger(tmp_path/'ledger.sqlite')
    bad=prediction(); bad['central_price']=999.
    with pytest.raises(ValueError,match='price/return'): ledger.append(bad)
    ledger.append(prediction())
    with pytest.raises(ValueError,match='Incomplete'): ledger.attach_outcome('id1',{'status':'matured'})

def test_missing_session_does_not_shift_maturity():
    dates=pd.DatetimeIndex(['2024-01-01T00:00Z','2024-01-02T00:00Z','2024-01-03T00:00Z'])
    bars=pd.DataFrame({'Close':[100.,120.],'Adj Close':[100.,120.]},index=dates[[0,2]])
    record=prediction(); record['horizon']=1
    assert resolve_outcome(record,bars,'2024-01-10T00:00Z',expected_sessions=dates) is None

def test_preopen_forecast_uses_known_previous_close_as_origin():
    dates=pd.DatetimeIndex(['2023-12-29T00:00Z','2024-01-01T00:00Z','2024-01-02T00:00Z'])
    bars=pd.DataFrame({'Close':[100.,200.,220.],'Adj Close':[100.,200.,220.]},index=dates)
    record=prediction(); record.update(horizon=1,forecast_as_of='2024-01-01T03:30Z',information_cutoff='2024-01-01T03:30Z')
    result=resolve_outcome(record,bars,'2024-01-10T00:00Z',expected_sessions=dates)
    assert result['actual_return']==pytest.approx(1.)


def test_batch_prediction_admission_is_atomic(tmp_path):
    ledger=PredictionLedger(tmp_path/'ledger.sqlite')
    first=prediction()
    second=prediction(); second['prediction_id']='id2'; second['central_price']=999.
    with pytest.raises(ValueError): ledger.append_many([first,second])
    assert ledger.rows()==[]
    second['central_price']=first['central_price']
    assert ledger.append_many([first,second])==['id1','id2']
    with pytest.raises(ValueError): ledger.append_many([prediction()])
    assert len(ledger.rows())==2
