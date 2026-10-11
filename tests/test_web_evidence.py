import copy
import json
import numpy as np
import pandas as pd
import pytest
from application.services.prospective import ServiceError

def test_historical_evidence_reads_preserved_failed_gate_separately():
    from application.services.historical_evidence import HistoricalEvidence
    result=HistoricalEvidence().read('final')
    assert result.final_gate=='FAILED' and result.evidence_stream=='RETROSPECTIVE_RECONSTRUCTED'
    assert result.metrics and result.production_promotion=='BLOCKED'
    assert 'parameters' not in result.model_dump_json() and 'C:' not in result.model_dump_json()
    development=HistoricalEvidence().read('development')
    assert development.models and development.point_in_time_verified is False

def test_cross_ticker_history_keeps_rows_and_hides_sealed_outcomes(tmp_path):
    from test_automation_api import setup
    from application.services.archived_forecast import ArchivedForecasts
    _,ticker,prospective=setup(tmp_path)
    rows=prospective.rows()
    rows[0]['evaluation_sealed']=True;rows[0]['outcome']={'actual_return':999}
    prospective.rows=lambda group='canonical':copy.deepcopy(rows)
    service=ArchivedForecasts(prospective)
    result=service.history(limit=2)
    assert result.total==4 and len(result.items)==2
    assert all(row.outcome is None for row in result.items if row.evaluation_sealed)
    assert 'feature_values' not in result.model_dump_json()

def test_scenario_clones_all_inputs_and_public_allowlist():
    from application.services.archived_forecast import ArchivedForecasts
    from application.web_schemas import ScenarioRequest
    from test_automation_outcomes import snapshot
    source=snapshot('^NSEI',['2026-10-07','2026-10-08'],[100.,101.],'2026-10-09T03:45Z',reference=True)
    record={'prediction_id':'fixture','ticker':'TCS.NS','market':'india_equity','information_cutoff':'2026-10-09T04:00Z','current_price':100,'horizon':1,'feature_values':{'a':1}}
    inputs={'record':record,'bundle':{'metadata':{'model_version':'research-20261008T190859'},'values':[1]},'stock':source.frame,'references':{'market_india':source},'settings':{}}
    seen=[]
    class Engine:
        def __init__(self,bundle,stock,ticker,family,references,settings,current_price,as_of):
            self.bundle=bundle;self.stock=stock;self.references=references
        def run(self,shocks):
            seen.append(float(self.stock.Close.iloc[-1]));self.stock.iloc[-1,0]=999;self.bundle['values'].append(9)
            return {'central_return':.03,'central_price':103,'quantiles':[0,.01,.02,.03,.04],'changed_features':['a'],'feature_values':{'private':2}}
    svc=ArchivedForecasts(input_loader=lambda prediction_id,group:inputs,engine_factory=Engine)
    request=ScenarioRequest(shocks={'stock':.1})
    one=svc.scenario('fixture',request);two=svc.scenario('fixture',request)
    assert one==two and seen==[101,101] and inputs['bundle']['values']==[1]
    assert one.abstain and not one.recommendation_allowed and 'feature_values' not in one.model_dump_json()

def test_archive_model_identity_and_source_cutoff_fail_closed():
    from application.services.archived_forecast import ArchivedForecasts
    from application.web_schemas import ScenarioRequest
    from test_automation_outcomes import snapshot
    source=snapshot('^NSEI',['2026-10-09'],[100.],'2026-10-10T12:00Z',reference=True)
    inputs={'record':{'ticker':'TCS.NS','market':'india_equity','information_cutoff':'2026-10-09T01:00Z','current_price':100},'bundle':{},'stock':source.frame,'references':{'market_india':source},'settings':{}}
    svc=ArchivedForecasts(input_loader=lambda *args:inputs)
    with pytest.raises(ServiceError):svc.scenario('fixture',ScenarioRequest(shocks={'stock':.1}))

def test_analogues_exclude_labels_at_or_after_saved_origin():
    from application.services.archived_forecast import ArchivedForecasts
    class Regression:
        feature_names=['a']
        def scaled(self,row):return np.array([[0.]])
    bundle={'regression':Regression(),'analogues':{'label_ends':np.array(['2026-10-08T12:00Z','2026-10-09T04:00Z','2026-10-10T12:00Z']), 'features':np.array([[1.],[0.],[0.]]),'returns':np.array([.01,.9,.9]),'timestamps':np.array(['2026-10-01','2026-10-02','2026-10-03']), 'tickers':np.array(['TCS.NS']*3)}}
    inputs={'record':{'information_cutoff':'2026-10-09T04:00Z','feature_values':{'a':0}},'bundle':bundle,'stock':pd.DataFrame(), 'references':{}, 'settings':{}}
    service=ArchivedForecasts(input_loader=lambda *args:inputs)
    result=service.analogues('fixture')
    assert result.n==1 and result.median_return==pytest.approx(np.expm1(.01))

def test_original_v5_ledger_history_read_only_and_separate():
    import hashlib
    from application.config import ROOT
    from application.services.archived_forecast import ArchivedForecasts
    path=ROOT/'data/v5/predictions.sqlite';before=hashlib.sha256(path.read_bytes()).hexdigest()
    result=ArchivedForecasts().history(group='v5')
    assert result.total==4 and all(r.origin_group=='v5' for r in result.items)
    assert before==hashlib.sha256(path.read_bytes()).hexdigest()

def test_development_preserves_raw_and_calibrated_rejection_and_final_lock():
    from application.services.historical_evidence import HistoricalEvidence
    result=HistoricalEvidence().read('development')
    assert 'raw_probability_metrics' in result.models[0]['metadata']
    assert 'calibration_accepted' in result.models[0]['metadata']
    final=HistoricalEvidence().read('final')
    assert final.protocol_lock and final.final_test_consumed

def test_archived_detail_public_fields_and_research_only_gets(tmp_path,monkeypatch):
    from application.services.archived_forecast import ArchivedForecasts
    from test_automation_api import setup
    _,_,prospective=setup(tmp_path)
    row=prospective.rows()[0]
    row['drivers']={'features':['ret_1d'],'values':[.2],'base_value':.01,'label':'Association','password':'secret'}
    row['relative_forecasts']={'alpha_market':{'central_excess_log_return':.01,'raw_p_outperform':.6,'calibrated_p_outperform':None,'validation_metrics':{'n':10},'scope':'research_only','private_path':'C:/private'}}
    prospective.rows=lambda group='canonical':[row]
    detail=ArchivedForecasts(prospective).detail(row['prediction_id'])
    payload=detail.model_dump_json()
    assert 'private_path' not in payload and 'password' not in payload
    assert detail.drivers.values==[.2]
    from core.forecasting import service as forecasting
    monkeypatch.setattr(forecasting,'load_research_model',lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('No model loading')))
    from application.services.historical_evidence import HistoricalEvidence
    assert HistoricalEvidence().read('final').final_gate=='FAILED'

def test_legacy_pagination_preserves_388_and_saved_low_abstention():
    from application.services.archived_forecast import ArchivedForecasts
    svc=ArchivedForecasts();items=[]
    for offset in range(0,388,100):items.extend(svc.history('legacy',offset=offset,limit=100).items)
    assert len(items)==388 and len({r.prediction_id for r in items})==388
    assert all(r.abstain and r.confidence=='LOW' and not r.recommendation_allowed for r in items)

@pytest.mark.parametrize('value',['/home/user/private/file.json','./data/private.json','../secret/models.bin','data/private.json','Reference /tmp/internal/data.sqlite','C:/private/key','C:\\private\\key'])
def test_historical_metrics_redact_paths_on_all_platforms(value):
    from application.services.historical_evidence import public_metrics
    assert public_metrics({'source':value})=={'source':None}

def test_historical_metrics_do_not_leak_path_dictionary_keys():
    from application.services.historical_evidence import public_metrics
    assert public_metrics({'/tmp/private/report':5,'metrics':{'mae':.1}})=={'metrics':{'mae':.1}}

def test_metric_redaction_preserves_labels_dates_and_ordinary_prose():
    from application.services.historical_evidence import public_metrics
    value={'label':'MAE/RMSE comparison','month':'2026-10','date':'2026-10-09T01:00Z','note':'Historical neighbours are not independent.','metrics':[{'mae':.2,'path':'../private/model.bin'}]}
    expected=dict(value,metrics=[{'mae':.2,'path':None}])
    assert public_metrics(value)==expected

def test_quarantine_and_limitations_apply_portable_path_redaction(tmp_path):
    from application.services.historical_evidence import HistoricalEvidence
    path=tmp_path/'models/v5/QUARANTINE.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'versions':[{'model_version':'test','reason':'Stored at /home/operator/private/model.bin','state':'QUARANTINED','scope':'research_only'}],'limitations':['Private data/research/hidden.json']}),encoding='utf-8')
    result=HistoricalEvidence(tmp_path).read('quarantine')
    assert '/home' not in result.model_dump_json() and 'data/research/' not in result.model_dump_json()
