import json
import pytest
from test_post_v5_archive import source

def test_prediction_preserves_feature_vector_and_actual_input_lineage(tmp_path):
    from research.post_v5.shadow import record_prediction
    snapshot=source();features={'ret_1d':.01}
    result={'central_log_return':.02,'central_price':102.02013400267558,'quantiles':[-.1,-.05,.02,.05,.1],
        'p_positive':.52,'p_positive_calibrated':None,'trust':{'confidence':'LOW','abstain':True,'agreement':.8,'reasons':['RESEARCH_ONLY']},'ood_score':.8}
    metadata={'model_version':'research-20261008T190859','training_cutoff':'2025-09-01T00:00:00Z','feature_version':'fixture','feature_schema_hash':'schema','code_version':'code'}
    record=record_prediction(snapshot,result,metadata,features,5,'2026-10-08T10:01:00Z','2026-10-02','input-run',{'ABC.NS':'stock','market_india':'market'},'unknown',{'fixture':'1'},{'eligible':True})
    features['ret_1d']=9
    assert record['feature_values']=={'ret_1d':.01}
    assert record['input_run_id']=='input-run'
    assert record['scope']=='research_only' and record['point_in_time_verified'] is False
    assert record['input_archive_observed_at_prediction'] is True
    assert record['abstain'] and record['confidence']=='LOW'
    assert record['source_retrieved_at']=='2026-10-08T10:00:00+00:00'

def test_shadow_cannot_persist_manufactured_confidence():
    from research.post_v5.shadow import record_prediction
    snapshot=source()
    with pytest.raises(ValueError,match='abstain'):
        record_prediction(snapshot,{'trust':{'confidence':'HIGH','abstain':False}}, {},{},5,'2026-10-08T10:01:00Z','2026-10-02','run',{},'unknown',{}, {})


def test_shadow_record_roundtrips_real_immutable_ledger(tmp_path):
    from research.post_v5.shadow import record_prediction
    from core.forecasting.ledger import PredictionLedger
    result={'central_log_return':.02,'central_price':102.02013400267558,'quantiles':[-.1,-.05,.02,.05,.1],
        'p_positive':.52,'p_positive_calibrated':None,'trust':{'confidence':'LOW','abstain':True,'agreement':.8,'reasons':['RESEARCH_ONLY']},'ood_score':.8}
    metadata={'model_version':'research-20261008T190859','training_cutoff':'2025-09-01T00:00:00Z','feature_version':'fixture','feature_schema_hash':'schema','code_version':'code'}
    record=record_prediction(source(),result,metadata,{'ret_1d':.01},5,'2026-10-08T10:01:00Z','2026-10-02','run',{'ABC.NS':'stock'},'unknown',{'fixture':'1'},{'eligible':True})
    ledger=PredictionLedger(tmp_path/'shadow.sqlite');ledger.append(record)
    assert ledger.rows()[0]['feature_values']=={'ret_1d':.01}
    with pytest.raises(ValueError):ledger.append(record)
    assert len(ledger.rows())==1
