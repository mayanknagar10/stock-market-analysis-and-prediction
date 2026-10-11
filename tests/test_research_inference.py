import numpy as np
import pandas as pd
import pytest
from core.forecasting.inference import infer_bundle,model_contributions,historical_analogues
from core.forecasting.heads import DirectRegressor
from core.forecasting.distribution import DistributionHeads
from core.forecasting.trust import OODDetector

def bundle():
    rng=np.random.default_rng(20)
    times=pd.date_range('2020-01-01',periods=100,tz='UTC')
    X=pd.DataFrame(rng.normal(size=(100,2)),columns=['a','b'],index=times)
    y=.02*X.a+rng.normal(0,.01,100)
    reg=DirectRegressor('india_equity',5,n_estimators=10).fit(X,y,times,times+pd.Timedelta('5D'),pd.Timestamp('2021-01-01',tz='UTC'))
    dist=DistributionHeads(n_estimators=10).fit(reg.scaled(X),y)
    return {'regression':reg,'distribution':dist,'calibrator':None,'ood':OODDetector().fit(reg.scaled(X)),
        'conformal_margin':0.,'policy':{},'analogues':None,'metadata':{'model_version':'test','training_cutoff':'2020-12-01T00:00Z',
        'validation_cutoff':'2020-12-02T00:00Z','point_in_time_verified':False,'final_test_passed':False,
        'calibration_accepted':False,'validation_n':100,'feature_groups':{'stock':['a','b']}}},X

def test_inference_stays_research_and_contributions_match_return():
    data,X=bundle()
    result=infer_bundle(data,X.iloc[[-1]],100.,'2021-01-01T00:00Z',data_eligible=True)
    assert result['trust']['abstain']
    assert result['scope']=='research_only'
    assert result['p_positive_calibrated'] is None
    contributions=model_contributions(data,X.iloc[[-1]])
    assert sum(contributions['values'])+contributions['base_value']==pytest.approx(result['central_log_return'],abs=1e-6)

def test_future_training_or_validation_checkpoint_cannot_backtest_earlier_period():
    data,X=bundle()
    with pytest.raises(ValueError,match='cutoff'):
        infer_bundle(data,X.iloc[[-1]],100.,'2020-06-01T00:00Z',data_eligible=True)

def test_analogue_outcomes_must_have_matured_before_asof():
    data,X=bundle()
    data['analogues']={'features':np.array([[0.,0.],[0.,0.]]),'returns':np.array([.1,.9]),
        'timestamps':np.array(['2020-01-01T00:00Z','2020-02-01T00:00Z']),
        'label_ends':np.array(['2020-01-06T00:00Z','2030-02-06T00:00Z']),'tickers':np.array(['A','B'])}
    result=historical_analogues(data,X.iloc[[-1]],'2021-01-01T00:00Z')
    assert result['n']==1
    assert result['median_return']==pytest.approx(np.expm1(.1))


def test_future_creation_time_and_research_promotion_are_rejected():
    data,X=bundle()
    data['metadata']['created_at']='2030-01-01T00:00Z'
    with pytest.raises(ValueError,match='cutoff'): infer_bundle(data,X.iloc[[-1]],100.,'2021-01-01T00:00Z')
    data['metadata']['created_at']='2020-12-03T00:00Z'
    data['metadata']['point_in_time_verified']=True
    data['metadata']['final_test_passed']=True
    with pytest.raises(ValueError,match='research-only'):
        infer_bundle(data,X.iloc[[-1]],100.,'2021-01-01T00:00Z',research_only=False)


def test_analogue_arithmetic_excess_return_uses_both_stock_and_market():
    data,X=bundle()
    data['analogues']={'features':np.array([[0.,0.]]),'returns':np.array([np.log(1.1)]),
        'excess_market':np.array([np.log(1.1)-np.log(1.05)]),
        'timestamps':np.array(['2020-01-01T00:00Z']),'label_ends':np.array(['2020-01-06T00:00Z']),
        'tickers':np.array(['A'])}
    assert historical_analogues(data,X.iloc[[-1]],'2021-01-01T00:00Z')['median_excess_return']==pytest.approx(.05)
