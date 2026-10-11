import numpy as np
import pandas as pd
import pytest
from core.forecasting.scenarios import ScenarioEngine
from core.forecasting.context import research_feature_frame
from core.forecasting.heads import DirectRegressor
from core.forecasting.distribution import DistributionHeads
from core.forecasting.trust import OODDetector

def test_scenario_rebuilds_real_features_and_does_not_mutate_base(candles):
    ref=candles.copy(); ref.index=ref.index.tz_localize('UTC'); ref['Adj Close']=ref.Close
    refs={'nasdaq':ref}
    features,_=research_feature_frame(candles,'TEST.NS','india_equity',refs,{},research_only=True)
    X=features[['ret_1d','nasdaq_ret_1d']].dropna()
    y=.3*X.nasdaq_ret_1d+.1*X.ret_1d
    reg=DirectRegressor('india_equity',5,n_estimators=10).fit(X,y,X.index,X.index+pd.Timedelta('5D'),pd.Timestamp('2022-01-01',tz='UTC'))
    dist=DistributionHeads(n_estimators=10).fit(reg.scaled(X),y)
    bundle={'regression':reg,'distribution':dist,'calibrator':None,'ood':OODDetector().fit(reg.scaled(X)),
        'conformal_margin':0.,'policy':{},'metadata':{'model_version':'scenario','training_cutoff':'2021-08-20T00:00Z',
        'validation_cutoff':'2021-09-01T00:00Z','point_in_time_verified':False,'final_test_passed':False,'validation_n':100}}
    before=ref.copy()
    engine=ScenarioEngine(bundle,candles,'TEST.NS','india_equity',refs,{},float(candles.Close.iloc[-1]),'2022-01-01T00:00Z')
    result=engine.run({'nasdaq':.1})
    assert result['scope']=='hypothetical_research'
    assert 'nasdaq_ret_1d' in result['changed_features']
    assert np.isfinite(result['central_log_return'])
    pd.testing.assert_frame_equal(ref,before)
    with pytest.raises(ValueError): engine.run({'invented':.1})
    with pytest.raises(ValueError): engine.run({'nasdaq':-1.})
