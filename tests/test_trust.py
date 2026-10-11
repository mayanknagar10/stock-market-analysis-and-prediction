import numpy as np
import pandas as pd
import pytest
from core.forecasting.trust import PlattCalibrator, OODDetector, evidence_policy

def test_calibration_cannot_overlap_base_training():
    t=pd.date_range('2020-01-01',periods=100,tz='UTC')
    p=np.linspace(.1,.9,100)
    labels=(p>.5).astype(int)
    with pytest.raises(ValueError): PlattCalibrator().fit(p,labels,t,base_label_cutoff=t[1])
    calibration=PlattCalibrator().fit(p,labels,t,base_label_cutoff=t[0]-pd.Timedelta('1D'))
    out=calibration.predict(p)
    assert np.all((out>=0)&(out<=1))

def test_ood_extreme_features_force_abstention():
    rng=np.random.default_rng(22)
    detector=OODDetector().fit(rng.normal(size=(200,3)))
    assert detector.score([[100,100,100]])[0]>1
    result=evidence_policy(probability=.9,mean_return=.1,quantiles=[-.01,0,.1,.2,.3],
        component_returns=[.1,.1],data_eligible=True,ood_score=10,validation_support=100,
        calibrated=True,point_in_time_verified=False,final_test_passed=False)
    assert result['abstain']
    assert result['confidence']=='LOW'

def test_research_and_stale_inputs_never_gain_production_confidence():
    result=evidence_policy(probability=.9,mean_return=.1,quantiles=[-.01,0,.1,.2,.3],
        component_returns=[.1,.1],data_eligible=False,ood_score=0,validation_support=100,
        calibrated=True,point_in_time_verified=False,final_test_passed=False)
    assert result['abstain']
    assert result['confidence']=='LOW'
    assert 'DATA_UNHEALTHY' in result['reasons']
    assert 'RESEARCH_ONLY' in result['reasons']
