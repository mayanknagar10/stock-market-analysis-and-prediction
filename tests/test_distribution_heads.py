import numpy as np
import pytest
from core.forecasting.distribution import DistributionHeads, ordered_quantiles

def test_quantiles_are_ordered_and_crossing_is_reported():
    fixed,crossed=ordered_quantiles([[.2,-.2,0,.1,-.1]])
    np.testing.assert_allclose(fixed,[[-.2,-.1,0,.1,.2]])
    assert crossed.tolist()==[True]
    with pytest.raises(ValueError): ordered_quantiles([[float('nan')]*5])

def test_both_models_supply_direction_and_conditional_quantiles():
    rng=np.random.default_rng(19)
    X=rng.normal(size=(100,2))
    y=.01*X[:,0]+rng.normal(0,.01,100)
    heads=DistributionHeads(n_estimators=10).fit(X,y)
    result=heads.predict(X[:3])
    assert result['quantiles'].shape==(3,5)
    assert np.all(np.diff(result['quantiles'],axis=1)>=0)
    assert np.all((result['probability']>=0)&(result['probability']<=1))
    assert set(result['probability_components'])=={'xgb','lgb'}
