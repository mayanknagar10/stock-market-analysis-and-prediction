import numpy as np
import pandas as pd
import pytest
from core.forecasting.heads import DirectRegressor, price_from_log_return

def training():
    rng=np.random.default_rng(13)
    times=pd.date_range('2020-01-01',periods=160,tz='UTC')
    X=pd.DataFrame({'momentum':rng.normal(size=160),'volatility':rng.uniform(.01,.04,160)},index=times)
    y=.01*X.momentum+.002
    return X,y,times

def test_direct_regression_fits_both_tree_models_and_rejects_schema_drift():
    X,y,times=training()
    model=DirectRegressor('india_equity',5,n_estimators=12).fit(X,y,times,times+pd.Timedelta('5D'),
                                                           pd.Timestamp('2021-01-01',tz='UTC'))
    result=model.predict(X.iloc[[-1]])
    assert set(result['components'])=={'xgb','lgb'}
    assert np.isfinite(result['central_log_return'][0])
    with pytest.raises(ValueError,match='schema'): model.predict(X.iloc[[-1]][['volatility','momentum']])

def test_horizon_model_does_not_multiply_one_day_inference():
    X,y,times=training()
    one=DirectRegressor('india_equity',1,n_estimators=12).fit(X,y,times,times+pd.Timedelta('1D'),pd.Timestamp('2021-01-01',tz='UTC'))
    five=DirectRegressor('india_equity',5,n_estimators=12).fit(X,-y,times,times+pd.Timedelta('5D'),pd.Timestamp('2021-01-01',tz='UTC'))
    assert not np.isclose(five.predict(X.iloc[[-1]])['central_log_return'][0],
                          5*one.predict(X.iloc[[-1]])['central_log_return'][0])

def test_price_conversion_validates_price_and_preserves_return_units():
    assert price_from_log_return(100.,np.log(1.05))==pytest.approx(105.)
    with pytest.raises(ValueError): price_from_log_return(-1.,.02)
    with pytest.raises(ValueError): price_from_log_return(100.,float('inf'))

def test_unsupported_families_and_horizons_fail():
    with pytest.raises(ValueError): DirectRegressor('crypto',5)
    with pytest.raises(ValueError): DirectRegressor('india_equity',60)
