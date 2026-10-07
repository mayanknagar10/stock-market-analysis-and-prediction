import numpy as np
import pandas as pd
import pytest
from core.validation.metrics import forecast_metrics, grouped_metrics

def test_regression_metrics_and_probability_scores():
    result = forecast_metrics([-.1,.1,.2], [0,.2,.1], probability=[.2,.8,.7])
    assert result['mae'] == pytest.approx(.1)
    assert result['rmse'] == pytest.approx(.1)
    assert result['median_absolute_error'] == pytest.approx(.1)
    assert result['bias'] == pytest.approx(1/30)
    assert result['brier'] == pytest.approx(.17/3)
    assert result['directional_accuracy'] == pytest.approx(2/3)
    assert result['precision_up'] == 1
    assert result['recall_down'] == 1
    assert result['calibration'][0]['n'] >= 0

def test_missing_probabilities_stay_unavailable():
    result = forecast_metrics([-.1,.1], [.05,.05])
    assert result['brier'] is None
    assert result['log_loss'] is None
    assert result['calibration'] == []

def test_interval_coverage_and_width_are_measured():
    q = np.array([[-.2,-.15,0,.15,.2]] * 3)
    result = forecast_metrics([-.1,.1,.3], [0,0,0], quantiles=q)
    assert result['coverage_80'] == pytest.approx(2/3)
    assert result['width_80'] == pytest.approx(.4)
    assert result['pinball_q10'] is not None

def test_metrics_reject_nonfinite_or_misaligned_values():
    with pytest.raises(ValueError): forecast_metrics([1,float('nan')], [0,0])
    with pytest.raises(ValueError): forecast_metrics([1,2], [1])
    with pytest.raises(ValueError): forecast_metrics([1,2], [1,2], probability=[.5,1.01])
    with pytest.raises(ValueError): forecast_metrics([1,2], [1,2], quantiles=[[1,0,0,0,0]]*2)

def test_group_metrics_do_not_hide_losing_stocks():
    rows = pd.DataFrame({'actual':[.1,-.1], 'predicted':[.1,.1], 'ticker':['WIN','LOSE']})
    result = grouped_metrics(rows, ['ticker'])
    assert result['ticker']['LOSE']['mae'] == pytest.approx(.2)
    assert result['ticker']['WIN']['mae'] == 0

def test_constant_predictions_have_no_defined_information_coefficient():
    result = forecast_metrics([-.1,.1,.2], [0,0,0])
    assert result['pearson_ic'] is None
    assert result['spearman_ic'] is None
