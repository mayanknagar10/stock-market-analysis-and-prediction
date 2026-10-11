"""Direct cumulative log-return labels over explicitly supplied exchange sessions."""
import numpy as np
import pandas as pd
from core.validation.alignment import aware_index

SUPPORTED_HORIZONS = (1,5,10,20)

def direct_return_targets(adjusted_close,expected_sessions,horizons=SUPPORTED_HORIZONS):
    """Only adjusted closes may be supplied; raw closes are for display.

    Missing candles are errors, never squeezed into a shorter horizon. Trailing
    unmatured labels remain NaN/NaT and must not enter fitting/evaluation.
    """
    if not isinstance(adjusted_close,pd.Series): raise ValueError('Prices must be a Series')
    dates = aware_index(adjusted_close.index,'price timestamps')
    expected = aware_index(expected_sessions,'expected sessions')
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('Price sessions must be unique and ordered')
    if not expected.is_unique or not expected.is_monotonic_increasing or not dates.equals(expected):
        raise ValueError('Missing, extra or misaligned exchange sessions')
    values = adjusted_close.to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values<=0).any():
        raise ValueError('Adjusted prices must be positive and finite')
    if not horizons or any(type(h) is not int or h not in SUPPORTED_HORIZONS for h in horizons):
        raise ValueError('Unsupported target horizon')
    close = pd.Series(values,index=dates)
    endpoints = pd.Series(dates,index=dates)
    result = pd.DataFrame(index=dates)
    for horizon in horizons:
        result['target_'+str(horizon)+'d'] = np.log(close.shift(-horizon)/close)
        result['label_end_'+str(horizon)+'d'] = endpoints.shift(-horizon)
    return result
