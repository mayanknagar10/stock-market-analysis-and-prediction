"""Preprocessing fit is explicitly bounded by training label availability."""
import numpy as np
from sklearn.preprocessing import RobustScaler
from core.validation.alignment import aware_index

def fit_training_scaler(values,feature_times,label_ends,fit_before):
    starts = aware_index(feature_times,'feature_times')
    ends = aware_index(label_ends,'label_ends')
    boundary = aware_index([fit_before],'fit_before')[0]
    array = np.asarray(values,dtype=float)
    if array.ndim!=2 or not len(array) or len(array)!=len(starts) or len(starts)!=len(ends):
        raise ValueError('Training values/timestamps have incompatible shapes')
    if not np.isfinite(array).all(): raise ValueError('Training features contain missing/nonfinite values')
    if (ends<=starts).any() or starts.max()>=boundary or ends.max()>=boundary:
        raise ValueError('Training labels reach fit boundary')
    return RobustScaler().fit(values)
