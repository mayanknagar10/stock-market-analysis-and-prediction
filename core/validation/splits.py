"""Global-calendar expanding evaluation; purge overlapping labels across all tickers."""
import numpy as np
import pandas as pd
from core.validation.alignment import aware_index

def _times(metadata):
    if not {'feature_time','label_end'}.issubset(metadata): raise ValueError('Missing label interval metadata')
    origins = aware_index(metadata.feature_time,'feature_time')
    ends = aware_index(metadata.label_end,'label_end')
    if (ends<=origins).any(): raise ValueError('Label end must follow its feature timestamp')
    return origins,ends

def purged_expanding_splits(metadata,min_train_sessions=252,validation_sessions=63):
    if min_train_sessions<2 or validation_sessions<1: raise ValueError('Invalid fold sizes')
    origins,ends = _times(metadata)
    calendar = origins.unique().sort_values()
    cursor = min_train_sessions
    while cursor+validation_sessions<=len(calendar):
        start = calendar[cursor]
        stop = calendar[cursor+validation_sessions-1]
        train = np.flatnonzero((origins<start)&(ends<start))
        test = np.flatnonzero((origins>=start)&(origins<=stop))
        if origins[train].nunique()>=min_train_sessions and len(test):
            yield train,test
        cursor += validation_sessions

def development_rows(metadata,final_test_start):
    origins,ends = _times(metadata)
    boundary = aware_index([final_test_start],'final_test_start')[0]
    return metadata.loc[(origins<boundary)&(ends<boundary)].copy()

def assert_disjoint_label_intervals(train_metadata,test_metadata):
    train_origin,train_end = _times(train_metadata)
    test_origin,_ = _times(test_metadata)
    if not len(train_origin) or not len(test_origin): raise ValueError('Empty split')
    if train_origin.max()>=test_origin.min() or train_end.max()>=test_origin.min():
        raise ValueError('Training label intervals overlap validation/test information')
    return True
