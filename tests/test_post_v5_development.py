import numpy as np
import pandas as pd
import pytest

def fixture_frame():
    dates=pd.bdate_range('2023-01-01','2026-10-08')
    close=100*np.exp(np.arange(len(dates))*.0002+np.sin(np.arange(len(dates))*.13)*.01)
    return pd.DataFrame({'Open':close,'High':close*1.01,'Low':close*.99,'Close':close,'Volume':10000.},index=dates)

def test_consumed_period_never_changes_development_features_or_targets():
    from research.post_v5.development import design_frame
    original=fixture_frame();modified=original.copy();modified.loc[modified.index>='2025-10-01','Close']=1e100
    first=design_frame(original,5);second=design_frame(modified,5)
    pd.testing.assert_frame_equal(first,second)
    assert first.label_end.max()<pd.Timestamp('2025-10-01',tz='UTC')

def test_fixed_fold_purges_all_same_date_stock_labels():
    from research.post_v5.development import fixed_folds
    metadata=pd.DataFrame({'feature_time':pd.to_datetime(['2024-09-20','2024-09-25','2024-10-01','2024-10-02'],utc=True),
        'label_end':pd.to_datetime(['2024-09-27','2024-10-02','2024-10-08','2024-10-09'],utc=True)})
    train,test=list(fixed_folds(metadata,[['2024-10-01','2024-12-31']]))[0]
    assert train.tolist()==[0]
    assert test.tolist()==[2,3]

def test_forbidden_fold_or_development_row_rejects():
    from research.post_v5.development import fixed_folds
    metadata=pd.DataFrame({'feature_time':pd.to_datetime(['2025-10-01'],utc=True),'label_end':pd.to_datetime(['2025-10-03'],utc=True)})
    with pytest.raises(ValueError,match='consumed'): list(fixed_folds(metadata,[['2025-07-01','2025-09-30']]))
