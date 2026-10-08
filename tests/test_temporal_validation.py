import numpy as np
import pandas as pd
import pytest
from core.validation.targets import direct_return_targets
from core.validation.splits import purged_expanding_splits, development_rows
from core.validation.alignment import align_available, assert_feature_cutoff, session_cutoffs
from core.validation.preprocessing import fit_training_scaler

def test_direct_horizons_use_observed_exchange_sessions():
    dates = pd.date_range('2024-01-01',periods=25,tz='UTC')
    close = pd.Series(100*np.exp(np.arange(25)/100),index=dates)
    result = direct_return_targets(close,dates)
    for horizon in [1,5,10,20]:
        assert result.loc[dates[0],'target_'+str(horizon)+'d'] == pytest.approx(horizon/100)
        assert result.loc[dates[0],'label_end_'+str(horizon)+'d'] == dates[horizon]
        assert result['target_'+str(horizon)+'d'].iloc[-horizon:].isna().all()

def test_missing_sessions_do_not_shorten_targets():
    dates = pd.date_range('2024-01-01',periods=25,tz='UTC')
    prices = pd.Series(100.,index=dates.delete(5))
    with pytest.raises(ValueError,match='session'): direct_return_targets(prices,dates)

def test_invalid_adjusted_prices_are_rejected():
    dates = pd.date_range('2024-01-01',periods=25,tz='UTC')
    with pytest.raises(ValueError): direct_return_targets(pd.Series([0.]*25,index=dates),dates)

@pytest.mark.parametrize('horizon',[1,5,10,20])
def test_pooled_splits_purge_all_ticker_label_endpoints(horizon):
    dates = pd.date_range('2020-01-01',periods=150,tz='UTC')
    records = [{'ticker':ticker,'feature_time':dates[i], 'label_end':dates[i+horizon]}
               for i in range(150-horizon) for ticker in ['A','B']]
    metadata = pd.DataFrame(records)
    folds = list(purged_expanding_splits(metadata,min_train_sessions=30,validation_sessions=20))
    assert len(folds)>=2
    for train,test in folds:
        assert set(metadata.iloc[train].feature_time).isdisjoint(metadata.iloc[test].feature_time)
        assert metadata.iloc[train].label_end.max()<metadata.iloc[test].feature_time.min()
        assert metadata.iloc[train].feature_time.nunique()>=30
        assert set(metadata.iloc[test].ticker)=={'A','B'}

def test_final_period_and_overlapping_labels_are_sealed():
    records = pd.DataFrame({'feature_time':pd.to_datetime(['2024-01-01','2024-01-02','2024-01-03'],utc=True),
                            'label_end':pd.to_datetime(['2024-01-02','2024-01-03','2024-01-04'],utc=True)})
    dev = development_rows(records,pd.Timestamp('2024-01-03',tz='UTC'))
    assert dev.index.tolist()==[0]

def test_availability_join_uses_publication_time_instead_of_calendar_day():
    predictions = pd.DatetimeIndex(['2024-01-02T03:30:00Z','2024-01-02T12:00:00Z'])
    source = pd.DataFrame({'source_timestamp':pd.to_datetime(['2024-01-01T21:00Z','2024-01-02T10:00Z'],utc=True),
                           'available_at':pd.to_datetime(['2024-01-01T21:05Z','2024-01-02T10:05Z'],utc=True),
                           'nasdaq_return':[.01,-.04]})
    source['vintage_at']=source['available_at']
    joined = align_available(predictions,source,max_age=pd.Timedelta('24h'))
    assert joined.nasdaq_return.tolist()==[.01,-.04]
    assert (joined.available_at<=predictions).all()

def test_no_backward_filling_before_publication_and_stale_is_visible():
    times = pd.DatetimeIndex(['2024-01-01T00:00Z','2024-01-03T00:00Z'])
    source = pd.DataFrame({'source_timestamp':pd.to_datetime(['2024-01-01T20:00Z'],utc=True),
                           'available_at':pd.to_datetime(['2024-01-01T21:00Z'],utc=True),'value':[3.]})
    source['vintage_at']=source['available_at']
    result = align_available(times,source,max_age=pd.Timedelta('1h'))
    assert pd.isna(result.value.iloc[0])
    assert result.is_stale.tolist()==[True,True]

def test_future_feature_availability_hard_fails():
    times = pd.DatetimeIndex(['2024-01-01T10:00Z'])
    available = pd.DataFrame({'x':pd.to_datetime(['2024-01-01T10:01Z'],utc=True)},index=times)
    with pytest.raises(ValueError,match='future'): assert_feature_cutoff(times,available)

def test_naive_and_inconsistent_source_times_hard_fail():
    times = pd.DatetimeIndex(['2024-01-01T10:00Z'])
    source = pd.DataFrame({'available_at':[pd.Timestamp('2024-01-01')],
                           'source_timestamp':[pd.Timestamp('2024-01-01')],'x':[1]})
    source['vintage_at']=source['available_at']
    with pytest.raises(ValueError,match='timezone'): align_available(times,source)
    source[['available_at','source_timestamp','vintage_at']] = source[['available_at','source_timestamp','vintage_at']].apply(lambda s:pd.to_datetime(s,utc=True))
    source.loc[0,'source_timestamp']=pd.Timestamp('2024-01-02',tz='UTC')
    with pytest.raises(ValueError): align_available(times,source)

def test_exchange_timezone_and_us_dst_cutoffs():
    dates = pd.DatetimeIndex(['2024-03-08','2024-03-11'])
    us = session_cutoffs(dates,'us_equity')
    assert us[0]==pd.Timestamp('2024-03-08T21:00Z')
    assert us[1]==pd.Timestamp('2024-03-11T20:00Z')
    india = session_cutoffs(dates,'india_equity',phase='preopen')
    assert india[0]==pd.Timestamp('2024-03-08T03:30Z')

def test_scaler_uses_only_training_rows_before_boundary():
    values = pd.DataFrame({'x':[1.,2.,3.]})
    times = pd.date_range('2020-01-01',periods=3,tz='UTC')
    scaler = fit_training_scaler(values,times,times+pd.Timedelta('1D'),pd.Timestamp('2020-02-01',tz='UTC'))
    np.testing.assert_allclose(scaler.center_,[2.])
    assert scaler.transform(pd.DataFrame({'x':[999.]}))[0,0]>900
    with pytest.raises(ValueError):
        fit_training_scaler(values,times,times+pd.Timedelta('50D'),pd.Timestamp('2020-02-01',tz='UTC'))


def test_target_one_session_skips_weekend_without_guessing_weekdays():
    dates=pd.DatetimeIndex(['2024-01-05T21:00Z','2024-01-08T21:00Z','2024-01-09T21:00Z'])
    close=pd.Series([100.,110.,99.],index=dates)
    result=direct_return_targets(close,dates,horizons=[1])
    assert result.loc[dates[0],'target_1d']==pytest.approx(np.log(1.1))
    assert result.loc[dates[0],'label_end_1d']==dates[1]

def test_label_touching_test_start_is_purged():
    from core.validation.splits import assert_disjoint_label_intervals
    train=pd.DataFrame({'feature_time':[pd.Timestamp('2024-01-01T00:00Z')],
                        'label_end':[pd.Timestamp('2024-01-05T00:00Z')]})
    test=pd.DataFrame({'feature_time':[pd.Timestamp('2024-01-05T00:00Z')],
                       'label_end':[pd.Timestamp('2024-01-06T00:00Z')]})
    with pytest.raises(ValueError,match='overlap'): assert_disjoint_label_intervals(train,test)

def test_ordered_training_intervals_pass_contract():
    from core.validation.splits import assert_disjoint_label_intervals
    train=pd.DataFrame({'feature_time':[pd.Timestamp('2024-01-01T00:00Z')],
                        'label_end':[pd.Timestamp('2024-01-04T00:00Z')]})
    test=pd.DataFrame({'feature_time':[pd.Timestamp('2024-01-05T00:00Z')],
                       'label_end':[pd.Timestamp('2024-01-06T00:00Z')]})
    assert assert_disjoint_label_intervals(train,test)


def test_revision_vintage_is_not_joined_into_earlier_information_state():
    predictions=pd.DatetimeIndex(['2024-01-02T12:00Z','2025-01-02T12:00Z'])
    source=pd.DataFrame({'source_timestamp':pd.to_datetime(['2024-01-01T21:00Z'],utc=True),
        'available_at':pd.to_datetime(['2024-01-01T21:05Z'],utc=True),
        'vintage_at':pd.to_datetime(['2025-01-01T21:05Z'],utc=True),'value':[99.]})
    joined=align_available(predictions,source,max_age=pd.Timedelta('24h'))
    assert pd.isna(joined.value.iloc[0])
    assert joined.value.iloc[1]==99.
    assert joined.is_stale.tolist()==[True,True]


def test_historical_alignment_requires_revision_vintage_evidence():
    times=pd.DatetimeIndex(['2024-01-01T10:00Z'])
    source=pd.DataFrame({'source_timestamp':pd.to_datetime(['2024-01-01T09:00Z'],utc=True),
        'available_at':pd.to_datetime(['2024-01-01T09:01Z'],utc=True),'value':[1.]})
    with pytest.raises(ValueError,match='vintage'): align_available(times,source)


def test_equivalent_timezone_indices_align_to_same_information_instant():
    india=pd.DatetimeIndex(['2024-01-01T15:30:00+05:30']).tz_convert('Asia/Kolkata')
    available=pd.DataFrame({'x':pd.DatetimeIndex(['2024-01-01T10:00Z'])},index=india)
    assert assert_feature_cutoff(india,available)


def test_late_revision_to_older_observation_does_not_replace_latest_state():
    predictions=pd.DatetimeIndex(['2024-01-02T12:00Z','2024-01-03T12:00Z'])
    source=pd.DataFrame({
        'source_timestamp':pd.to_datetime(['2024-01-01T09:00Z','2024-01-02T09:00Z','2024-01-01T09:00Z'],utc=True),
        'available_at':pd.to_datetime(['2024-01-01T10:00Z','2024-01-02T10:00Z','2024-01-01T10:00Z'],utc=True),
        'vintage_at':pd.to_datetime(['2024-01-01T10:00Z','2024-01-02T10:00Z','2024-01-03T10:00Z'],utc=True),
        'value':[1.,2.,10.]})
    joined=align_available(predictions,source)
    assert joined.value.tolist()==[2.,2.]
    assert (joined.source_timestamp==pd.Timestamp('2024-01-02T09:00Z')).all()

def test_revision_of_latest_observation_updates_only_after_release():
    predictions=pd.DatetimeIndex(['2024-01-02T12:00Z','2024-01-03T12:00Z'])
    source=pd.DataFrame({
        'source_timestamp':pd.to_datetime(['2024-01-02T09:00Z','2024-01-02T09:00Z'],utc=True),
        'available_at':pd.to_datetime(['2024-01-02T10:00Z','2024-01-02T10:00Z'],utc=True),
        'vintage_at':pd.to_datetime(['2024-01-02T10:00Z','2024-01-03T10:00Z'],utc=True),
        'value':[2.,20.]})
    assert align_available(predictions,source).value.tolist()==[2.,20.]


def test_microsecond_and_nanosecond_timestamps_join_without_rounding():
    predictions=pd.DatetimeIndex(['2024-01-01T10:00:00.000000001Z'])
    source=pd.DataFrame({'source_timestamp':pd.DatetimeIndex(['2024-01-01T09:00Z']).as_unit('us'),
        'available_at':pd.DatetimeIndex(['2024-01-01T10:00Z']).as_unit('us'),
        'vintage_at':pd.DatetimeIndex(['2024-01-01T10:00:00.000000001Z']),'value':[1.]})
    assert align_available(predictions,source).value.iloc[0]==1.
    before=pd.DatetimeIndex(['2024-01-01T10:00:00Z']).as_unit('us')
    assert pd.isna(align_available(before,source).value.iloc[0])


def test_weekend_special_session_uses_conservative_information_cutoff():
    cutoff=session_cutoffs(pd.DatetimeIndex(['2024-11-03']),'india_equity')
    assert cutoff[0]==pd.Timestamp('2024-11-04T00:00Z')
