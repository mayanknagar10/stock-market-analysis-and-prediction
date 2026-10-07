from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from core.data.contracts import SourceMetadata, DataSnapshot, PointInTimeError
from core.data.snapshots import SnapshotStore
from core.data.quality import validate_market_data
from core.validation.contracts import ModelMetadata, schema_hash

def metadata(**kwargs):
    values = dict(provider_id='fixture_archive', provider_version='1', symbol='TEST.NS',
                  family='india_equity', exchange_timezone='Asia/Kolkata', interval='1d',
                  adjustment_policy='raw_ohlcv_with_adj_close_and_actions',
                  source_timestamp='2024-01-02T10:00Z', available_at='2024-01-02T10:05Z',
                  fetched_at='2024-01-02T10:06Z', vintage_at='2024-01-02T10:06Z',
                  availability_verified=True, revision_history_verified=True)
    values.update(kwargs)
    return SourceMetadata(**values)

def snapshot():
    frame = pd.DataFrame({'Close':[100.,101.], 'Adj Close':[100.,101.], 'Volume':[20.,30.]},
                         index=pd.DatetimeIndex(['2024-01-01T10:00Z','2024-01-02T10:00Z']))
    return DataSnapshot(frame,metadata())

def test_snapshot_vintage_cannot_be_replayed_before_it_was_known():
    with pytest.raises(PointInTimeError,match='vintage'):
        snapshot().assert_as_of('2024-01-02T10:05Z')
    assert snapshot().assert_as_of('2024-01-02T10:07Z') is True

def test_retrospective_available_at_estimates_do_not_pass_verified_gate():
    data = snapshot()
    data = DataSnapshot(data.frame,replace(data.metadata,availability_verified=False))
    with pytest.raises(PointInTimeError,match='verified'):
        data.assert_as_of('2024-01-03T00:00Z')

def test_metadata_requires_timezone_and_consistent_source_clock():
    with pytest.raises(ValueError): metadata(available_at='2024-01-01T10:00Z')
    with pytest.raises(ValueError): metadata(fetched_at='2024-01-02T10:00')

def test_replay_roundtrip_is_content_addressed_and_exact(tmp_path):
    store = SnapshotStore(tmp_path)
    data = snapshot()
    identifier = store.save(data)
    restored = store.load(identifier)
    pd.testing.assert_frame_equal(data.frame,restored.frame)
    assert restored.metadata==data.metadata
    assert store.save(data)==identifier

def test_corrupted_snapshot_is_rejected(tmp_path):
    store = SnapshotStore(tmp_path)
    identifier = store.save(snapshot())
    file = tmp_path/(identifier+'.json')
    file.write_bytes(file.read_bytes().replace(b'100.0',b'900.0'))
    with pytest.raises(ValueError,match='integrity'): store.load(identifier)

def test_snapshot_identifier_cannot_escape_store(tmp_path):
    with pytest.raises(ValueError): SnapshotStore(tmp_path).load('../outside')

def test_data_health_rejects_stale_missing_and_impossible_candles(candles):
    frame = candles.copy()
    frame.index = frame.index.tz_localize('UTC')
    frame['Adj Close']=frame.Close
    frame['Dividends']=0.
    frame['Stock Splits']=0.
    expected = frame.index
    now = expected[-1]+pd.Timedelta('1h')
    healthy = validate_market_data(frame,now,expected)
    assert healthy.eligible
    assert healthy.price_data_status=='OK'
    assert not validate_market_data(frame.iloc[:-1],now,expected).eligible
    bad = frame.copy()
    bad.iloc[-1,bad.columns.get_loc('High')]=1.
    assert not validate_market_data(bad,now,expected).eligible
    bad = frame.copy()
    bad.iloc[-1,bad.columns.get_loc('Volume')]=float('nan')
    assert not validate_market_data(bad,now,expected).eligible
    assert healthy.to_dict()['overall_quality_score']==1.

def test_adjusted_labels_ignore_mechanical_split_price_jump():
    dates = pd.date_range('2024-01-01',periods=25,tz='UTC')
    raw = np.r_[np.repeat(100.,12),np.repeat(50.,13)]
    frame = pd.DataFrame({'Open':raw,'High':raw*1.01,'Low':raw*.99,'Close':raw,
        'Adj Close':50.,'Volume':100.,'Dividends':0.,'Stock Splits':np.r_[np.zeros(12),2.,np.zeros(12)]},index=dates)
    health = validate_market_data(frame,dates[-1]+pd.Timedelta('1h'),dates)
    assert health.eligible
    from core.validation.targets import direct_return_targets
    targets = direct_return_targets(frame['Adj Close'],dates)
    assert targets.loc[dates[11],'target_1d']==0.

def test_unexplained_jump_abstains_instead_of_fabricating_adjustments():
    dates = pd.date_range('2024-01-01',periods=25,tz='UTC')
    close = np.r_[np.repeat(100.,12),np.repeat(200.,13)]
    frame = pd.DataFrame({'Open':close,'High':close*1.01,'Low':close*.99,'Close':close,
                          'Adj Close':close,'Volume':100.,'Dividends':0.,'Stock Splits':0.},index=dates)
    health = validate_market_data(frame,dates[-1]+pd.Timedelta('1h'),dates)
    assert not health.eligible
    assert 'abnormal_adjusted_jump' in health.critical_flags

def test_model_schema_and_training_cutoff_fail_closed():
    names=['rsi','return_5d']
    model = ModelMetadata(model_version='candidate-1',family='india_equity',horizon=5,
        feature_version='features-1',feature_names=tuple(names),feature_schema_hash=schema_hash(names),
        training_start='2020-01-01T00:00Z',training_cutoff='2024-01-01T00:00Z',
        created_at='2024-01-02T00:00Z',training_universe=('TEST.NS',),parameters={},validation_metrics={},
        code_version='abc',data_snapshot_ids=('snapshot1',),point_in_time_verified=False)
    model.assert_compatible(names,'india_equity','2024-02-01T00:00Z')
    with pytest.raises(ValueError): model.assert_compatible(names[::-1],'india_equity','2024-02-01T00:00Z')
    with pytest.raises(ValueError): model.assert_compatible(names,'us_equity','2024-02-01T00:00Z')
    with pytest.raises(ValueError): model.assert_compatible(names,'india_equity','2023-01-01T00:00Z')
    with pytest.raises(ValueError,match='point-in-time'): model.assert_production_eligible()


def test_replay_preserves_named_exchange_timezone_and_frequency(tmp_path):
    frame=pd.DataFrame({'Close':[100.,101.]},index=pd.date_range('2024-01-01',periods=2,tz='Asia/Kolkata'))
    data=DataSnapshot(frame,metadata())
    store=SnapshotStore(tmp_path)
    restored=store.load(store.save(data))
    pd.testing.assert_frame_equal(frame,restored.frame)

def test_incompatible_fallback_adjustment_and_unlabelled_routing_fail():
    from core.data.contracts import assert_compatible_fallback
    primary=metadata()
    fallback=metadata(provider_id='alternative',fallback_from='fixture_archive')
    assert assert_compatible_fallback(primary,fallback)
    with pytest.raises(ValueError,match='adjustment_policy'):
        assert_compatible_fallback(primary,replace(fallback,adjustment_policy='raw_only'))
    with pytest.raises(ValueError,match='provenance'):
        assert_compatible_fallback(primary,replace(fallback,fallback_from=None))

def test_duplicate_naive_and_future_price_timestamps_abstain(candles):
    frame=candles.copy()
    frame['Adj Close']=frame.Close
    frame['Dividends']=0.
    frame['Stock Splits']=0.
    expected=frame.index.tz_localize('UTC')
    as_of=expected[-1]+pd.Timedelta('1h')
    assert not validate_market_data(frame,as_of,expected).eligible
    frame.index=expected
    repeated=pd.concat([frame,frame.iloc[[-1]]])
    assert not validate_market_data(repeated,as_of,expected).eligible
    assert not validate_market_data(frame,expected[-2],expected[:-1]).eligible

def test_unsupported_quantile_and_market_families_fail_closed():
    from core.validation.targets import direct_return_targets
    dates=pd.date_range('2024-01-01',periods=25,tz='UTC')
    with pytest.raises(ValueError): direct_return_targets(pd.Series(100.,index=dates),dates,horizons=[60])
    with pytest.raises(ValueError): direct_return_targets(pd.Series(100.,index=dates),dates,horizons=[True])


def test_string_verification_flags_cannot_turn_into_true():
    with pytest.raises(ValueError,match='boolean'): metadata(availability_verified='false')
    with pytest.raises(ValueError,match='boolean'): metadata(revision_history_verified='false')

def test_external_frame_mutations_cannot_modify_frozen_snapshot():
    data=snapshot()
    original=data.frame
    original.loc[pd.Timestamp('2030-01-01T00:00Z')]=[900.,900.,50.]
    assert len(data.frame)==2
    assert data.assert_as_of('2024-01-02T10:07Z')

def test_model_verification_flags_require_actual_booleans():
    names=['return_1d']
    values=dict(model_version='candidate',family='india_equity',horizon=1,feature_version='1',
        feature_names=tuple(names),feature_schema_hash=schema_hash(names),training_start='2020-01-01T00:00Z',
        training_cutoff='2024-01-01T00:00Z',created_at='2024-01-02T00:00Z',
        training_universe=('TEST.NS',),parameters={},validation_metrics={},code_version='abc',
        data_snapshot_ids=('snap',))
    with pytest.raises(ValueError,match='boolean'): ModelMetadata(**values,point_in_time_verified='false')
    values['validation_metrics']={'final_test_passed':'false'}
    with pytest.raises(ValueError,match='boolean'): ModelMetadata(**values,point_in_time_verified=True)


def test_snapshot_store_rejects_categorical_metadata_loss(tmp_path):
    frame=pd.DataFrame({'sector':pd.Categorical(['Energy'],categories=['Tech','Energy','Bank'],ordered=True)},
                       index=pd.DatetimeIndex(['2024-01-02T10:00Z']))
    with pytest.raises(ValueError,match='categorical'):
        SnapshotStore(tmp_path).save(DataSnapshot(frame,metadata()))
    assert not list(tmp_path.glob('*.json'))

def test_verified_old_vintage_can_be_retrieved_later_without_rewriting_time():
    data=snapshot()
    revised=replace(data.metadata,fetched_at='2025-01-01T00:00Z')
    assert DataSnapshot(data.frame,revised).assert_as_of('2024-01-02T10:07Z')

def test_per_row_future_availability_is_rejected_even_if_header_is_old():
    data=snapshot()
    frame=data.frame
    frame['available_at']=pd.to_datetime(['2024-01-01T10:00Z','2025-01-01T10:00Z'],utc=True)
    with pytest.raises(PointInTimeError,match='row'):
        DataSnapshot(frame,data.metadata).assert_as_of('2024-01-02T10:07Z')
