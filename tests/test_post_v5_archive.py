from datetime import datetime,timezone
import json
import pandas as pd
import pytest
from core.data.contracts import DataSnapshot,SourceMetadata

def source(fetched='2026-10-08T10:00:00+00:00',price=100.):
    index=pd.date_range('2026-10-01',periods=2,tz='UTC')
    frame=pd.DataFrame({'Open':price,'High':price+1,'Low':price-1,'Close':price,'Adj Close':price,'Volume':100.,'Dividends':0.,'Stock Splits':0.},index=index)
    meta=SourceMetadata(provider_id='fixture',provider_version='1',symbol='ABC.NS',family='india_equity',exchange_timezone='Asia/Kolkata',interval='1d',adjustment_policy='fixture',source_timestamp=index[-1].isoformat(),available_at=fetched,fetched_at=fetched,vintage_at=fetched,availability_verified=False,revision_history_verified=False)
    return DataSnapshot(frame,meta)

def test_future_retrieval_cannot_enter_a_shadow_run(tmp_path):
    from research.post_v5.archive import RunArchive
    with pytest.raises(ValueError,match='retrieval'):
        RunArchive(tmp_path).save_run({'ABC.NS':source()}, {},'2026-10-08T09:59:00+00:00','fixture')

def test_archived_revision_is_new_and_old_bytes_never_change(tmp_path):
    from research.post_v5.archive import RunArchive
    archive=RunArchive(tmp_path)
    first=archive.save_run({'ABC.NS':source()}, {'sector':'unavailable'},'2026-10-08T10:01:00+00:00','fixture')
    before=(tmp_path/'runs'/f'{first}.json').read_bytes()
    second=archive.save_run({'ABC.NS':source(price=102.)},{},'2026-10-08T10:02:00+00:00','fixture')
    assert first!=second
    assert (tmp_path/'runs'/f'{first}.json').read_bytes()==before
    assert archive.load_run(first)['failures']=={'sector':'unavailable'}

def test_tampered_run_or_snapshot_is_rejected(tmp_path):
    from research.post_v5.archive import RunArchive
    archive=RunArchive(tmp_path)
    identifier=archive.save_run({'ABC.NS':source()}, {},'2026-10-08T10:01:00+00:00','fixture')
    file=tmp_path/'runs'/f'{identifier}.json'
    file.write_text('{}')
    with pytest.raises(ValueError,match='integrity'): archive.load_run(identifier)

def test_preservation_detects_any_protected_file_change(tmp_path):
    from research.post_v5.preservation import seal,verify
    (tmp_path/'models/v5').mkdir(parents=True)
    file=tmp_path/'models/v5/model.json';file.write_text('original')
    target=tmp_path/'protected.json'
    seal(tmp_path,target,commit='fixture')
    assert verify(tmp_path,target)['verified_files']==1
    file.write_text('rewritten')
    with pytest.raises(ValueError,match='modified'): verify(tmp_path,target)


def test_missing_prices_are_preserved_exactly_and_snapshot_tampering_fails(tmp_path):
    import numpy as np
    from research.post_v5.archive import RunArchive
    snapshot=source();frame=snapshot.frame;frame.iloc[-1,frame.columns.get_loc('Close')]=np.nan
    dirty=DataSnapshot(frame,snapshot.metadata);archive=RunArchive(tmp_path)
    identifier=archive.save_snapshot(dirty)
    assert np.isnan(archive.load_snapshot(identifier).frame.Close.iloc[-1])
    file=tmp_path/'snapshots'/f'{identifier}.json';file.write_bytes(b'{}')
    with pytest.raises(ValueError,match='integrity'):archive.load_snapshot(identifier)
