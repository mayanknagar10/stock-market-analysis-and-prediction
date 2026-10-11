"""Contract tests use isolated fixture data, never the real research archives."""
from datetime import datetime, timezone
import sqlite3
import pytest
from application.collection.calendar import Calendar, CalendarUnavailable, aware
from application.collection.store import CollectionStore
from application.collection.collector import Collector
from test_ledger import prediction

def test_calendar_origin_holidays_specials_and_unknown():
    calendar=Calendar()
    assert calendar.cutoff("2026-10-09").isoformat()=="2026-10-10T01:00:00+00:00"
    assert not calendar.session("2026-01-15")
    assert calendar.session("2026-02-01")
    assert not calendar.session("2026-10-10")
    with pytest.raises(CalendarUnavailable): calendar.session("2026-11-08")
    with pytest.raises(CalendarUnavailable): calendar.session("2027-01-01")
    with pytest.raises(ValueError): aware(datetime(2026,10,9))
    assert calendar.advance("2026-10-19",1).isoformat()=="2026-10-21"

def record(batch, ticker, horizon):
    row=prediction()
    row.update(ticker=ticker,horizon=horizon,model_version=batch["model_version"],
        origin_session=batch["session"],forecast_as_of=batch["cutoff"],
        information_cutoff=batch["cutoff"],generated_at="2026-10-10T01:01:00Z",
        protocol_version=batch["protocol_version"],batch_id=batch["batch_id"])
    return row

def test_atomic_membership_and_immutable_seal(tmp_path):
    store=CollectionStore(tmp_path)
    batch=store.begin("2026-10-09")
    ticker=store.members[0]["ticker"]
    with pytest.raises(ValueError): store.save_stock(batch,[record(batch,"FAKE.NS",h) for h in [1,5,10,20]])
    with pytest.raises(ValueError): store.save_stock(batch,[record(batch,ticker,1)])
    store.save_stock(batch,[record(batch,ticker,h) for h in [1,5,10,20]])
    sealed=store.seal(batch,"2026-10-10T01:05:00Z")
    assert sealed["successful_stocks"]==1 and sealed["failed_stocks"]==99
    assert sealed["abstained_forecasts"]==4 and sealed["health"]=="DEGRADED"
    assert len(sealed["members"])==100
    with pytest.raises(ValueError): store.save_stock(batch,[record(batch,store.members[1]["ticker"],h) for h in [1,5,10,20]])
    with sqlite3.connect(store.path) as con:
        with pytest.raises(sqlite3.IntegrityError): con.execute("DELETE FROM batches")
    assert store.verify(batch["batch_id"])["batch_id"]==batch["batch_id"]

def test_sealed_rerun_has_no_provider_or_inference(tmp_path):
    store=CollectionStore(tmp_path); batch=store.begin("2026-10-09")
    store.seal(batch,"2026-10-10T01:05:00Z")
    def forbidden(*args,**kwargs): raise AssertionError("No side effects")
    result=Collector(store,provider=forbidden,engine=forbidden,clock=lambda:aware("2026-10-10T01:06:00Z")).run("2026-10-09","finalize")
    assert result["status"]=="ALREADY_EXISTS"

def test_late_origin_records_every_unavailable_member(tmp_path):
    store=CollectionStore(tmp_path)
    result=Collector(store,clock=lambda:aware("2026-10-10T02:00:00Z")).run("2026-10-09","capture")
    assert result["health"]=="FAILED" and len(result["members"])==100
    assert all(m["code"]=="MISSED_WINDOW" for m in result["members"].values())
    assert store.ledger.rows()==[]

def test_partial_resume_only_missing_members(tmp_path):
    store=CollectionStore(tmp_path);batch=store.begin("2026-10-09")
    first=store.members[0]["ticker"]
    store.save_stock(batch,[record(batch,first,h) for h in [1,5,10,20]])
    store.freeze_sources(batch,{},{},"2026-10-10T00:59:00Z")
    calls=[]
    def engine(ticker,*args):
        calls.append(ticker)
        return [record(batch,ticker,h) for h in [1,5,10,20]]
    result=Collector(store,engine=engine,clock=lambda:aware("2026-10-10T01:01:00Z")).run("2026-10-09","finalize")
    assert first not in calls and len(calls)==99
    assert result["successful_forecasts"]==400
    assert len(store.ledger.rows())==400
    assert len({r["prediction_id"] for r in store.ledger.rows()})==400

def test_future_and_stale_snapshot_rejected():
    from application.collection.collector import validate_source
    from test_post_v5_archive import source
    snapshot=source()
    with pytest.raises(ValueError): validate_source(snapshot,aware("2026-10-08T09:00Z"),"2026-10-08",False)
    with pytest.raises(ValueError): validate_source(snapshot,aware("2026-10-09T01:00Z"),"2026-10-08",False)


def test_capture_fails_visibly_and_retries_are_bounded(tmp_path):
    calls=[]
    store=CollectionStore(tmp_path)
    def provider(symbol,reference,timeout):
        calls.append(symbol);raise TimeoutError("private error path")
    collector=Collector(store,provider=provider,clock=lambda:aware("2026-10-10T00:35:00Z"),sleep=lambda _:None)
    result=collector.run("2026-10-09","capture")
    assert result["captured_sources"]==0 and result["failed_sources"]>=100
    assert max(calls.count(x) for x in set(calls))==3
    sealed=Collector(store,clock=lambda:aware("2026-10-10T01:01:00Z")).run("2026-10-09","finalize")
    assert sealed["health"]=="FAILED" and len(sealed["members"])==100

def test_lock_excludes_simultaneous_writers(tmp_path):
    first=CollectionStore(tmp_path);second=CollectionStore(tmp_path)
    with first.writer():
        with pytest.raises(ValueError,match="BUSY"):
            with second.writer():pass

def test_non_session_is_skipped_without_provider_work(tmp_path):
    store=CollectionStore(tmp_path)
    result=Collector(store,clock=lambda:aware("2026-10-11T00:35Z")).run("2026-10-10","capture")
    assert result["health"]=="SKIPPED" and result["skipped_forecasts"]==400
    assert result["failed_forecasts"]==0

def test_rejected_canonical_history_schema_changes(tmp_path):
    from application.config import universe
    _,expected=universe()
    store=CollectionStore(tmp_path);batch=store.begin("2026-10-09")
    row=record(batch,store.members[0]["ticker"],1);row["confidence"]="HIGH"
    with pytest.raises(ValueError):store.save_stock(batch,[row]+[record(batch,row["ticker"],h) for h in [5,10,20]])
    assert store.rows()==[]


def test_capture_reuses_archived_sources_after_interruption(tmp_path):
    from test_automation_outcomes import snapshot
    store=CollectionStore(tmp_path);batch=store.begin("2026-10-09")
    ticker=store.members[0]["ticker"]
    snap=snapshot(ticker,["2026-10-09"],[100.],"2026-10-10T00:35Z")
    sid=store.archive.save_snapshot(snap)
    store.event("SOURCE_CAPTURED",batch_id=batch["batch_id"],source=ticker,snapshot_id=sid,archived_at="2026-10-10T00:36Z",attempt=1)
    calls=[]
    def provider(symbol,reference,timeout):
        calls.append(symbol);return snapshot(symbol,["2026-10-09"],[100.],"2026-10-10T00:40Z",reference=reference)
    result=Collector(store,provider=provider,clock=lambda:aware("2026-10-10T00:40Z"),sleep=lambda _:None).run("2026-10-09","capture")
    assert result["status"]=="INPUTS_FROZEN" and ticker not in calls
    assert store.sources(batch)["snapshots"][ticker]==sid

def test_retrieval_crossing_cutoff_never_enters_inputs(tmp_path):
    from test_automation_outcomes import snapshot
    store=CollectionStore(tmp_path)
    def provider(symbol,reference,timeout):
        return snapshot(symbol,["2026-10-09"],[100.],"2026-10-10T01:00:01Z",reference=reference)
    result=Collector(store,provider=provider,clock=lambda:aware("2026-10-10T00:59:50Z"),sleep=lambda _:None).run("2026-10-09","capture")
    assert result["captured_sources"]==0
    assert store.sources(store.begin("2026-10-09"))["snapshots"]=={}


def test_engine_completing_after_deadline_never_admitted(tmp_path):
    store=CollectionStore(tmp_path);batch=store.begin("2026-10-09")
    store.freeze_sources(batch,{},{},"2026-10-10T00:59:00Z")
    current=[aware("2026-10-10T01:29:00Z")]
    def engine(ticker,*args):
        current[0]=aware("2026-10-10T01:31:00Z")
        return [record(batch,ticker,h) for h in (1,5,10,20)]
    result=Collector(store,engine=engine,clock=lambda:current[0]).run("2026-10-09","finalize")
    assert result["health"]=="FAILED" and store.rows()==[]
    assert all(m["code"]=="MISSED_WINDOW" for m in result["members"].values())

def test_crashed_provider_attempts_count_against_retry_bound(tmp_path):
    from test_automation_outcomes import snapshot
    store=CollectionStore(tmp_path);batch=store.begin("2026-10-09");ticker=store.members[0]["ticker"]
    for attempt in range(1,4):store.event("SOURCE_ATTEMPT_STARTED",batch_id=batch["batch_id"],source=ticker,attempt=attempt)
    calls=[]
    def provider(symbol,reference,timeout):
        calls.append(symbol);return snapshot(symbol,["2026-10-09"],[100.],"2026-10-10T00:40Z",reference=reference)
    result=Collector(store,provider=provider,clock=lambda:aware("2026-10-10T00:40Z"),sleep=lambda _:None).run("2026-10-09","capture")
    assert ticker not in calls
    assert store.sources(batch)["failures"][ticker]["attempts"]==3


def test_provider_attempt_is_persisted_before_external_call(tmp_path):
    from test_automation_outcomes import snapshot
    store=CollectionStore(tmp_path);batch=store.begin("2026-10-09");observed=[]
    def provider(symbol,reference,timeout):
        events=store.source_events(batch)
        observed.append(any(e["code"]=="SOURCE_ATTEMPT_STARTED" and e.get("symbol")==symbol for e in events))
        return snapshot(symbol,["2026-10-09"],[100.],"2026-10-10T00:40Z",reference=reference)
    Collector(store,provider=provider,clock=lambda:aware("2026-10-10T00:40Z"),sleep=lambda _:None).run("2026-10-09","capture")
    assert observed and all(observed)
