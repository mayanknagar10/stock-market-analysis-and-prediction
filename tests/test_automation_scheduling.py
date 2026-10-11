from application.collection.calendar import aware
from application.collection.store import CollectionStore
from jobs.scheduled_tick import tick
def test_dispatch_uses_indian_time_not_machine_timezone(tmp_path):
    store=CollectionStore(tmp_path)
    class Collector:
        def run(self,origin,stage):return {"origin":origin,"stage":stage}
    first=tick(store,clock=lambda:aware("2026-10-10T00:40Z"),collector=Collector())
    assert first["jobs"]==[{"origin":"2026-10-09","stage":"capture"}]
    second=tick(store,clock=lambda:aware("2026-10-10T01:01Z"),collector=Collector())
    assert second["jobs"]==[{"origin":"2026-10-09","stage":"finalize"}]
def test_downtime_marks_unavailable_never_backfills(tmp_path):
    store=CollectionStore(tmp_path)
    result=tick(store,clock=lambda:aware("2026-10-10T03:00Z"))
    assert result["jobs"][0]["health"]=="FAILED"
    assert store.rows()==[]
def test_before_protocol_start_does_not_create_past_forecasts(tmp_path):
    store=CollectionStore(tmp_path)
    assert tick(store,clock=lambda:aware("2026-10-09T11:00Z"))["status"]=="IDLE"
    assert store.rows()==[]
