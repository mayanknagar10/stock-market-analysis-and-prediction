"""Timezone-independent five-minute scheduler entry point; no catch-up predictions."""
from datetime import timedelta
import json
import logging
from application.collection.calendar import Calendar,aware
from application.collection.collector import Collector,utcnow
from application.collection.store import CollectionStore
from application.collection.outcomes import Resolver
START_SESSION="2026-10-09"
def tick(store=None,clock=utcnow,collector=None,resolver=None):
    store=store or CollectionStore();calendar=Calendar();now=aware(clock())
    session=calendar.origin_for(now);cutoff=calendar.cutoff(session)
    collector=collector or Collector(store,clock=clock)
    results=[]
    # Record missed older eligible origins after machine downtime; never backfill inputs.
    day=__import__("datetime").date.fromisoformat(START_SESSION)
    last=__import__("datetime").date.fromisoformat(session)
    while day<last:
        origin=day.isoformat()
        try:eligible=calendar.session(day)
        except Exception:eligible=True
        if eligible:
            batch=store.begin(origin)
            if not store.sealed(batch["batch_id"]):results.append(collector.run(origin,"finalize"))
        day+=timedelta(days=1)
    if session>=START_SESSION:
        batch=store.begin(session)
        if not store.sealed(batch["batch_id"]):
            if now>=cutoff-timedelta(minutes=30):
                stage="capture" if now<cutoff else "finalize"
                results.append(collector.run(session,stage))
    local=now.astimezone(__import__("zoneinfo").ZoneInfo("Asia/Kolkata"))
    # One independent outcome attempt per Indian calendar day at/after20:30.
    # Clock delay never alters forecast cutoffs. A completed attempt is recorded
    # separately; source failures can be retried explicitly by the resolver job.
    marker=store.directory/("resolver-"+local.date().isoformat()+".json")
    if local.hour>=20 and (local.hour>20 or local.minute>=30) and not marker.exists():
        result=(resolver or Resolver(store,clock=clock,include_legacy=True)).run()
        with marker.open("x",encoding="utf-8") as stream:json.dump(result,stream,sort_keys=True)
        results.append(result)
    return {"timestamp":now.isoformat(),"status":"DISPATCHED" if results else "IDLE","jobs":results}
def main():
    try:
        result=tick();print(json.dumps(result,sort_keys=True,allow_nan=False))
        return 2 if any(job.get("health")=="FAILED" for job in result["jobs"]) else 0
    except Exception:
        logging.exception("Scheduled dispatcher failed");return 2
if __name__=="__main__":raise SystemExit(main())
