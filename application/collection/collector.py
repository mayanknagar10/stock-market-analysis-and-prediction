"""Explicit bounded acquisition and finalization; never invoked by HTTP/page reads."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
import json
import time
import numpy as np
from application.config import ROOT
from application.collection.calendar import Calendar,CalendarUnavailable,aware
from core.data.providers import YahooMarketDataProvider
def utcnow(): return datetime.now(timezone.utc)
def validate_source(snapshot,cutoff,session,reference):
    import pandas as pd
    receipt=aware(snapshot.metadata.fetched_at)
    if receipt>=cutoff: raise ValueError("SOURCE_AFTER_CUTOFF")
    if receipt<cutoff-timedelta(minutes=30):raise ValueError("STALE_DATA")
    frame=snapshot.frame
    if frame.empty: raise ValueError("DATA_UNAVAILABLE")
    dates=frame.index.tz_convert(snapshot.metadata.exchange_timezone).date
    # Equity must include the exact complete origin, never a merely last available row.
    if not reference:
        positions=np.flatnonzero(dates==__import__("datetime").date.fromisoformat(session))
        if len(positions)!=1: raise ValueError("STALE_DATA")
        row=frame.iloc[int(positions[0])]
        if not np.isfinite(row[["Open","High","Low","Close","Adj Close","Volume","Dividends","Stock Splits"]].to_numpy(dtype=float)).all() or row.Volume<=0 or row.Close<=0 or row["Adj Close"]<=0:
            raise ValueError("DATA_QUALITY_FAILURE")
        if receipt<Calendar().close(session)+timedelta(minutes=20): raise ValueError("INCOMPLETE_SESSION")
    elif aware(snapshot.metadata.source_timestamp)>receipt:
        raise ValueError("SOURCE_OBSERVATION_AFTER_RECEIPT")
def bounded_provider(symbol,reference,timeout):
    def transport(ticker,period,interval):
        import yfinance as yf
        return yf.Ticker(ticker).history(period=period,interval=interval,auto_adjust=False,actions=True,timeout=timeout)
    return YahooMarketDataProvider(transport).capture_history(symbol,reference=reference)
class Collector:
    def __init__(self,store,provider=None,engine=None,clock=utcnow,sleep=time.sleep):
        self.store=store;self.provider=provider or bounded_provider
        if engine is None:
            from application.collection.engine import FrozenEngine
            engine=FrozenEngine(store)
        self.engine=engine;self.clock=clock;self.sleep=sleep;self.calendar=Calendar()
    def run(self,session=None,stage="capture"):
        if stage not in {"capture","finalize"}: raise ValueError("Invalid collection stage")
        session=session or self.calendar.origin_for(self.clock())
        with self.store.writer():
            if session<"2026-10-09":raise ValueError("PROTOCOL_NOT_ACTIVE")
            batch=self.store.begin(session);existing=self.store.sealed(batch["batch_id"])
            if existing:
                result=self.store.verify(batch["batch_id"])
                self.store.event("DUPLICATE_PREVENTED",batch_id=batch["batch_id"])
                return {**result,"status":"ALREADY_EXISTS"}
            from application.provenance import verify_runtime
            try:verify_runtime(batch)
            except ValueError:
                self.store.event("SYSTEM_FAILURE",batch_id=batch["batch_id"],reason="RUNTIME_IDENTITY_CHANGED")
                existing_tickers={r["ticker"] for r in self.store.rows(batch)}
                for m in self.store.members:
                    if m["ticker"] not in existing_tickers:self.store.fail(batch,m["ticker"],"RUNTIME_IDENTITY_CHANGED","provenance")
                return self.store.seal(batch,self.clock().isoformat(),"RUNTIME_IDENTITY_CHANGED")
            try: eligible=self.calendar.session(session)
            except CalendarUnavailable:
                for member in self.store.members: self.store.fail(batch,member["ticker"],"CALENDAR_UNCONFIRMED","calendar",timestamp=self.clock().isoformat())
                return self.store.seal(batch,self.clock().isoformat(),"CALENDAR_UNCONFIRMED")
            if not eligible: return self.store.seal(batch,self.clock().isoformat(),"NON_SESSION","SKIPPED")
            cutoff=aware(batch["cutoff"]);now=aware(self.clock());start=cutoff-timedelta(minutes=30);deadline=cutoff+timedelta(minutes=30)
            if stage=="capture":
                if self.store.sources(batch): return {**batch,"status":"INPUTS_ALREADY_FROZEN"}
                if now<start: return {**batch,"status":"NOT_DUE","next_due":start.isoformat()}
                if now>=cutoff: return self.store.seal(batch,self.clock().isoformat(),"MISSED_WINDOW")
                return self.capture(batch,cutoff)
            if now<cutoff: return {**batch,"status":"NOT_DUE","next_due":cutoff.isoformat()}
            if now>=deadline: return self.store.seal(batch,self.clock().isoformat(),"MISSED_WINDOW")
            source=self.store.sources(batch)
            if source is None: return self.store.seal(batch,self.clock().isoformat(),"INPUTS_NOT_CAPTURED")
            existing_tickers={r["ticker"] for r in self.store.rows(batch)}
            for member in self.store.members:
                ticker=member["ticker"]
                if ticker in existing_tickers: continue
                if aware(self.clock())>=deadline:
                    self.store.fail(batch,ticker,"MISSED_WINDOW","finalize",timestamp=self.clock().isoformat());continue
                if ticker not in source["snapshots"] and ticker in source["failures"]:
                    failure=source["failures"][ticker]
                    self.store.fail(batch,ticker,failure["code"],"capture",failure["provider"],failure["attempts"],failure["retry_status"],failure["timestamp"])
                    continue
                try:
                    inference_started=time.perf_counter()
                    records=self.engine(ticker,batch,source)
                    self.store.event("INFERENCE_LATENCY",batch_id=batch["batch_id"],ticker=ticker,milliseconds=(time.perf_counter()-inference_started)*1000)
                    if aware(self.clock())>=deadline:
                        self.store.fail(batch,ticker,"MISSED_WINDOW","ledger_admission",timestamp=self.clock().isoformat())
                        continue
                    self.store.save_stock(batch,records)
                except Exception as error:
                    code=str(error) if str(error) in {"STALE_DATA","SCHEMA_MISMATCH","MODEL_UNAVAILABLE","DATA_QUALITY_FAILURE","MISSING_MARKET_REFERENCE","MISSING_SECTOR_REFERENCE"} else "FORECAST_UNAVAILABLE"
                    self.store.fail(batch,ticker,code,"inference",provider="yahoo/yfinance",timestamp=self.clock().isoformat())
                    # Detailed tracebacks remain in private logs, not public batch metadata.
                    __import__("logging").getLogger(__name__).exception("Forecast unavailable for %s",ticker)
            return self.store.seal(batch,self.clock().isoformat())
    def capture(self,batch,cutoff):
        settings=json.loads((ROOT/"config/context_sources.json").read_text(encoding="utf-8"))
        sources={m["ticker"]:(m["ticker"],False) for m in self.store.members}
        sources.update({key:(value,True) for key,value in settings["sources"].items()})
        prior=self.store.source_events(batch)
        def fetch(item):
            key,(symbol,reference)=item
            previous=[e for e in prior if e.get("source")==key]
            successes=[e for e in previous if e["code"]=="SOURCE_CAPTURED"]
            if successes:
                saved=successes[0];snapshot=self.store.archive.load_snapshot(saved["snapshot_id"])
                validate_source(snapshot,cutoff,batch["session"],reference)
                if aware(saved["archived_at"])>=cutoff:raise ValueError("Late source event")
                return key,saved["snapshot_id"],None
            completed=max([e.get("attempt",0) for e in previous]+[0])
            executed=completed
            code=previous[-1].get("reason","PROVIDER_FAILURE") if previous else "MISSED_WINDOW"
            for attempt,delay in enumerate((0,20,60),1):
                if attempt<=completed:continue
                if aware(self.clock())+timedelta(seconds=delay)>=cutoff: break
                if delay:self.sleep(delay)
                now=aware(self.clock())
                if now>=cutoff:break
                self.store.event("SOURCE_ATTEMPT_STARTED",batch_id=batch["batch_id"],source=key,symbol=symbol,attempt=attempt,started_at=now.isoformat())
                executed=attempt
                sid=None
                try:
                    snapshot=self.provider(symbol,reference,max(.1,min(10,(cutoff-now).total_seconds())))
                    # Preserve failed payloads too, but never admit a postcut snapshot as input.
                    sid=self.store.archive.save_snapshot(snapshot)
                    archived=aware(self.clock())
                    validate_source(snapshot,cutoff,batch["session"],reference)
                    if archived>=cutoff: raise ValueError("SOURCE_AFTER_CUTOFF")
                    self.store.event("SOURCE_CAPTURED",batch_id=batch["batch_id"],source=key,snapshot_id=sid,
                        retrieved_at=snapshot.metadata.fetched_at,archived_at=archived.isoformat(),attempt=attempt)
                    return key,sid,None
                except Exception as error:
                    code=str(error) if str(error) in {"STALE_DATA","DATA_QUALITY_FAILURE","INCOMPLETE_SESSION","SOURCE_AFTER_CUTOFF"} else "PROVIDER_FAILURE"
                    self.store.event("SOURCE_ATTEMPT_FAILED",batch_id=batch["batch_id"],source=key,provider="yahoo/yfinance",
                        attempt=attempt,reason=code,rejected_snapshot_id=sid)
            return key,None,{"source":key,"ticker":symbol,"stage":"capture","code":code,
                "provider":"yahoo/yfinance","timestamp":self.clock().isoformat(),"attempts":executed,"retry_status":"EXHAUSTED" if executed>=3 else "WINDOW_CLOSED"}
        with ThreadPoolExecutor(max_workers=3) as pool: results=list(pool.map(fetch,sources.items()))
        snapshots={key:sid for key,sid,failure in results if sid}
        failures={key:failure for key,sid,failure in results if failure}
        for member in self.store.members:
            if member["ticker"] in failures:
                f=failures[member["ticker"]]
                self.store.fail(batch,member["ticker"],f["code"],f["stage"],f["provider"],f["attempts"],f["retry_status"],f["timestamp"])
        if aware(self.clock())>=cutoff:
            return self.store.seal(batch,self.clock().isoformat(),"MISSED_WINDOW")
        self.store.freeze_sources(batch,snapshots,failures,self.clock().isoformat())
        return {**batch,"status":"INPUTS_FROZEN","captured_sources":len(snapshots),"failed_sources":len(failures)}
