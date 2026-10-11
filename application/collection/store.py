"""Separate operational controls and append-only canonical forecast/batch payloads."""
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid
from application.config import MODEL,PROTOCOL,data_root,universe
from application.collection.calendar import Calendar,aware
from core.forecasting.ledger import PredictionLedger
from research.post_v5.archive import RunArchive
def identity(*parts): return str(uuid.uuid5(uuid.NAMESPACE_URL,"stockpro:"+":".join(map(str,parts))))
def payload(value):
    text=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
    return text,hashlib.sha256(text.encode()).hexdigest()
class CollectionStore:
    def __init__(self,directory=None):
        self.directory=Path(directory or data_root());self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/"collection.sqlite"
        self.ledger=PredictionLedger(self.path);self.archive=RunArchive(self.directory/"archive")
        self.universe,self.universe_hash=universe();self.members=self.universe["members"]
        with self.connect() as con:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS controls(batch_id TEXT PRIMARY KEY,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sources(batch_id TEXT PRIMARY KEY,payload TEXT NOT NULL,hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS batches(batch_id TEXT PRIMARY KEY,payload TEXT NOT NULL,hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS outcome_annotations(annotation_id TEXT PRIMARY KEY,payload TEXT NOT NULL,hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,payload TEXT NOT NULL,hash TEXT NOT NULL);
            """)
            for table in ("sources","batches","events","outcome_annotations"):
                for action in ("UPDATE","DELETE"):
                    con.execute(f"CREATE TRIGGER IF NOT EXISTS no_{table}_{action} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT,'immutable {table}'); END")
    def connect(self):
        con=sqlite3.connect(self.path,timeout=30);con.execute("PRAGMA foreign_keys=ON");return con
    @contextmanager
    def writer(self):
        """OS lock released on crash, shared across capture/finalize/resolver processes."""
        file=(self.directory/"writer.lock").open("a+b")
        file.seek(0);file.write(b"0");file.flush();file.seek(0)
        try:
            if __import__("os").name=="nt":
                import msvcrt
                msvcrt.locking(file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except (OSError,BlockingIOError):
            file.close();raise ValueError("COLLECTION_BUSY")
        try: yield
        finally:
            file.seek(0)
            if __import__("os").name=="nt":
                import msvcrt
                msvcrt.locking(file.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(file.fileno(),fcntl.LOCK_UN)
            file.close()
    def event(self,code,**details):
        text,digest=payload({"code":code,"timestamp":datetime.now(timezone.utc).isoformat(),**details})
        with self.connect() as con: con.execute("INSERT INTO events(payload,hash) VALUES (?,?)",(text,digest))
    def source_events(self,batch):
        with self.connect() as con:rows=con.execute("SELECT payload,hash FROM events ORDER BY id").fetchall()
        events=[]
        for text,digest in rows:
            if hashlib.sha256(text.encode()).hexdigest()!=digest:raise ValueError("Event integrity failure")
            event=json.loads(text)
            if event.get("batch_id")==batch["batch_id"]:events.append(event)
        return events
    def begin(self,session):
        cut=Calendar().cutoff(session).isoformat()
        bid=identity(PROTOCOL,session,cut,MODEL,self.universe_hash)
        with self.connect() as con:
            found=con.execute("SELECT payload FROM controls WHERE batch_id=?",(bid,)).fetchone()
            if found: return json.loads(found[0])
            from application.provenance import runtime_identity
            record={**runtime_identity(),"batch_id":bid,"session":session,"cutoff":cut,"model_version":MODEL,"protocol_version":PROTOCOL,
                "calendar_version":Calendar.version,"universe_hash":self.universe_hash,
                "started_at":datetime.now(timezone.utc).isoformat(),"failures":{}}
            con.execute("INSERT INTO controls VALUES (?,?)",(bid,payload(record)[0]))
        return record
    def sealed(self,bid):
        with self.connect() as con: row=con.execute("SELECT payload,hash FROM batches WHERE batch_id=?",(bid,)).fetchone()
        if row:
            if hashlib.sha256(row[0].encode()).hexdigest()!=row[1]: raise ValueError("BATCH_INTEGRITY_FAILURE")
            return json.loads(row[0])
        return None
    def fail(self,batch,ticker,code,stage,provider=None,attempts=0,retry_status="EXHAUSTED",timestamp=None):
        if self.sealed(batch["batch_id"]): raise ValueError("Batch sealed")
        failure={"ticker":ticker,"code":code,"stage":stage,"timestamp":timestamp or datetime.now(timezone.utc).isoformat(),
            "provider":provider,"attempts":attempts,"retry_status":retry_status}
        with self.connect() as con:
            row=json.loads(con.execute("SELECT payload FROM controls WHERE batch_id=?",(batch["batch_id"],)).fetchone()[0])
            row["failures"][ticker]=failure
            con.execute("UPDATE controls SET payload=? WHERE batch_id=?",(payload(row)[0],batch["batch_id"]))
        self.event("STOCK_UNAVAILABLE",batch_id=batch["batch_id"],failure=failure)
    def freeze_sources(self,batch,snapshots,failures,archived_at):
        if aware(archived_at)>=aware(batch["cutoff"]): raise ValueError("Inputs not archived before cutoff")
        source={"snapshots":snapshots,"failures":failures,"archived_at":archived_at}
        text,digest=payload(source)
        with self.connect() as con:
            con.execute("INSERT INTO sources VALUES (?,?,?)",(batch["batch_id"],text,digest))
    def sources(self,batch):
        with self.connect() as con: row=con.execute("SELECT payload,hash FROM sources WHERE batch_id=?",(batch["batch_id"],)).fetchone()
        if row is None: return None
        if hashlib.sha256(row[0].encode()).hexdigest()!=row[1]: raise ValueError("SOURCE_MANIFEST_INTEGRITY_FAILURE")
        source=json.loads(row[0])
        if aware(source["archived_at"])>=aware(batch["cutoff"]): raise ValueError("Late archived sources")
        for sid in source["snapshots"].values(): self.archive.get("snapshots",sid)
        return source
    def save_stock(self,batch,records):
        if self.sealed(batch["batch_id"]): raise ValueError("Batch sealed")
        if len(records)!=4 or {r["horizon"] for r in records}!={1,5,10,20} or len({r["ticker"] for r in records})!=1:
            raise ValueError("Atomic four-horizon stock admission required")
        ticker=records[0]["ticker"]
        if ticker not in {m["ticker"] for m in self.members}: raise ValueError("Outside frozen universe")
        prepared=[]
        for r in records:
            if r["model_version"]!=MODEL or r["information_cutoff"]!=batch["cutoff"] or r["origin_session"]!=batch["session"]:
                raise ValueError("Canonical provenance mismatch")
            if not (aware(batch["cutoff"])<=aware(r["generated_at"])<aware(batch["cutoff"])+__import__("datetime").timedelta(minutes=30)):
                raise ValueError("Generation outside accepted window")
            r=dict(r, prediction_id=identity(PROTOCOL,ticker,batch["cutoff"],r["horizon"],MODEL),
                protocol_version=PROTOCOL,batch_id=batch["batch_id"],recommendation_allowed=False)
            if not r["abstain"] or r["confidence"]!="LOW": raise ValueError("Frozen research abstention required")
            prepared.append(r)
        self.ledger.append_many(prepared)
    def rows(self,batch=None):
        rows=self.ledger.rows()
        return [r for r in rows if not batch or r.get("batch_id")==batch["batch_id"]]
    def seal(self,batch,finished_at,missing_code="NOT_PROCESSED",status="COMPLETED"):
        found=self.sealed(batch["batch_id"])
        if found: return self.verify(batch["batch_id"])
        with self.connect() as con:
            control=json.loads(con.execute("SELECT payload FROM controls WHERE batch_id=?",(batch["batch_id"],)).fetchone()[0])
        rows=self.rows(batch);success={r["ticker"] for r in rows};members={}
        for member in self.members:
            ticker=member["ticker"]
            members[ticker]={"ticker":ticker,"status":"FORECAST_RECORDED"} if ticker in success else {
                "status":"UNAVAILABLE",**control["failures"].get(ticker,{"ticker":ticker,"code":missing_code,"stage":"finalize",
                    "timestamp":finished_at,"provider":None,"attempts":0,"retry_status":"CLOSED"})}
        source=self.sources(batch);failures=(source or {}).get("failures",{})
        data_degraded=any(any(s in {"STALE","UNAVAILABLE","INSUFFICIENT_OR_INVALID_HISTORY"} for s in r["data_quality"].get("context",{}).get("source_statuses",{}).values()) for r in rows)
        freshness={"archived_at":(source or {}).get("archived_at"),"sources":{}}
        for key,sid in (source or {}).get("snapshots",{}).items():
            meta=self.archive.get("snapshots",sid)["metadata"]
            freshness["sources"][key]={name:meta[name] for name in ("provider_id","fetched_at","source_timestamp")}
        attempts=self.source_events(batch)
        provider_failures=sum(e["code"]=="SOURCE_ATTEMPT_FAILED" and e.get("reason")=="PROVIDER_FAILURE" for e in attempts)
        integrity_failed=any(f["code"] in {"RUNTIME_IDENTITY_CHANGED","INTEGRITY_FAILURE","SYSTEM_FAILURE"} for f in control["failures"].values())
        metadata={**batch,"finished_at":finished_at,"status":status,"expected_universe_size":100,
            "successful_stocks":len(success),"failed_stocks":0 if status=="SKIPPED" else 100-len(success),"skipped_stocks":100 if status=="SKIPPED" else 0,"successful_forecasts":len(rows),
            "abstained_forecasts":sum(r["abstain"] for r in rows),"failed_forecasts":0 if status=="SKIPPED" else 400-len(rows),
            "skipped_forecasts":400 if status=="SKIPPED" else 0,"members":members,
            "provider_failures":provider_failures,"unresolved_source_failures":len(failures),"source_failures":failures,"data_quality_failures":sum(f["code"] in {"STALE_DATA","SCHEMA_MISMATCH","DATA_QUALITY_FAILURE"} for f in control["failures"].values()),
            "duplicate_prevention_events":0,"source_freshness":freshness,
            "prediction_hashes":{r["prediction_id"]:self.ledger._payload({k:v for k,v in r.items() if k!="outcome"})[1] for r in rows},
            "health":"FAILED" if integrity_failed else "SKIPPED" if status=="SKIPPED" else "HEALTHY" if len(success)==100 and not failures and not provider_failures and not data_degraded else "DEGRADED" if rows else "FAILED"}
        text,digest=payload(metadata)
        with self.connect() as con: con.execute("INSERT INTO batches VALUES (?,?,?)",(batch["batch_id"],text,digest))
        return self.verify(batch["batch_id"])
    def verify(self,bid):
        batch=self.sealed(bid)
        if not batch: raise ValueError("Unsealed batch")
        rows=self.rows(batch)
        hashes={r["prediction_id"]:self.ledger._payload({k:v for k,v in r.items() if k!="outcome"})[1] for r in rows}
        if hashes!=batch["prediction_hashes"] or len(batch["members"])!=100: raise ValueError("Batch ledger integrity failure")
        self.sources(batch)
        return batch
