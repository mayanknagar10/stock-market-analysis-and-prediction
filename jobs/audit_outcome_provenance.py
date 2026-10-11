"""Append a verified timestamp amendment; preserve original predictions/outcomes and return values."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sqlite3
import numpy as np
from application.collection.store import CollectionStore,payload,identity
from application.collection.calendar import Calendar,aware
from application.config import ROOT
from application.services.prospective import read_table
from research.post_v5.outcomes import endpoint_return,prices_for
def audit(store):
    path=ROOT/"data/research/prospective-india-20261008-v1/shadow_predictions.sqlite"
    forecasts={r["prediction_id"]:r for r in read_table(path,"forecasts")}
    preservation=json.loads((ROOT/"reports/automation/RESEARCH_PRESERVATION.json").read_text(encoding="utf-8"))["legacy_forecast_hashes"]
    corrected=[];invalid=[]
    with store.writer(),sqlite3.connect(path.as_uri()+"?mode=ro",uri=True) as con:
        for pid,text,stored_hash in con.execute("SELECT prediction_id,payload,payload_hash FROM outcomes"):
            outcome=json.loads(text);row=forecasts[pid]
            if payload(outcome)[1]!=stored_hash or payload(row)[1]!=preservation[pid]:raise ValueError("Original payload integrity failure")
            if row.get("origin_session","")>="2027-10-11" or outcome.get("maturity_timestamp","")[:10]>="2027-10-11":continue
            receipts=[outcome.get(k) for k in ("outcome_source_retrieved_at","benchmark_source_retrieved_at","sector_source_retrieved_at") if outcome.get(k)]
            if not any(aware(t)>aware(outcome["resolved_at"]) for t in receipts):continue
            annotation_id=identity("outcome-timestamp-amendment-v1",pid,stored_hash)
            with store.connect() as control:
                if control.execute("SELECT 1 FROM outcome_annotations WHERE annotation_id=?",(annotation_id,)).fetchone():continue
            if outcome.get("outcome_archive_namespace")!="prospective-automation":
                invalid.append(pid);continue
            try:
                stock=store.archive.load_snapshot(outcome["outcome_snapshot_id"])
                market=store.archive.load_snapshot(outcome["benchmark_snapshot_id"])
                if stock.metadata.fetched_at!=outcome["outcome_source_retrieved_at"] or market.metadata.fetched_at!=outcome["benchmark_source_retrieved_at"]:raise ValueError("Source receipt mismatch")
                target=Calendar().advance(row["origin_session"],row["horizon"]).isoformat()
                now=datetime.now(timezone.utc).isoformat()
                stock_value=endpoint_return(prices_for(stock,now),row["origin_session"],target)
                market_value=endpoint_return(prices_for(market,now),row["origin_session"],target)
                if not np.isclose(stock_value,outcome["actual_log_return"],atol=1e-10) or not np.isclose(market_value,outcome["benchmark_log_return"],atol=1e-10):raise ValueError("Outcome values fail source verification")
                annotation={"annotation_id":annotation_id,"prediction_id":pid,"origin_group":"legacy",
                    "kind":"VERIFIED_RESOLUTION_TIMESTAMP_AMENDMENT","original_outcome_hash":stored_hash,
                    "original_resolved_at":outcome["resolved_at"],"effective_resolved_at":now,"audited_at":now,
                    "reason":"Initial resolver used batch start before capture; effective timestamp is later audited acceptance, not claimed original execution time.",
                    "source_receipts":receipts,"financial_values_verified":True,"return_values_changed":False,"original_prediction_unchanged":True}
                text,digest=payload(annotation)
                with store.connect() as control:control.execute("INSERT INTO outcome_annotations VALUES (?,?,?)",(annotation_id,text,digest))
                corrected.append(pid)
            except Exception:invalid.append(pid)
        store.event("OUTCOME_PROVENANCE_AUDIT",corrected=len(corrected),invalid=len(invalid),return_values_changed=False)
    return {"audited_at":datetime.now(timezone.utc).isoformat(),"amended_count":len(corrected),"invalid_count":len(invalid),
        "amended_prediction_ids":corrected,"invalid_prediction_ids":invalid,"predictions_changed":False,"outcomes_rewritten":False,"return_values_changed":False}
def main():
    result=audit(CollectionStore())
    file=ROOT/"reports/automation/INITIAL_OUTCOME_TIMESTAMP_AUDIT.json"
    with file.open("x",encoding="utf-8") as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))
if __name__=="__main__":main()
