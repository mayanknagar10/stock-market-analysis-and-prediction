"""Read-only ledger/validation facade used by Streamlit and FastAPI."""
from datetime import datetime,timezone
import hashlib
import json
import sqlite3
import numpy as np
from application.config import ROOT,PROTOCOL,MODEL,data_root
from application.schemas import ForecastResponse,HorizonForecast,ValidationResponse,CollectionResponse
from core.validation.metrics import forecast_metrics
FINAL_START="2027-10-11";FINAL_END="2028-10-10"
class ServiceError(Exception):
    def __init__(self,code,status=503):
        super().__init__(code);self.code=code;self.status=status
def read_table(file,table):
    if not file.exists():return []
    if table not in {"forecasts","outcomes","batches","controls","events","outcome_annotations"}:raise ValueError("Unsupported table")
    with sqlite3.connect(file.as_uri()+"?mode=ro",uri=True) as con:
        columns={r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        if not columns:return []
        hashcol="payload_hash" if "payload_hash" in columns else "hash" if "hash" in columns else None
        rows=con.execute(f"SELECT payload{','+hashcol if hashcol else ''} FROM {table}").fetchall()
    result=[]
    for row in rows:
        if hashcol and hashlib.sha256(row[0].encode()).hexdigest()!=row[1]:raise ServiceError("SCHEMA_MISMATCH")
        result.append(json.loads(row[0]))
    return result
class ProspectiveService:
    def __init__(self,directory=None):
        self.directory=__import__("pathlib").Path(directory or data_root());self.path=self.directory/"collection.sqlite"
    def rows(self,group="canonical"):
        if group not in {"canonical","legacy"}:raise ServiceError("INVALID_ORIGIN_GROUP",422)
        path=self.path if group=="canonical" else ROOT/"data/research/prospective-india-20261008-v1/shadow_predictions.sqlite"
        records=read_table(path,"forecasts")
        if group=="canonical":
            batches=read_table(path,"batches")
            hashes={pid:sha for b in batches for pid,sha in b["prediction_hashes"].items()}
            records=[r for r in records if r["prediction_id"] in hashes and r.get("protocol_version")==PROTOCOL]
            for row in records:
                digest=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
                if digest!=hashes[row["prediction_id"]]:raise ServiceError("SCHEMA_MISMATCH")
        outcomes=read_table(path,"outcomes")
        # Outcome table payload omits prediction ID: read keyed rows without exposing paths.
        keyed={}
        if path.exists():
            with sqlite3.connect(path.as_uri()+"?mode=ro",uri=True) as con:
                for identity,payload in con.execute("SELECT prediction_id,payload FROM outcomes"):keyed[identity]=json.loads(payload)
        annotations={a["prediction_id"]:a for a in read_table(self.path,"outcome_annotations") if a.get("origin_group")==group}
        for row in records:
            outcome=keyed.get(row["prediction_id"])
            if outcome:
                annotation=annotations.get(row["prediction_id"])
                digest=hashlib.sha256(json.dumps(outcome,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
                if annotation and annotation["original_outcome_hash"]==digest:
                    outcome=dict(outcome,original_resolution_timestamp=outcome["resolved_at"],resolved_at=annotation["effective_resolved_at"],
                        timestamp_correction=annotation["annotation_id"])
                receipts=[outcome.get(k) for k in ("outcome_source_retrieved_at","benchmark_source_retrieved_at","sector_source_retrieved_at") if outcome.get(k)]+outcome.get("sector_peer_source_retrieved_at",[])
                if any(__import__("pandas").Timestamp(t)>__import__("pandas").Timestamp(outcome["resolved_at"]) for t in receipts):
                    outcome=dict(outcome,status="invalid_outcome_provenance",reason="OUTCOME_CLOCK_ORDER")
            row["evaluation_sealed"]=row.get("origin_session","")>=FINAL_START or bool(outcome and outcome.get("maturity_timestamp","")[:10]>=FINAL_START)
            row["outcome"]=None if row["evaluation_sealed"] else outcome
        return records
    def forecast(self,ticker,group="canonical"):
        rows=[r for r in self.rows(group) if r["ticker"]==ticker]
        if not rows:raise ServiceError("MODEL_UNAVAILABLE",404)
        latest=max(r["forecast_as_of"] for r in rows)
        rows=[r for r in rows if r["forecast_as_of"]==latest]
        # Legacy repeated manual requests have separate generated timestamps; return a single run.
        newest=max(r["generated_at"] for r in rows)
        run=next(r for r in rows if r["generated_at"]==newest).get("input_run_id")
        if group=="legacy":rows=[r for r in rows if r.get("input_run_id")==run]
        first=rows[0];horizons=[]
        for r in sorted(rows,key=lambda x:x["horizon"]):
            if not r["abstain"] or r["confidence"]!="LOW":raise ServiceError("SCHEMA_MISMATCH")
            horizons.append(HorizonForecast(prediction_id=r["prediction_id"],horizon=r["horizon"],
                predicted_log_return=r["central_log_return"],predicted_return=float(np.expm1(r["central_log_return"])),
                predicted_price=r["central_price"],quantile_log_returns=r["quantiles"],
                quantile_prices=[float(r["current_price"]*np.exp(v)) for v in r["quantiles"]],
                probability=r.get("p_positive_calibrated"),raw_probability=r.get("raw_p_positive",r["p_positive"]),
                calibration_state=r.get("calibration_state","UNVERIFIED"),abstention_reasons=r.get("trust_reasons",["RESEARCH_ONLY"]),
                ood_score=r.get("ood_score"),ood_status="UNKNOWN" if r.get("ood_score") is None else "OUT_OF_DOMAIN" if r["ood_score"]>1 else "IN_DOMAIN",
                feature_vector_hash=r.get("feature_vector_hash"),regime=r["regime"],
                data_quality_status=r["data_quality"].get("price_data_status","ELIGIBLE" if r["data_quality"].get("eligible") else "UNAVAILABLE"),
                training_cutoff=r["training_cutoff"],feature_schema=r["feature_schema_hash"],provenance_references=r["snapshot_ids"]))
        return ForecastResponse(ticker=ticker,current_price=first["current_price"],
            price_timestamp=first.get("price_timestamp") or __import__("pandas").Timestamp(first["origin_session"],tz="Asia/Kolkata" if first["market"]=="india_equity" else "America/New_York").tz_convert("UTC").isoformat(),
            prediction_origin=first["forecast_as_of"],information_cutoff=first["information_cutoff"],
            model_version=first["model_version"],protocol_version=first.get("protocol_version","legacy-manual-v1"),
            origin_session=first["origin_session"],origin_group=group,calendar_version=first.get("calendar_version"),issuing_code_identity=first.get("issuing_code_identity"),
            model_artifact_identity=first.get("model_artifact_identity"),horizons=horizons)
    def validation(self,group="canonical"):
        rows=self.rows(group)
        def scores(part):
            matured=[r for r in part if r["outcome"] and r["outcome"]["status"]=="matured" and not r["evaluation_sealed"]]
            result={"forecasts":len(part),"matured":len(matured),"pending":sum(r["outcome"] is None and not r["evaluation_sealed"] for r in part),
                "sealed":sum(r["evaluation_sealed"] for r in part),"invalid_outcomes":sum(r["outcome"] is not None and r["outcome"]["status"]!="matured" for r in part),"abstention_rate":float(np.mean([r["abstain"] for r in part])) if part else None,
                "actionable_rate":float(np.mean([not r["abstain"] for r in part])) if part else None,"sample_reliability":"INSUFFICIENT_OR_UNREVIEWED",
                "probability_basis":"RAW_UNCALIBRATED_DIAGNOSTIC","mae":None,"rmse":None,"directional_accuracy":None,"balanced_accuracy":None,
                "brier":None,"log_loss":None,"coverage_50":None,"coverage_80":None,"bias":None}
            if matured:
                raw=[r.get("raw_p_positive",r["p_positive"] if r.get("p_positive_calibrated") is None else None) for r in matured]
                raw_all=all(p is not None for p in raw)
                result["raw_probability_n"]=sum(p is not None for p in raw)
                result["calibrated_probability_n"]=sum(r.get("p_positive_calibrated") is not None for r in matured)
                result.update(forecast_metrics([r["outcome"]["actual_log_return"] for r in matured],
                    [r["central_log_return"] for r in matured],raw if raw_all else None,[r["quantiles"] for r in matured]))
            for name in ("market","sector"):
                errors=[r["outcome"].get(name+"_relative_error") for r in matured if r["outcome"].get(name+"_relative_error") is not None]
                result[name+"_relative_error_n"]=len(errors)
                result[name+"_relative_mae"]=float(np.mean(np.abs(errors))) if errors else None
            return result
        dimensions={"horizon":lambda r:str(r["horizon"]),"stock":lambda r:r["ticker"],
            "sector":lambda r:r.get("sector_at_freeze",r.get("industry_at_freeze","UNKNOWN")),
            "industry":lambda r:r.get("industry_at_freeze","UNKNOWN"),"month":lambda r:r["origin_session"][:7],
            "regime":lambda r:r["regime"],"confidence_bucket":lambda r:r["confidence"],
            "model_version":lambda r:r["model_version"],"protocol_version":lambda r:r.get("protocol_version","legacy-manual-v1")}
        groups={}
        for name,key in dimensions.items():
            groups[name]={value:scores([r for r in rows if key(r)==value]) for value in sorted({key(r) for r in rows})}
        requirements={"per_horizon_matured":1000,"stocks":60,"industries":10,"regimes":3,"distinct_origin_sessions":60,"data_quality_coverage":.8}
        eligible=group=="canonical"
        quality_coverage=sum(b["successful_stocks"] for b in read_table(self.path,"batches") if b["status"]!="SKIPPED")/max(1,100*sum(b["status"]!="SKIPPED" for b in read_table(self.path,"batches")))
        evidence={}
        for h in (1,5,10,20):
            part=[r for r in rows if r["horizon"]==h and r["outcome"] and r["outcome"]["status"]=="matured" and not r["evaluation_sealed"]]
            regime_set={r["regime"] for r in part if r["regime"].lower() not in {"unknown","unavailable"}}
            count={"matured":len(part),"stocks":len({r["ticker"] for r in part}),"industries":len({r.get("industry_at_freeze") for r in part if r.get("industry_at_freeze")}),
                "regimes":len(regime_set),"distinct_origin_sessions":len({r["origin_session"] for r in part})}
            evidence[str(h)]=count
            eligible=eligible and count["matured"]>=1000 and count["stocks"]>=60 and count["industries"]>=10 and count["regimes"]>=3 and count["distinct_origin_sessions"]>=60 and quality_coverage>=.8
        overall=scores(rows)
        return ValidationResponse(model_version=MODEL,protocol_version=PROTOCOL if group=="canonical" else "legacy-manual-v1",
            evidence_state="ENOUGH_FORWARD_EVIDENCE_FOR_REVIEW" if eligible else "INSUFFICIENT_FORWARD_EVIDENCE",
            origin_group=group,counts={"forecasts":len(rows),"matured":overall["matured"],"pending":overall["pending"],"sealed":overall["sealed"],"invalid_outcomes":overall["invalid_outcomes"],
                "horizon_evidence":evidence,"data_quality_coverage":quality_coverage},
            overall=overall,groups=groups,evidence_requirements=requirements,
            limitations=["Overlapping horizons and repeated stocks are dependent observations.","Counts do not establish statistical reliability.",
                "Industry labels serve as fixed sector proxies when official sector mappings are absent.",
                "Research only; evidence sufficiency cannot authorize production.","Legacy intraday and historical reconstructed research are separate."])
    def collection_status(self):
        batches=read_table(self.path,"batches");sealed={b["batch_id"] for b in batches}
        # Public status is allowlisted, never internal artifact paths or raw sources.
        keys=("batch_id","session","cutoff","model_version","protocol_version","calendar_version","started_at","finished_at","status",
              "expected_universe_size","successful_stocks","failed_stocks","successful_forecasts","abstained_forecasts","failed_forecasts","skipped_forecasts",
              "provider_failures","source_failures","issuing_code_identity","model_artifact_identity","model_parameters_identity","data_quality_failures","source_freshness","members","health","duplicate_prevention_events")
        opened=[{k:r[k] for k in ("batch_id","session","cutoff","started_at","model_version","protocol_version")} for r in read_table(self.path,"controls") if r["batch_id"] not in sealed]
        events=read_table(self.path,"events")
        duplicates=sum(e.get("code")=="DUPLICATE_PREVENTED" for e in events)
        return CollectionResponse(protocol_version=PROTOCOL,health=max(batches,key=lambda b:b["finished_at"])["health"] if batches else "NOT_STARTED",
            batches=[{**{k:b[k] for k in keys if k in b},"duplicate_prevention_events":sum(e.get("code")=="DUPLICATE_PREVENTED" and e.get("batch_id")==b["batch_id"] for e in events)} for b in batches],open_batches=opened,duplicate_prevention_events=duplicates)
