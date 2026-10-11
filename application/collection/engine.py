"""Pure saved native V5 inference orchestration; no fitting and no Streamlit calls."""
from functools import lru_cache
import json
import numpy as np
import pandas as pd
from application.config import MODEL,PROTOCOL,ROOT
from application.collection.collector import validate_source
from application.collection.calendar import aware
from core.data.cleaning import research_bar_cleanup
from core.data.quality import validate_market_data
from core.forecasting.service import adjusted_frame,active_research,load_research_model
from core.forecasting.context import research_feature_frame,market_regime
from core.forecasting.inference import infer_bundle,model_contributions
from research.post_v5.shadow import record_prediction
def required_context_issues(names,statuses):
    issues=[]
    for key,status in statuses.items():
        prefix="market_" if key.startswith("market_") else "sector_" if key.startswith("sector_") else key+"_"
        if status in {"STALE","UNAVAILABLE","INSUFFICIENT_OR_INVALID_HISTORY"} and any(name.startswith(prefix) for name in names):issues.append(key)
    return issues
@lru_cache(maxsize=1)
def bundles():
    pointer=active_research()
    if pointer["model_version"]!=MODEL: raise ValueError("MODEL_UNAVAILABLE")
    return {h:load_research_model("india_equity",h,pointer) for h in (1,5,10,20)}
class FrozenEngine:
    def __init__(self,store):
        self.store=store;self.loaded={}
        self.settings=json.loads((ROOT/"config/context_sources.json").read_text(encoding="utf-8"))
    def __call__(self,ticker,batch,manifest):
        bid=batch["batch_id"]
        if bid not in self.loaded:
            self.loaded={bid:{key:self.store.archive.load_snapshot(sid) for key,sid in manifest["snapshots"].items()}}
        sources=self.loaded[bid]
        if ticker not in sources:raise ValueError("DATA_QUALITY_FAILURE")
        if "market_india" not in sources:raise ValueError("MISSING_MARKET_REFERENCE")
        snapshot=sources[ticker];cut=batch["cutoff"];validate_source(snapshot,aware(cut),batch["session"],False)
        raw,frame=adjusted_frame(snapshot,cut)
        if frame.index[-1].date().isoformat()!=batch["session"]:raise ValueError("STALE_DATA")
        calendar=None
        for key,peer in sources.items():
            if key==ticker or peer.metadata.family!="india_equity":continue
            try: _,part=adjusted_frame(peer,cut)
            except ValueError:continue
            dates=part.index[part.Volume>0]
            calendar=dates if calendar is None else calendar.union(dates)
        if calendar is None:raise ValueError("DATA_QUALITY_FAILURE")
        calendar=calendar[calendar>=frame.index.min()].sort_values()
        frame,cleanup=research_bar_cleanup(frame,calendar)
        raw=raw.loc[raw.index.tz_convert(snapshot.metadata.exchange_timezone).tz_localize(None).normalize().isin(frame.index)]
        if any(pd.Timestamp(day) in frame.index[-200:] for day in cleanup["unresolved_session_dates"]):raise ValueError("DATA_QUALITY_FAILURE")
        health=validate_market_data(raw,cut,calendar.tz_localize(snapshot.metadata.exchange_timezone).tz_convert("UTC").as_unit("ns"))
        if not health.eligible:raise ValueError("DATA_QUALITY_FAILURE")
        references={key:value for key,value in sources.items() if value.metadata.family=="research_reference"}
        features,context=research_feature_frame(frame,ticker,"india_equity",references,self.settings,research_only=True)
        member=next(m for m in self.store.members if m["ticker"]==ticker);current=float(raw.Close.iloc[-1]);outputs=[]
        for horizon,bundle in bundles().items():
            names=list(bundle["regression"].feature_names)
            if not set(names).issubset(features):raise ValueError("SCHEMA_MISMATCH")
            row=features.iloc[[-1]][names]
            if not np.isfinite(row.to_numpy()).all():raise ValueError("SCHEMA_MISMATCH")
            stale=any(status=="STALE" for key,status in context["source_statuses"].items()
                if any(name.startswith("market_" if key.startswith("market_") else "sector_" if key.startswith("sector_") else key+"_") for name in names))
            if required_context_issues(names,context["source_statuses"]):raise ValueError("STALE_DATA")
            result=infer_bundle(bundle,row,current,cut,data_eligible=True,research_only=True)
            if stale:result["trust"]["reasons"].append("CONTEXT_STALE")
            if context["sector_status"] in {"UNAVAILABLE","INSUFFICIENT_OR_INVALID_HISTORY","STALE"}:result["relative_forecasts"].pop("sector",None)
            record=record_prediction(snapshot,result,bundle["metadata"],row.iloc[0].to_dict(),horizon,cut,batch["session"],
                bid,manifest["snapshots"],market_regime(features).iloc[-1],
                {s.metadata.provider_id:s.metadata.provider_version for s in sources.values()},
                {**health.to_dict(),"context":context,"source_cleanup":cleanup},current_price=current)
            record.update(study_id=PROTOCOL,protocol_version=PROTOCOL,universe_hash=self.store.universe_hash,
                industry_at_freeze=member["industry"],sector_at_freeze=member["industry"],frozen_universe_member=True,
                calendar_version=batch["calendar_version"],calendar_status="REGISTERED_ORIGIN_AND_MATURITY",
                historical_feature_calendar="CAPTURED_PEER_PROXY_UNVERIFIED",price_timestamp=raw.index[-1].isoformat(),
                benchmark_source_key="market_india",sector_source_key=self.settings["sector_map"].get(ticker),
                relative_forecasts=result.get("relative_forecasts",{}),drivers=model_contributions(bundle,row),
                prediction_origin="DELAYED_AFTER_CLOSE",recommendation_allowed=False,return_convention="vendor_total_return_adj_close")
            from application.provenance import digest_document
            record.update(issuing_code_identity=batch["issuing_code_identity"],model_artifact_identity=batch["model_artifact_identity"],
                model_parameters_identity=batch["model_parameters_identity"],model_manifest_hash=batch["model_manifest_hashes"][str(horizon)],
                feature_vector_hash=digest_document(record["feature_values"]),ood_interval_expanded=result["ood_interval_expanded"])
            outputs.append(record)
        return outputs
