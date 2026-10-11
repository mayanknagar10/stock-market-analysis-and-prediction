"""Read an archived workspace for Streamlit; optional scenario calculations never persist forecasts."""
import json
import numpy as np
import pandas as pd
from application.config import ROOT,PROTOCOL,data_root
from application.services.prospective import ProspectiveService,ServiceError
from core.forecasting.service import adjusted_frame,load_research_model,active_research
from core.forecasting.inference import historical_analogues
from core.forecasting.scenarios import ScenarioEngine
from research.post_v5.archive import RunArchive
def load_workspace(ticker,group="canonical"):
    service=ProspectiveService();contract=service.forecast(ticker,group)
    selected={r.prediction_id for r in contract.horizons}
    records=[r for r in service.rows(group) if r["prediction_id"] in selected]
    archive=RunArchive(data_root()/"archive" if group=="canonical" else ROOT/"data/research/prospective-india-20261008-v1")
    snapshot_ids=records[0]["input_snapshot_ids"]
    sources={key:archive.load_snapshot(sid) for key,sid in snapshot_ids.items()}
    raw,frame=adjusted_frame(sources[ticker],records[0]["information_cutoff"])
    references={key:s for key,s in sources.items() if s.metadata.family=="research_reference"}
    settings=json.loads((ROOT/"config/context_sources.json").read_text(encoding="utf-8"))
    pointer=active_research();items=[]
    for record in records:
        bundle=load_research_model(record["market"],record["horizon"],pointer)
        row=pd.DataFrame([record["feature_values"]])[list(bundle["regression"].feature_names)]
        forecast={"horizon":record["horizon"],"central_log_return":record["central_log_return"],
            "central_return":float(np.expm1(record["central_log_return"])),"central_price":record["central_price"],
            "quantiles":record["quantiles"],"quantile_prices":[float(record["current_price"]*np.exp(v)) for v in record["quantiles"]],
            "p_positive":record["p_positive"],"raw_p_positive":record.get("raw_p_positive",record["p_positive"]),
            "p_positive_calibrated":record.get("p_positive_calibrated"),"ood_score":record["ood_score"],
            "ood_interval_expanded":record.get("ood_interval_expanded",False),
            "component_returns":record.get("component_returns",{}),"probability_components":record.get("probability_components",{}),
            "relative_forecasts":record.get("relative_forecasts",{}),"trust":{"status":"RESEARCH ONLY / ABSTAINED",
                "confidence":"LOW","abstain":True,"reasons":record.get("trust_reasons",["RESEARCH_ONLY"])}}
        items.append({"forecast":forecast,"record":record,"metadata":bundle["metadata"],
            "analogues":historical_analogues(bundle,row,record["information_cutoff"]),
            "scenario_engine":ScenarioEngine(bundle,frame,ticker,record["market"],references,settings,record["current_price"],record["information_cutoff"])})
    return {"ticker":ticker,"as_of":contract.information_cutoff.isoformat(),"origin_session":contract.origin_session,
        "origin_group":group,"current_price":contract.current_price,"forecasts":items,"health":records[0]["data_quality"],
        "context_health":records[0]["data_quality"].get("context",{}),"input_run_id":records[0]["input_run_id"]}
