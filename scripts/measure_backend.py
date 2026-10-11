"""Bounded API/provider timings and frozen inference parity; never trains or writes predictions."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from time import perf_counter
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from api.main import create_app
from application.services.analytics import AnalyticsService
from application.services.prospective import ProspectiveService
from application.collection.collector import bounded_provider
from core.forecasting.service import active_research,load_research_model
from core.forecasting.inference import infer_bundle
def measure(live=False):
    service=ProspectiveService();analytics=AnalyticsService()
    if not live:
        from core.data.contracts import DataSnapshot,SourceMetadata
        def fixture(symbol,reference,timeout):
            index=pd.bdate_range("2025-01-01",periods=450,tz="Asia/Kolkata").tz_convert("UTC")
            price=np.linspace(100.,150.,450)
            frame=pd.DataFrame({"Open":price,"High":price+1,"Low":price-1,"Close":price,"Adj Close":price,"Volume":100.,"Dividends":0.,"Stock Splits":0.},index=index)
            receipt=datetime.now(timezone.utc).isoformat()
            return DataSnapshot(frame,SourceMetadata(provider_id="performance_fixture",provider_version="1",symbol=symbol,
                family="research_reference" if reference else "india_equity",exchange_timezone="Asia/Kolkata",interval="1d",
                adjustment_policy="fixture",source_timestamp=index[-1].isoformat(),available_at=receipt,fetched_at=receipt,
                vintage_at=receipt,availability_verified=False,revision_history_verified=False))
        analytics.data.provider=fixture
    client=TestClient(create_app(service,analytics));statuses={}
    paths=["/stocks/ABB.NS/overview","/stocks/ABB.NS/history","/stocks/ABB.NS/technicals",
        "/stocks/ABB.NS/forecast?group=legacy","/validation/summary","/markets/overview"]
    for path in paths:
        statuses[path]=[]
        for _ in range(5):statuses[path].append(client.get("/api/v1"+path).status_code)
    replay=[];parity=True;rows=service.rows("legacy");pointer=active_research()
    for horizon in (1,5,10,20):
        record=next(r for r in rows if r["horizon"]==horizon)
        bundle=load_research_model(record["market"],horizon,pointer)
        vector=pd.DataFrame([record["feature_values"]])[list(bundle["regression"].feature_names)]
        for _ in range(5):
            start=perf_counter()
            result=infer_bundle(bundle,vector,record["current_price"],record["information_cutoff"],data_eligible=True,research_only=True)
            replay.append((perf_counter()-start)*1000)
            parity=parity and bool(np.isclose(result["central_log_return"],record["central_log_return"],rtol=1e-8,atol=1e-10))
    return {"measured_at":datetime.now(timezone.utc).isoformat(),"provider_mode":"BOUNDED_LIVE_PROVIDER" if live else "SYNTHETIC_PROVIDER_FIXTURE",
        "api_status_codes":statuses,"timings":analytics.telemetry.summary(),
        "native_frozen_inference_replay":{"n":len(replay),"median_ms":float(np.median(replay)),"p95_ms":float(np.percentile(replay,95)),
            "central_return_parity":parity,"origin_group":"legacy-manual-v1","model_parameters_changed":False},
        "limits":["Five requests per route; includes cold startup and warm cache.","Legacy saved retrieval measured because no on-time canonical origin has occurred.",
                  "Native replay is a correctness/latency check, not new research evaluation.","No public HTTP route performs inference/training.",
                  "Latency is not a capacity or provider reliability claim."]}
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--live",action="store_true");parser.add_argument("--tag",default="",choices=["","V2"]);args=parser.parse_args()
    output=measure(args.live);file=ROOT/("reports/automation/PERFORMANCE_"+("LIVE" if args.live else "FIXTURE")+("_"+args.tag if args.tag else "")+".json")
    if file.exists():raise SystemExit("Refuse to overwrite an earlier measurement")
    file.write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(output,indent=2))
