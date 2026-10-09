"""HTTP contracts with provider fixtures; no network/model training in request paths."""
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from application.config import MODEL,PROTOCOL
from application.collection.store import CollectionStore
from application.services.prospective import ProspectiveService
from test_automation_collection import record

def setup(tmp_path):
    from application.services.analytics import AnalyticsService
    from api.main import create_app
    from test_automation_outcomes import snapshot
    from application.collection.calendar import Calendar
    store=CollectionStore(tmp_path);batch=store.begin("2026-10-09");ticker=store.members[0]["ticker"]
    records=[record(batch,ticker,h) for h in [1,5,10,20]]
    for r in records:
        r.update(source_observation_timestamp="2026-10-09T03:45:00Z",industry_at_freeze="Capital Goods",trust_reasons=["RESEARCH_ONLY"],raw_p_positive=.6)
    store.save_stock(batch,records);store.seal(batch,"2026-10-10T01:05:00Z")
    dates=pd.bdate_range("2026-08-01",periods=50).date.astype(str).tolist()
    def provider(symbol,reference,timeout):return snapshot(symbol,dates,np.arange(100.,150.),"2026-10-09T12:00:00Z",reference=reference)
    services=ProspectiveService(tmp_path);analytics=AnalyticsService(provider=provider)
    return TestClient(create_app(services,analytics)),ticker,services

def test_health_stock_search_history_and_technicals(tmp_path):
    client,ticker,service=setup(tmp_path)
    assert client.get("/api/v1/health").json()["research_state"]=="RESEARCH_ONLY"
    assert client.get("/api/v1/stocks/search",params={"q":"ABB"}).json()["total"]>=1
    assert client.get(f"/api/v1/stocks/{ticker}/overview").status_code==200
    history=client.get(f"/api/v1/stocks/{ticker}/history",params={"limit":5}).json()
    assert len(history["items"])==5 and pd.Timestamp(history["items"][0]["timestamp"]).utcoffset().total_seconds()==0
    assert client.get(f"/api/v1/stocks/{ticker}/technicals").status_code==200

def test_forecast_abstention_and_schema(tmp_path):
    client,ticker,service=setup(tmp_path)
    result=client.get(f"/api/v1/stocks/{ticker}/forecast")
    assert result.status_code==200
    response=result.json()
    assert response["recommendation_allowed"] is False and response["research_state"]=="RESEARCH_ONLY"
    assert len(response["horizons"])==4
    assert all(h["abstain"] and h["probability"] is None and not h["recommendation_allowed"] for h in response["horizons"])
    assert "feature_values" not in result.text and str(tmp_path) not in result.text
    assert client.get(f"/api/v1/stocks/{ticker}/forecast/history").status_code==200
    schema=client.get("/openapi.json").json()
    assert "ForecastResponse" in schema["components"]["schemas"]
    assert not any("train" in p for p in schema["paths"])

def test_validation_counts_only_forward_matured_and_structured_errors(tmp_path):
    client,ticker,service=setup(tmp_path)
    response=client.get("/api/v1/validation/prospective").json()
    assert response["counts"]["matured"]==0 and response["overall"]["mae"] is None
    assert response["evidence_state"]=="INSUFFICIENT_FORWARD_EVIDENCE"
    error=client.get("/api/v1/stocks/%5ENSEI/overview")
    assert error.status_code==422 and error.json()["error"]["code"]=="INVALID_TICKER"
    assert client.get("/api/v1/stocks/UNKNOWN.NS/forecast").json()["error"]["code"]=="MODEL_UNAVAILABLE"
    assert client.get(f"/api/v1/stocks/{ticker}/history?limit=10001").status_code==422

def test_api_never_captures_forecasts(tmp_path,monkeypatch):
    client,ticker,service=setup(tmp_path)
    from application.collection.collector import Collector
    monkeypatch.setattr(Collector,"run",lambda *a,**k:(_ for _ in ()).throw(AssertionError("No collection")))
    for endpoint in ["system/status","model/status","model/health","validation/summary","data/health","collection/status","markets/overview","markets/sectors"]:
        assert client.get("/api/v1/"+endpoint).status_code==200
    assert client.get(f"/api/v1/stocks/{ticker}/forecast/explanation").status_code==200


def test_authentication_and_provider_error_do_not_leak_paths(tmp_path,monkeypatch):
    client,ticker,service=setup(tmp_path)
    monkeypatch.setenv("STOCKPRO_API_TOKEN","fixture-only")
    assert client.get("/api/v1/health").status_code==200
    assert client.get("/api/v1/system/status").status_code==401
    assert client.get("/api/v1/system/status",headers={"Authorization":"Bearer fixture-only"}).status_code==200
    monkeypatch.delenv("STOCKPRO_API_TOKEN")
    def fail(*args):raise RuntimeError("secret C:/private/snapshots/token")
    client.app.state.analytics.data.provider=fail
    result=client.get("/api/v1/stocks/TCS.NS/overview")
    assert result.json()["error"]["code"]=="PROVIDER_FAILURE"
    assert "private" not in result.text and "traceback" not in result.text


def test_validation_scores_raw_probability_with_correct_label(tmp_path):
    from application.collection.store import CollectionStore
    from test_automation_collection import record
    from application.collection.outcomes import build_outcome
    from test_automation_outcomes import snapshot
    store=CollectionStore(tmp_path);batch=store.begin("2026-10-09");ticker=store.members[0]["ticker"]
    records=[record(batch,ticker,h) for h in (1,5,10,20)]
    for r in records:r.update(raw_p_positive=.6,p_positive=.9,p_positive_calibrated=.9,industry_at_freeze="Capital Goods")
    store.save_stock(batch,records);store.seal(batch,"2026-10-10T01:05Z")
    service=ProspectiveService(tmp_path);row=store.rows()[0]
    stock=snapshot(ticker,["2026-10-09","2026-10-12"],[100.,110.],"2026-10-13T12:00Z")
    market=snapshot("^NSEI",["2026-10-09","2026-10-12"],[100.,102.],"2026-10-13T12:00Z",reference=True)
    out=build_outcome(row,stock,market,None,"2026-10-13T12:00Z",{"stock":"s","market":"m"})
    store.ledger.attach_outcome(row["prediction_id"],out)
    result=service.validation().model_dump()
    assert result["counts"]["matured"]==1
    assert result["overall"]["probability_basis"]=="RAW_UNCALIBRATED_DIAGNOSTIC"
    assert result["overall"]["brier"]==__import__("pytest").approx(.16)
