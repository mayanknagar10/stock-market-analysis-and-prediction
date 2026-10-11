from datetime import timedelta
import numpy as np
import pandas as pd
import pytest
from application.collection.calendar import Calendar,aware
from application.collection.outcomes import build_outcome
from core.data.contracts import DataSnapshot,SourceMetadata
from test_ledger import prediction

def snapshot(symbol,days,prices,fetched,splits=None,reference=False):
    index=pd.DatetimeIndex(days).tz_localize("Asia/Kolkata").tz_convert("UTC")
    frame=pd.DataFrame({"Open":prices,"High":np.array(prices)+1,"Low":np.array(prices)-1,
        "Close":prices,"Adj Close":prices,"Volume":100.,"Dividends":0.,"Stock Splits":splits or [0.]*len(days)},index=index)
    return DataSnapshot(frame,SourceMetadata(provider_id="fixture",provider_version="1",symbol=symbol,
        family="research_reference" if reference else "india_equity",exchange_timezone="Asia/Kolkata",interval="1d",
        adjustment_policy="vendor_adj_close",source_timestamp=index[-1].isoformat(),available_at=fetched,
        fetched_at=fetched,vintage_at=fetched,availability_verified=False,revision_history_verified=False))

@pytest.mark.parametrize("h",[1,5,10,20])
def test_exact_horizon_benchmark_maturity_and_no_original_mutation(h):
    calendar=Calendar();origin="2026-09-01";end=calendar.advance(origin,h)
    dates=[origin,end.isoformat()];receipt=(calendar.close(end)+timedelta(minutes=21)).isoformat()
    row=prediction();row.update(origin_session=origin,horizon=h,information_cutoff=calendar.cutoff(origin).isoformat(),
        generated_at=calendar.cutoff(origin).isoformat(),forecast_as_of=calendar.cutoff(origin).isoformat())
    before=dict(row)
    stock=snapshot("TEST.NS",dates,[100.,110.],receipt)
    market=snapshot("^NSEI",dates,[100.,102.],receipt,reference=True)
    out=build_outcome(row,stock,market,None,receipt,{"stock":"s","market":"m"},calendar)
    assert out["actual_return"]==pytest.approx(.1)
    assert out["benchmark_return"]==pytest.approx(.02)
    assert out["alpha_market"]==pytest.approx(np.log(1.1)-np.log(1.02))
    assert out["absolute_error"]==pytest.approx(abs(out["prediction_error_log"]))
    assert row==before
    assert build_outcome(row,stock,market,None,(calendar.close(end)-timedelta(minutes=1)).isoformat(),{},calendar) is None

def test_corporate_actions_and_missing_benchmark():
    calendar=Calendar();origin="2026-09-01";end=calendar.advance(origin,1).isoformat();receipt="2026-09-03T12:00Z"
    row=prediction();row.update(origin_session=origin,horizon=1,information_cutoff=calendar.cutoff(origin).isoformat(),
        generated_at=calendar.cutoff(origin).isoformat())
    stock=snapshot("TEST.NS",[origin,end],[100.,50.],receipt,[0.,2.])
    frame=stock.frame;frame["Adj Close"]=[50.,50.]
    stock=DataSnapshot(frame,stock.metadata)
    market=snapshot("^NSEI",[origin,end],[100.,101.],receipt,reference=True)
    out=build_outcome(row,stock,market,None,receipt,{"stock":"s","market":"m"},calendar)
    assert out["actual_return"]==0 and out["price_error"] is None
    assert out["corporate_actions_status"]=="PRESENT"
    assert build_outcome(row,stock,None,None,receipt,{},calendar) is None

def test_missing_exact_endpoint_never_shifts():
    calendar=Calendar();row=prediction();row.update(origin_session="2026-09-01",horizon=1,
        information_cutoff=calendar.cutoff("2026-09-01").isoformat(),generated_at=calendar.cutoff("2026-09-01").isoformat())
    stock=snapshot("TEST.NS",["2026-09-01","2026-09-03"],[100.,120.],"2026-09-04T12:00Z")
    assert build_outcome(row,stock,stock,None,"2026-09-04T12:00Z",{},calendar) is None


def test_sector_maturity_and_duplicate_outcome_prevention(tmp_path):
    from application.collection.store import CollectionStore
    from test_automation_collection import record
    store=CollectionStore(tmp_path);batch=store.begin("2026-10-09")
    ticker=store.members[0]["ticker"];rows=[record(batch,ticker,h) for h in (1,5,10,20)]
    store.save_stock(batch,rows);store.seal(batch,"2026-10-10T01:05Z")
    row=store.rows()[0];calendar=Calendar();end=calendar.advance("2026-10-09",1).isoformat()
    dates=["2026-10-09",end];receipt="2026-10-13T12:00Z"
    stock=snapshot(ticker,dates,[100.,110.],receipt)
    market=snapshot("^NSEI",dates,[100.,102.],receipt,reference=True)
    sector=snapshot("^SECTOR",dates,[100.,104.],receipt,reference=True)
    out=build_outcome(row,stock,market,sector,receipt,{"stock":"s","market":"m","sector":"sec"},calendar)
    assert out["sector_return"]==pytest.approx(.04)
    assert out["alpha_sector_arithmetic"]==pytest.approx(.06)
    store.ledger.attach_outcome(row["prediction_id"],out)
    with pytest.raises(ValueError):store.ledger.attach_outcome(row["prediction_id"],out)
    store.verify(batch["batch_id"])

def test_fixed_peer_basket_never_drops_missing_members():
    from research.post_v5.outcomes import peer_sector_return
    members=[{"ticker":t,"industry":"sector"} for t in ("A.NS","B.NS","C.NS")]
    prices={"B.NS":pd.Series([100.,105.],index=pd.to_datetime(["2026-09-01","2026-09-02"]))}
    assert peer_sector_return("A.NS","sector",members,prices,"2026-09-01","2026-09-02")["log_return"] is None


def test_legacy_job_resolves_separately_without_changing_prediction(tmp_path,monkeypatch):
    from application.collection.store import CollectionStore
    from application.collection.outcomes import Resolver
    import application.collection.outcomes as resolver_module
    from application.config import ROOT
    from core.forecasting.ledger import PredictionLedger
    from research.post_v5.archive import RunArchive
    from test_automation_collection import record
    store=CollectionStore(tmp_path/"canonical");batch=store.begin("2026-10-09");ticker=store.members[0]["ticker"]
    fake_root=tmp_path/"legacy-root";config=fake_root/"config/context_sources.json";config.parent.mkdir(parents=True)
    config.write_bytes((ROOT/"config/context_sources.json").read_bytes())
    legacy=fake_root/"data/research/prospective-india-20261008-v1"
    archive=RunArchive(legacy)
    original=snapshot(ticker,["2026-10-09"],[100.],"2026-10-10T00:40Z")
    run=archive.save_run({ticker:original},{},"2026-10-10T00:50Z","prospective-india-20261008-v1")
    ledger=PredictionLedger(legacy/"shadow_predictions.sqlite")
    records=[dict(record(batch,ticker,h),prediction_id=f"legacy-{h}",input_run_id=run,
        benchmark_source_key="market_india",industry_at_freeze=store.members[0]["industry"],sector_source_key=None) for h in (1,5,10,20)]
    ledger.append_many(records);before={r["prediction_id"]:ledger._payload({k:v for k,v in r.items() if k!="outcome"})[1] for r in ledger.rows()}
    def provider(symbol,reference,timeout):
        return snapshot(symbol,["2026-10-09","2026-10-12"],[100.,110.],"2026-10-13T12:00Z",reference=reference)
    monkeypatch.setattr(resolver_module,"ROOT",fake_root)
    resolver=Resolver(store,provider=provider,clock=lambda:aware("2026-10-13T12:00Z"),include_legacy=True)
    result=resolver.run()
    assert result["matured_by_origin_group"]=={"canonical":0,"legacy":1}
    assert ledger.rows()[0]["outcome"]["origin_group"]=="legacy"
    assert before=={r["prediction_id"]:ledger._payload({k:v for k,v in r.items() if k!="outcome"})[1] for r in ledger.rows()}
    assert resolver.run()["matured"]==0


def test_outcome_cannot_claim_resolution_before_source_receipt():
    calendar=Calendar();origin="2026-09-01";end=calendar.advance(origin,1).isoformat()
    row=prediction();row.update(origin_session=origin,horizon=1,information_cutoff=calendar.cutoff(origin).isoformat(),
        generated_at=calendar.cutoff(origin).isoformat())
    stock=snapshot("TEST.NS",[origin,end],[100.,110.],"2026-09-03T12:00Z")
    market=snapshot("^NSEI",[origin,end],[100.,102.],"2026-09-03T12:00Z",reference=True)
    assert build_outcome(row,stock,market,None,"2026-09-03T11:00Z",{},calendar) is None

def test_resolver_uses_post_capture_resolution_clock(tmp_path):
    from application.collection.store import CollectionStore
    from application.collection.outcomes import Resolver
    from test_automation_collection import record
    store=CollectionStore(tmp_path);batch=store.begin("2026-10-09");ticker=store.members[0]["ticker"]
    original=snapshot(ticker,["2026-10-09"],[100.],"2026-10-10T00:40Z")
    sid=store.archive.save_snapshot(original);store.freeze_sources(batch,{ticker:sid},{},"2026-10-10T00:50Z")
    rows=[record(batch,ticker,h) for h in (1,5,10,20)]
    for r in rows:r.update(benchmark_source_key="market_india",industry_at_freeze=store.members[0]["industry"],sector_source_key=None)
    store.save_stock(batch,rows);store.seal(batch,"2026-10-10T01:05Z")
    current=[aware("2026-10-13T11:00Z")]
    def provider(symbol,reference,timeout):
        current[0]=aware("2026-10-13T12:00Z")
        return snapshot(symbol,["2026-10-09","2026-10-12"],[100.,110.],current[0].isoformat(),reference=reference)
    assert Resolver(store,provider=provider,clock=lambda:current[0]).run()["matured"]==1
    out=store.rows()[0]["outcome"]
    assert aware(out["resolved_at"])>=aware(out["outcome_source_retrieved_at"])
    assert out["resolved_at"]=="2026-10-13T12:00:00+00:00"


def test_sector_peer_receipt_after_resolution_rejects_outcome():
    calendar=Calendar();origin="2026-09-01";end=calendar.advance(origin,1).isoformat()
    row=prediction();row.update(origin_session=origin,horizon=1,information_cutoff=calendar.cutoff(origin).isoformat(),
        generated_at=calendar.cutoff(origin).isoformat())
    stock=snapshot("TEST.NS",[origin,end],[100.,110.],"2026-09-03T12:00Z")
    market=snapshot("^NSEI",[origin,end],[100.,102.],"2026-09-03T12:00Z",reference=True)
    details={"log_return":.03,"source_receipts":["2026-09-03T13:00Z"]}
    assert build_outcome(row,stock,market,None,"2026-09-03T12:00Z",{},calendar,details) is None
