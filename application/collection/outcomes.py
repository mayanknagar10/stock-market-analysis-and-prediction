"""Independent exact-session outcome maturation, separate from immutable predictions."""
from datetime import timedelta
import json
import logging
import numpy as np
import pandas as pd
from application.collection.calendar import Calendar,CalendarUnavailable,aware
from application.collection.collector import bounded_provider,utcnow
from application.config import ROOT
from core.forecasting.ledger import resolve_outcome
from research.post_v5.outcomes import prices_for,endpoint_return,peer_sector_return,revision_diagnostics
def build_outcome(record,stock,market,sector,asof,snapshot_ids,calendar=None,sector_details=None):
    calendar=calendar or Calendar()
    receipts=[stock.metadata.fetched_at]+([market.metadata.fetched_at] if market else [])+([sector.metadata.fetched_at] if sector else [])
    if (sector_details or {}).get("log_return") is not None:receipts+=sector_details.get("source_receipts",[])
    if any(aware(receipt)>aware(asof) for receipt in receipts):return None
    end=calendar.advance(record["origin_session"],record["horizon"])
    maturity=calendar.close(end)+timedelta(minutes=20)
    if market is None or min(aware(asof),aware(stock.metadata.fetched_at),aware(market.metadata.fetched_at))<maturity:return None
    stocks=prices_for(stock,asof);markets=prices_for(market,asof)
    actual=endpoint_return(stocks,record["origin_session"],end.isoformat())
    benchmark=endpoint_return(markets,record["origin_session"],end.isoformat())
    if actual is None or benchmark is None:return None
    days=[pd.Timestamp(record["origin_session"])]
    day=__import__("datetime").date.fromisoformat(record["origin_session"])
    while day<end:
        day+=timedelta(days=1)
        if calendar.session(day):days.append(pd.Timestamp(day))
    result=resolve_outcome(record,stock.frame,min(aware(asof),aware(stock.metadata.fetched_at)).isoformat(),
        benchmark=markets,expected_sessions=pd.DatetimeIndex(days))
    if result is None or result["status"]!="matured":return None
    sector_return=endpoint_return(prices_for(sector,asof),record["origin_session"],end.isoformat()) if sector else (sector_details or {}).get("log_return")
    error=result["prediction_error_log"]
    result.update(resolved_at=aware(asof).isoformat(),maturity_timestamp=maturity.isoformat(),
        actual_log_return=actual,actual_return=float(np.expm1(actual)),absolute_error=abs(error),squared_error=error**2,
        benchmark_log_return=benchmark,benchmark_return=float(np.expm1(benchmark)),alpha_market=actual-benchmark,
        alpha_market_arithmetic=float(np.expm1(actual)-np.expm1(benchmark)),
        sector_log_return=sector_return,sector_return=float(np.expm1(sector_return)) if sector_return is not None else None,
        alpha_sector=actual-sector_return if sector_return is not None else None,
        alpha_sector_arithmetic=float(np.expm1(actual)-np.expm1(sector_return)) if sector_return is not None else None,
        sector_reference=sector_details or {"policy":"Captured sector price index","source":record.get("sector_source_key")},
        sector_peer_source_retrieved_at=(sector_details or {}).get("source_receipts",[]) if sector_return is not None else [],
        benchmark_policy="Captured Nifty price index, not TRI",
        return_convention="vendor_total_return_adj_close",price_convention="raw_close; price_error null if corporate actions or revisions",
        outcome_source_retrieved_at=stock.metadata.fetched_at,benchmark_source_retrieved_at=market.metadata.fetched_at,
        sector_source_retrieved_at=sector.metadata.fetched_at if sector else None,
        outcome_snapshot_id=snapshot_ids.get("stock"),benchmark_snapshot_id=snapshot_ids.get("market"),
        outcome_source_snapshot_ids=snapshot_ids,calendar_version=calendar.version,
        market_relative_error=None,sector_relative_error=None)
    for name,observed in [("market",result["alpha_market"]),("sector",result["alpha_sector"])]:
        prediction=record.get("relative_forecasts",{}).get(name,{})
        predicted=prediction.get("central_excess_log_return")
        if predicted is not None and observed is not None:result[name+"_relative_error"]=float(predicted-observed)
    return result
class Resolver:
    def __init__(self,store,provider=bounded_provider,clock=utcnow,include_legacy=False):
        self.store=store;self.provider=provider;self.clock=clock;self.calendar=Calendar();self.include_legacy=include_legacy
    def run(self):
        with self.store.writer():
            now=aware(self.clock());rows=self.store.rows();due=[];pending=0
            legacy_ledger=None
            if self.include_legacy:
                from core.forecasting.ledger import PredictionLedger
                from research.post_v5.archive import RunArchive
                legacy_path=ROOT/"data/research/prospective-india-20261008-v1/shadow_predictions.sqlite"
                if legacy_path.exists():
                    legacy_ledger=PredictionLedger(legacy_path)
                    legacy_archive=RunArchive(legacy_path.parent)
                    rows.extend(dict(r,resolution_origin_group="legacy") for r in legacy_ledger.rows())
            # Only sealed canonical batches can be evaluated.
            for row in rows:
                if row["outcome"] is not None:continue
                if row.get("resolution_origin_group")!="legacy" and not self.store.sealed(row.get("batch_id")):continue
                try: target=self.calendar.close(self.calendar.advance(row["origin_session"],row["horizon"]))+timedelta(minutes=20)
                except CalendarUnavailable:
                    pending+=1;continue
                if now>=target:due.append(row)
                else:pending+=1
            if not due:return {"status":"NO_MATURED_FORECASTS","matured":0,"pending":pending,"resolved_at":now.isoformat()}
            settings=json.loads((ROOT/"config/context_sources.json").read_text(encoding="utf-8"))
            sources={m["ticker"]:(m["ticker"],False) for m in self.store.members}
            sources.update({key:(symbol,True) for key,symbol in settings["sources"].items()})
            captured={};ids={};failures={}
            for key,(symbol,reference) in sources.items():
                for attempt in range(1,4):
                    try:
                        snap=self.provider(symbol,reference,10)
                        ids[key]=self.store.archive.save_snapshot(snap);captured[key]=snap;break
                    except Exception:
                        failures[key]={"code":"PROVIDER_FAILURE","attempts":attempt,"provider":"yahoo/yfinance"}
            now=aware(self.clock())
            prices={}
            for key,snap in captured.items():
                try:prices[key]=prices_for(snap,now.isoformat())
                except (ValueError,TypeError):failures[key]={"code":"DATA_QUALITY_FAILURE","provider":snap.metadata.provider_id}
            matured=0;group_counts={"canonical":0,"legacy":0}
            for row in due:
                ticker=row["ticker"];market_key=row.get("benchmark_source_key","market_india");sector_key=row.get("sector_source_key")
                try:
                    if ticker not in captured:raise ValueError("STOCK_OUTCOME_UNAVAILABLE")
                    group=row.get("resolution_origin_group","canonical")
                    if group=="legacy":
                        source=legacy_archive.get("runs",row["input_run_id"])
                    else:
                        batch=self.store.verify(row["batch_id"]);source=self.store.sources(batch)
                    end=self.calendar.advance(row["origin_session"],row["horizon"]).isoformat()
                    sector_snap=captured.get(sector_key)
                    sector_details=None
                    if sector_snap is None or endpoint_return(prices_for(sector_snap,now.isoformat()),row["origin_session"],end) is None:
                        sector_snap=None
                        sector_details=peer_sector_return(ticker,row.get("industry_at_freeze"),self.store.members,prices,row["origin_session"],end)
                        sector_details["source_receipts"]=[captured[m["ticker"]].metadata.fetched_at for m in self.store.members if m["industry"]==row.get("industry_at_freeze") and m["ticker"]!=ticker and m["ticker"] in captured]
                    resolved_at=aware(self.clock()).isoformat()
                    out=build_outcome(row,captured[ticker],captured.get(market_key),sector_snap,resolved_at,
                        {"stock":ids.get(ticker),"market":ids.get(market_key),"sector":ids.get(sector_key),
                         "peers":{m["ticker"]:ids.get(m["ticker"]) for m in self.store.members if m["industry"]==row.get("industry_at_freeze") and m["ticker"]!=ticker}},self.calendar,sector_details)
                    if out is None:pending+=1;continue
                    original=(legacy_archive if group=="legacy" else self.store.archive).load_snapshot(source["snapshots"][ticker])
                    revisions=revision_diagnostics(original,captured[ticker],row["origin_session"])
                    out["origin_revision_diagnostics"]=revisions
                    if any(field in revisions["fields_changed"] for field in ("Open","High","Low","Close")):out["price_error"]=None
                    out.update(origin_group=group,outcome_archive_namespace="prospective-automation",original_prediction_unchanged=True)
                    (legacy_ledger if group=="legacy" else self.store.ledger).attach_outcome(row["prediction_id"],out)
                    self.store.archive.put("outcomes",{"prediction_id":row["prediction_id"],"outcome":out})
                    matured+=1;group_counts[group]+=1
                except Exception:
                    pending+=1;logging.getLogger(__name__).exception("Outcome remains pending")
                    self.store.event("OUTCOME_PENDING",prediction_id=row["prediction_id"],reason="OUTCOME_DATA_UNAVAILABLE")
            summary={"status":"RESOLVED","matured":matured,"pending":pending,"resolved_at":now.isoformat(),"source_failures":failures,"matured_by_origin_group":group_counts}
            self.store.event("OUTCOME_RESOLUTION",**summary)
            return summary
