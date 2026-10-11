"""Pure finance services and bounded data access, independent of Streamlit."""
from collections import defaultdict,deque
from datetime import datetime,timezone
import json
import math
import os
import re
from threading import RLock
from time import monotonic,perf_counter
from typing import Any
import numpy as np
import pandas as pd
from application.config import ROOT,universe
from application.collection.collector import bounded_provider
from application.services.prospective import ServiceError
from application.schemas import StockResponse,Page
from core.data.providers import equity_family
from core.indicators import add_all_indicators
from core.risk_metrics import full_risk_report
from core.factor_models import compute_momentum_signal,compute_lowvol_signal
def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,(float,int)):
        return value if math.isfinite(value) else None
    if isinstance(value,(pd.Series,pd.DataFrame)):return clean(value.to_dict())
    if isinstance(value,(datetime,pd.Timestamp)):
        if value.tzinfo is None:raise ValueError("Naive timestamp")
        return value.astimezone(timezone.utc).isoformat()
    return value
def valid_ticker(ticker):
    ticker=ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9&-]{0,24}(?:\.(?:NS|BO))?",ticker):raise ServiceError("INVALID_TICKER",422)
    try:equity_family(ticker)
    except ValueError:raise ServiceError("INVALID_TICKER",422)
    return ticker
class Telemetry:
    def __init__(self):self.values=defaultdict(lambda:deque(maxlen=2000));self.lock=RLock()
    def record(self,name,seconds):
        with self.lock:
            self.values[name].append(seconds*1000)
            file=os.environ.get("STOCKPRO_PERFORMANCE_LOG")
            if file:
                from pathlib import Path
                target=Path(file);target.parent.mkdir(parents=True,exist_ok=True)
                with target.open("a",encoding="utf-8") as stream:
                    stream.write(json.dumps({"timestamp":datetime.now(timezone.utc).isoformat(),"operation":name,"milliseconds":seconds*1000})+"\n")
    def summary(self):
        with self.lock:return {k:{"n":len(v),"median_ms":float(np.median(v)),"p95_ms":float(np.percentile(v,95))} for k,v in self.values.items() if v}
class DataAccess:
    def __init__(self,provider=bounded_provider,telemetry=None):
        self.provider=provider;self.telemetry=telemetry or Telemetry();self.cache={};self.lock=RLock()
    def snapshot(self,ticker,reference=False):
        key=("yahoo/yfinance",ticker,"5y","1d",reference)
        # Only successful payloads cached; fresh bounded TTL, never stale-on-error.
        with self.lock:
            for expired in [k for k,v in self.cache.items() if monotonic()-v[0]>=30]:self.cache.pop(expired,None)
            entry=self.cache.get(key)
            if entry and monotonic()-entry[0]<30:return entry[1]
            start=perf_counter()
            try:snapshot=self.provider(ticker,reference,10)
            except Exception:raise ServiceError("PROVIDER_FAILURE")
            finally:self.telemetry.record("provider.history",perf_counter()-start)
            if snapshot.frame.empty:raise ServiceError("DATA_UNAVAILABLE")
            if len(self.cache)>=128:self.cache.pop(min(self.cache,key=lambda k:self.cache[k][0]),None)
            self.cache[key]=(monotonic(),snapshot);return snapshot
class AnalyticsService:
    def __init__(self,provider=bounded_provider):
        self.telemetry=Telemetry();self.data=DataAccess(provider,self.telemetry);self.members=universe()[0]["members"]
        self.context=json.loads((ROOT/"config/context_sources.json").read_text(encoding="utf-8"))["sources"]
    def search(self,q="",offset=0,limit=100):
        query=q.casefold()
        rows=[{"ticker":m["ticker"],"company":m["company"],"industry":m["industry"],"isin":m["isin"]} for m in self.members if query in m["ticker"].casefold() or query in m["company"].casefold()]
        return Page(items=rows[offset:offset+limit],total=len(rows),offset=offset,limit=limit)
    def overview(self,ticker):
        ticker=valid_ticker(ticker);snap=self.data.snapshot(ticker);row=snap.frame.iloc[-1]
        if not np.isfinite(float(row.Close)) or row.Close<=0:raise ServiceError("DATA_UNAVAILABLE")
        return StockResponse(ticker=ticker,retrieved_at=snap.metadata.fetched_at,provider=snap.metadata.provider_id,
            data_quality_status="SESSION_BAR_NOT_LIVE_QUOTE",
            values=clean({"current_price":float(row.Close),"currency":"INR" if ticker.endswith((".NS",".BO")) else "USD",
                "price_timestamp":snap.frame.index[-1],"timestamp_kind":"provider_daily_session_label",
                "observation_timestamp":snap.metadata.source_timestamp,"adjustment_policy":snap.metadata.adjustment_policy,
                "industry":next((m["industry"] for m in self.members if m["ticker"]==ticker),None)}))
    def history(self,ticker,offset=0,limit=100):
        ticker=valid_ticker(ticker);snapshot=self.data.snapshot(ticker);rows=[]
        for timestamp,row in snapshot.frame.iterrows():
            rows.append(clean({"timestamp":timestamp,"open":row.Open,"high":row.High,"low":row.Low,"close":row.Close,
                "adjusted_close":row.get("Adj Close"),"volume":row.Volume,"dividends":row.get("Dividends"),"stock_splits":row.get("Stock Splits")}))
        return Page(items=rows[offset:offset+limit],total=len(rows),offset=offset,limit=limit)
    def technicals(self,ticker):
        ticker=valid_ticker(ticker);snap=self.data.snapshot(ticker);frame=add_all_indicators(snap.frame)
        return StockResponse(ticker=ticker,retrieved_at=snap.metadata.fetched_at,provider=snap.metadata.provider_id,
            data_quality_status="SESSION_BAR_NOT_LIVE_QUOTE",values=clean(frame.iloc[-1].to_dict()))
    def markets(self,sectors=False):
        selected={k:v for k,v in self.context.items() if k.startswith("sector_" if sectors else "market_")}
        rows=[]
        for key,symbol in selected.items():
            try:
                snap=self.data.snapshot(symbol,True);frame=snap.frame
                row={"source":key,"symbol":symbol,"price":float(frame.Close.iloc[-1]),"retrieved_at":snap.metadata.fetched_at,
                    "price_timestamp":frame.index[-1],"data_quality_status":"SESSION_BAR_NOT_LIVE_QUOTE","return_convention":"price_index"}
            except ServiceError as error:row={"source":key,"symbol":symbol,"error_code":error.code,"data_quality_status":"UNAVAILABLE"}
            rows.append(clean(row))
        return Page(items=rows,total=len(rows))
    def risk(self,ticker):
        ticker=valid_ticker(ticker);snap=self.data.snapshot(ticker)
        report=full_risk_report(snap.frame["Adj Close"])
        report.pop("drawdown_series",None)
        return StockResponse(ticker=ticker,retrieved_at=snap.metadata.fetched_at,provider=snap.metadata.provider_id,
            data_quality_status="HISTORICAL_ADJUSTED_RETURN",values=clean(report))
    def factors(self,ticker):
        ticker=valid_ticker(ticker);snap=self.data.snapshot(ticker);close=snap.frame["Adj Close"]
        return StockResponse(ticker=ticker,retrieved_at=snap.metadata.fetched_at,provider=snap.metadata.provider_id,
            data_quality_status="HISTORICAL_ADJUSTED_RETURN",values=clean({"momentum_12_1":compute_momentum_signal(close),
                "lowvol":compute_lowvol_signal(close),"fundamental_factors":"UNAVAILABLE_WITHOUT_SEPARATE_PROVIDER"}))
    def screener(self,offset=0,limit=100):
        # Fixed universe metadata screening; no implicit 100-provider query on page open.
        return self.search("",offset,limit)
class PortfolioService:
    """Explicit research calculation boundary; no user data access or trade execution."""
    def risk(self,adjusted_portfolio_value:pd.Series,benchmark:pd.Series|None=None)->dict[str,Any]:
        if adjusted_portfolio_value.empty or (adjusted_portfolio_value<=0).any():raise ServiceError("DATA_UNAVAILABLE")
        report=full_risk_report(adjusted_portfolio_value,benchmark)
        report.pop("drawdown_series",None)
        return clean(report)
