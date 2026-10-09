"""Versioned read API. Collection/training are explicit jobs, never request handlers."""
from datetime import datetime,timezone
import json
import logging
import os
import secrets
from time import perf_counter
import uuid
from fastapi import FastAPI,Query,Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from application.config import MODEL,PROTOCOL,ROOT
from application.schemas import HealthResponse,ModelResponse,ForecastResponse,ValidationResponse,CollectionResponse,StockResponse,Page,ErrorResponse,HistoryResponse,SearchResponse,ForecastHistoryResponse
from application.services.analytics import AnalyticsService,valid_ticker
from application.services.prospective import ProspectiveService,ServiceError
from core.forecasting.service import active_research
def create_app(prospective=None,analytics=None):
    prospective=prospective or ProspectiveService();analytics=analytics or AnalyticsService()
    app=FastAPI(title="StockPro Research Backend",version="1.0.0",
        description="Research only; no training, promotion or trade endpoints.",
        responses={422:{"model":ErrorResponse},503:{"model":ErrorResponse}})
    app.state.prospective=prospective;app.state.analytics=analytics
    origins=[x.strip() for x in os.environ.get("STOCKPRO_CORS_ORIGINS","").split(",") if x.strip()]
    if origins:app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=["GET"],allow_headers=["Authorization","Content-Type"])
    def failure(request,code,status):
        messages={"INVALID_TICKER":"Unsupported equity ticker.","PROVIDER_FAILURE":"Market provider is unavailable.",
            "DATA_UNAVAILABLE":"Required data are unavailable.","STALE_DATA":"Data freshness requirement failed.",
            "MODEL_UNAVAILABLE":"No eligible saved forecast is available.","SCHEMA_MISMATCH":"Stored data failed integrity checks.",
            "INVALID_REQUEST":"Request parameters are invalid.","UNAUTHORIZED":"Authentication required.","INTERNAL_ERROR":"Service request failed."}
        return JSONResponse(status_code=status,content={"error":{"code":code,"message":messages.get(code,"Requested research data are unavailable."),
            "request_id":getattr(request.state,"request_id",str(uuid.uuid4()))}})
    @app.middleware("http")
    async def instrumentation(request:Request,call_next):
        request.state.request_id=str(uuid.uuid4());start=perf_counter()
        token=os.environ.get("STOCKPRO_API_TOKEN")
        if token and request.url.path!="/api/v1/health":
            header=request.headers.get("authorization","")
            supplied=header[7:] if header.startswith("Bearer ") else ""
            if not secrets.compare_digest(supplied,token):return failure(request,"UNAUTHORIZED",401)
        try:response=await call_next(request)
        except Exception:
            logging.getLogger(__name__).exception("API failure request=%s",request.state.request_id)
            response=failure(request,"INTERNAL_ERROR",500)
        route=request.scope.get("route")
        analytics.telemetry.record("api."+getattr(route,"path","UNMATCHED"),perf_counter()-start)
        response.headers["X-Request-ID"]=request.state.request_id
        return response
    @app.exception_handler(ServiceError)
    async def service_error(request,error):return failure(request,error.code,error.status)
    @app.exception_handler(RequestValidationError)
    async def validation_error(request,error):return failure(request,"INVALID_REQUEST",422)
    @app.exception_handler(HTTPException)
    async def http_error(request,error):return failure(request,"NOT_FOUND" if error.status_code==404 else "INVALID_REQUEST",error.status_code)
    @app.get("/api/v1/health",response_model=HealthResponse)
    def health():return HealthResponse(status="OK",timestamp=datetime.now(timezone.utc))
    def model():
        try:
            pointer=active_research()
            state="FROZEN" if pointer["model_version"]==MODEL else "MODEL_UNAVAILABLE"
        except Exception:state="MODEL_UNAVAILABLE"
        return ModelResponse(model_version=MODEL,protocol_version=PROTOCOL,health=state,
            calibration_state="FROZEN_V5_ACCEPTANCE_UNCHANGED",training_lineage="HISTORICAL_AVAILABILITY_UNVERIFIED",
            details={"confidence":"LOW","abstain":True,"final_gate":"FAILED","forecast_mode":"SAVED_RESEARCH_FORECASTS"})
    @app.get("/api/v1/model/status",response_model=ModelResponse)
    def model_status():return model()
    @app.get("/api/v1/model/health",response_model=ModelResponse)
    def model_health():return model()
    @app.get("/api/v1/system/status",response_model=ModelResponse)
    def system_status():return model()
    @app.get("/api/v1/validation/summary",response_model=ValidationResponse)
    def validation_summary():return prospective.validation()
    @app.get("/api/v1/validation/prospective",response_model=ValidationResponse)
    def validation_prospective(group:str=Query("canonical",pattern="^(canonical|legacy)$")):return prospective.validation(group)
    @app.get("/api/v1/stocks/search",response_model=SearchResponse)
    def search(q:str=Query("",max_length=100),offset:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=1000)):return analytics.search(q,offset,limit)
    @app.get("/api/v1/stocks/{ticker}/overview",response_model=StockResponse)
    def overview(ticker:str):return analytics.overview(ticker)
    @app.get("/api/v1/stocks/{ticker}/history",response_model=HistoryResponse)
    def history(ticker:str,offset:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=1000)):return analytics.history(ticker,offset,limit)
    @app.get("/api/v1/stocks/{ticker}/technicals",response_model=StockResponse)
    def technicals(ticker:str):return analytics.technicals(ticker)
    @app.get("/api/v1/stocks/{ticker}/forecast",response_model=ForecastResponse)
    def forecast(ticker:str,group:str=Query("canonical",pattern="^(canonical|legacy)$")):
        return prospective.forecast(valid_ticker(ticker),group)
    @app.get("/api/v1/stocks/{ticker}/forecast/history",response_model=ForecastHistoryResponse)
    def forecast_history(ticker:str,group:str=Query("canonical",pattern="^(canonical|legacy)$"),offset:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=1000)):
        ticker=valid_ticker(ticker);rows=[r for r in prospective.rows(group) if r["ticker"]==ticker]
        keys=("prediction_id","ticker","forecast_as_of","information_cutoff","origin_session","horizon","current_price","central_log_return","central_price","quantiles",
            "p_positive_calibrated","raw_p_positive","calibration_state","regime","ood_score","model_version","protocol_version","training_cutoff","feature_schema_hash",
            "confidence","abstain","trust_reasons","recommendation_allowed","evaluation_sealed")
        items=[{k:r.get(k,False if k=="recommendation_allowed" else None) for k in keys} for r in rows]
        for item,row in zip(items,rows):
            item.update(research_state="RESEARCH_ONLY",production_promotion="BLOCKED",recommendation_allowed=False,origin_group=group)
            item.update(protocol_version=row.get("protocol_version","legacy-manual-v1"),provenance_references=row["snapshot_ids"],
                price_timestamp=row.get("price_timestamp") or __import__("pandas").Timestamp(row["origin_session"],tz="Asia/Kolkata" if row["market"]=="india_equity" else "America/New_York").tz_convert("UTC").isoformat())
            outcome=row.get("outcome")
            outcome_keys=("status","actual_return","actual_log_return","actual_future_price","absolute_error","squared_error","direction_correct","interval_80_hit","interval_50_hit",
                "benchmark_return","sector_return","alpha_market","alpha_sector","maturity_timestamp","resolved_at","return_convention","timestamp_correction")
            item["outcome"]={k:outcome.get(k) for k in outcome_keys} if outcome else None
            item["outcome_state"]="SEALED" if row["evaluation_sealed"] else "OUTCOME_PENDING" if outcome is None else outcome["status"]
        return Page(items=items[offset:offset+limit],total=len(items),offset=offset,limit=limit)
    @app.get("/api/v1/stocks/{ticker}/forecast/explanation",response_model=Page)
    def explanation(ticker:str,group:str=Query("canonical",pattern="^(canonical|legacy)$")):
        latest=prospective.forecast(valid_ticker(ticker),group)
        selected={h.prediction_id for h in latest.horizons}
        items=[{"prediction_id":r["prediction_id"],"horizon":r["horizon"],"drivers":r.get("drivers"),
            "abstention_reasons":r.get("trust_reasons"),"recommendation_allowed":False} for r in prospective.rows(group) if r["prediction_id"] in selected]
        return Page(items=items,total=len(items))
    @app.get("/api/v1/markets/overview",response_model=Page)
    def markets():return analytics.markets()
    @app.get("/api/v1/markets/sectors",response_model=Page)
    def sectors():return analytics.markets(True)
    @app.get("/api/v1/data/health",response_model=CollectionResponse)
    def data_health():return prospective.collection_status()
    @app.get("/api/v1/collection/status",response_model=CollectionResponse)
    def collection_status():return prospective.collection_status()
    @app.get("/api/v1/stocks/{ticker}/risk",response_model=StockResponse)
    def risk(ticker:str):return analytics.risk(ticker)
    @app.get("/api/v1/stocks/{ticker}/factors",response_model=StockResponse)
    def factors(ticker:str):return analytics.factors(ticker)
    @app.get("/api/v1/screener",response_model=SearchResponse)
    def screener(offset:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=1000)):return analytics.screener(offset,limit)
    @app.get("/api/v1/system/performance",response_model=Page)
    def performance():
        items=[{"operation":k,**v} for k,v in analytics.telemetry.summary().items()]
        return Page(items=items,total=len(items))
    return app
app=create_app()
