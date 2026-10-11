"""Additive typed web routes; old API contracts stay compatible."""
from datetime import date
from typing import Literal
from fastapi import APIRouter,Depends,Query,Request
from application.services.analytics import valid_ticker
from api.operations_routes import require_operator
from application.web_schemas import *

def web_router():
    router=APIRouter(prefix='/api/v1',tags=['web'])
    @router.get('/stocks/{ticker}/fundamentals',response_model=FundamentalsResponse)
    def fundamentals(ticker:str,request:Request):return request.app.state.web_analytics.fundamentals(ticker)
    @router.get('/stocks/{ticker}/news',response_model=NewsResponse)
    def news(ticker:str,request:Request,limit:int=Query(20,ge=1,le=50)):return request.app.state.web_analytics.news(ticker,limit)
    @router.get('/stocks/{ticker}/history/series',response_model=SessionHistory)
    def history(ticker:str,request:Request,start:date|None=None,end:date|None=None,offset:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=1000),latest:bool=True):
        return request.app.state.web_analytics.history(ticker,start.isoformat() if start else None,end.isoformat() if end else None,offset,limit,latest)
    @router.get('/stocks/{ticker}/technicals/series',response_model=TechnicalSeries)
    def technicals(ticker:str,request:Request,group:Literal['trend','momentum','volatility','volume']='trend',offset:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=1000)):
        return request.app.state.web_analytics.technicals(ticker,group,offset,limit)
    @router.get('/stocks/{ticker}/risk/report',response_model=RiskResponse)
    def risk(ticker:str,request:Request):return request.app.state.web_analytics.risk(ticker)
    @router.get('/stocks/{ticker}/factors/report',response_model=FactorResponse)
    def factors(ticker:str,request:Request):return request.app.state.web_analytics.factors(ticker)
    @router.get('/markets/context',response_model=MarketsResponse)
    def markets(request:Request):return request.app.state.web_analytics.markets()
    @router.get('/validation/historical/v5/{report}',response_model=HistoricalEvidenceResponse)
    def historical(report:Literal['development','final','quarantine'],request:Request):return request.app.state.historical_evidence.read(report)
    @router.get('/research/forecasts',response_model=ResearchHistoryResponse)
    def forecasts(request:Request,group:Literal['canonical','legacy','v5']='canonical',ticker:str|None=None,horizon:Literal[1,5,10,20]|None=None,offset:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=1000)):
        return request.app.state.archived_forecasts.history(group,valid_ticker(ticker) if ticker else None,offset,limit,horizon)
    @router.get('/research/forecasts/{prediction_id}',response_model=ArchivedDetail)
    def detail(prediction_id:str,request:Request,group:Literal['canonical','legacy','v5']='canonical'):return request.app.state.archived_forecasts.detail(prediction_id,group)
    @router.get('/research/forecasts/{prediction_id}/analogues',response_model=AnaloguesResponse)
    def analogues(prediction_id:str,request:Request,group:Literal['canonical','legacy']='canonical',count:int=Query(20,ge=1,le=50)):return request.app.state.archived_forecasts.analogues(prediction_id,group,count)
    @router.post('/research/forecasts/{prediction_id}/scenario',response_model=ScenarioResponse,dependencies=[Depends(require_operator)])
    def scenario(prediction_id:str,body:ScenarioRequest,request:Request):return request.app.state.archived_forecasts.scenario(prediction_id,body)
    @router.post('/analysis/portfolio',response_model=PortfolioResponse,dependencies=[Depends(require_operator)])
    def portfolio(body:PortfolioRequest,request:Request):return request.app.state.web_analytics.portfolio(body)
    @router.post('/analysis/watchlist',response_model=WatchlistResponse,dependencies=[Depends(require_operator)])
    def watchlist(body:WatchlistRequest,request:Request):return request.app.state.web_analytics.watchlist(body)
    @router.post('/analysis/compare',response_model=CompareResponse,dependencies=[Depends(require_operator)])
    def compare(body:CompareRequest,request:Request):return request.app.state.web_analytics.compare(body)
    @router.post('/analysis/screen',response_model=ScreenResponse,dependencies=[Depends(require_operator)])
    def screen(body:ScreenRequest,request:Request):return request.app.state.web_analytics.screen(body)
    return router
