"""Current-vintage analytics; all finance is Python, bounded acquisition is separate."""
from datetime import date,datetime,timezone
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from threading import BoundedSemaphore,RLock
from time import monotonic
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from application.config import ROOT
from application.services.analytics import AnalyticsService,valid_ticker,clean
from application.services.prospective import ServiceError
from application.web_schemas import *
from core.indicators import add_all_indicators
from core.risk_metrics import full_risk_report
from core.factor_models import compute_factor_exposures,compute_momentum_signal,compute_lowvol_signal,compute_value_signal,compute_quality_signal

_provider_slots=BoundedSemaphore(2)

def metadata_provider(kind,ticker,limit):
    """Killable provider process contains old unbounded convenience functions.

    The allowlisted module worker uses the existing functions without importing
    Streamlit pages. Timeout and output bounds apply across every fallback.
    """
    if kind not in {'fundamentals','news','factors'}:raise ValueError('Unsupported provider')
    if not _provider_slots.acquire(blocking=False):raise ServiceError('PROVIDER_BUSY',503)
    try:
        with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
            result=subprocess.run([sys.executable,'-m','application.services.web_analytics',kind,ticker,str(limit)],cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,timeout=20,shell=False,creationflags=0x08000000 if os.name=="nt" else 0)
            if result.returncode or out.tell()>2_000_000:raise ServiceError('PROVIDER_FAILURE')
            out.seek(0);payload=json.loads(out.read().decode('utf-8'))
        if kind=='factors':
            frame=pd.DataFrame(payload['data'],columns=payload['columns'],index=pd.to_datetime(payload['index']))
            return frame
        return payload
    except (subprocess.TimeoutExpired,ValueError,OSError):raise ServiceError('PROVIDER_FAILURE')
    finally:_provider_slots.release()

def currency(ticker):return 'INR' if ticker.endswith(('.NS','.BO')) else 'USD'

def number(value):
    try:return float(value) if math.isfinite(float(value)) else None
    except (ValueError,TypeError):return None

def safe_url(value):
    if not isinstance(value,str) or len(value)>2000:return None
    parsed=urlparse(value)
    return value if parsed.scheme in {'http','https'} and parsed.hostname and not parsed.username else None

def session_prices(snapshot,column='Adj Close'):
    frame=snapshot.frame
    series=frame[column].copy();series.index=pd.Index(frame.index.tz_convert(snapshot.metadata.exchange_timezone).date.astype(str))
    if not series.index.is_unique or (series<=0).any() or not np.isfinite(series).all():raise ServiceError('DATA_QUALITY_FAILURE')
    return series

def report(prices,benchmark=None):
    if len(prices)<10:raise ServiceError('INSUFFICIENT_HISTORY')
    result=full_risk_report(prices,benchmark)
    result.pop('drawdown_series',None)
    return clean(result)

class WebAnalytics:
    groups={'trend':('SMA','EMA','MACD','Signal','Hist','ADX','DI','SAR'),
            'momentum':('RSI','Stoch','Williams','CCI','ROC','MOM'),
            'volatility':('BB','ATR','HV','KC','DC'),
            'volume':('Volume','OBV','MFI','CMF','VWAP','VolRatio')}
    def __init__(self,analytics=None,metadata_provider=metadata_provider):
        self.analytics=analytics or AnalyticsService();self.data=self.analytics.data
        self.metadata_provider=metadata_provider;self.cache={};self.lock=RLock()
    def _metadata(self,kind,ticker,limit=20):
        key=(kind,ticker,limit)
        with self.lock:
            entry=self.cache.get(key)
            if entry and monotonic()-entry[0]<30:return entry[1]
        value=self.metadata_provider(kind,ticker,limit)
        with self.lock:
            if len(self.cache)>=128:self.cache.pop(min(self.cache,key=lambda k:self.cache[k][0]))
            self.cache[key]=(monotonic(),value)
        return value
    def history(self,ticker,start=None,end=None,offset=0,limit=100,latest=False):
        ticker=valid_ticker(ticker)
        if not 1<=limit<=1000 or offset<0:raise ServiceError('INVALID_REQUEST',422)
        if start and end and start>end:raise ServiceError('INVALID_REQUEST',422)
        for value in (start,end):
            if value:
                try:date.fromisoformat(value)
                except ValueError:raise ServiceError('INVALID_REQUEST',422)
        snapshot=self.data.snapshot(ticker);zone=snapshot.metadata.exchange_timezone
        rows=[]
        for timestamp,row in snapshot.frame.iterrows():
            session=timestamp.tz_convert(zone).date().isoformat()
            if (start and session<start) or (end and session>end):continue
            rows.append(SessionBar(exchange_session=session,timestamp=timestamp,open=number(row.Open),high=number(row.High),low=number(row.Low),close=number(row.Close),adjusted_close=number(row.get('Adj Close')),volume=number(row.Volume),dividends=number(row.get('Dividends')),stock_splits=number(row.get('Stock Splits'))))
        total=len(rows)
        selected=rows[max(0,total-offset-limit):max(0,total-offset)] if latest else rows[offset:offset+limit]
        return SessionHistory(items=selected,total=total,offset=offset,limit=limit,exchange_timezone=zone,currency=currency(ticker))
    def fundamentals(self,ticker):
        ticker=valid_ticker(ticker);now=datetime.now(timezone.utc)
        try:
            values=self._metadata('fundamentals',ticker)
            if not isinstance(values,dict) or not values:raise ValueError('Empty fundamentals')
            metrics={key:number(values.get(key)) for key in FundamentalMetrics.model_fields}
            raw_yield=number(values.get('dividend_yield'))
            yield_unit=values.get('dividend_yield_unit')
            yield_unit=yield_unit if yield_unit in {'fraction','percent'} else 'unverified'
            metrics['dividend_yield_provider_value']=raw_yield
            metrics['dividend_yield']=raw_yield/100 if raw_yield is not None and yield_unit=='percent' else raw_yield if yield_unit=='fraction' else None
            company={key:values.get(key) for key in Company.model_fields}
            for key in ('name','sector','industry','description','exchange'):
                value=company[key];company[key]=str(value)[:6000] if value and value not in {'—','â€”'} else None
            company['employees']=number(company['employees']);company['website']=safe_url(company['website'])
            status='AVAILABLE' if all(v is not None for v in metrics.values()) else 'PARTIAL' if any(v is not None for v in metrics.values()) else 'UNAVAILABLE'
            response=FundamentalsResponse(ticker=ticker,currency=str(values.get('currency') or currency(ticker))[:10],retrieved_at=now,status=status,company=Company(**company),metrics=FundamentalMetrics(**metrics),error_code='FUNDAMENTALS_UNAVAILABLE' if status=='UNAVAILABLE' else None,dividend_yield_source_unit=yield_unit)
            response.units['dividend_yield_provider_value']=yield_unit if yield_unit!='unverified' else 'provider_unit_unverified'
            return response
        except Exception:return FundamentalsResponse(ticker=ticker,currency=currency(ticker),retrieved_at=now,status='UNAVAILABLE',company=Company(),metrics=FundamentalMetrics(),error_code='FUNDAMENTALS_UNAVAILABLE')
    def news(self,ticker,limit=20):
        ticker=valid_ticker(ticker)
        if not 1<=limit<=50:raise ServiceError('INVALID_REQUEST',422)
        now=datetime.now(timezone.utc);items=[]
        try:
            raw=self._metadata('news',ticker,limit)
            from core.sentiment import analyze_text_sentiment,classify_sentiment,_VADER_AVAILABLE
            for original in raw[:limit]:
                item=original.get('content',original)
                title=item.get('title')
                if not isinstance(title,str) or not title.strip():continue
                timestamp=item.get('pubDate',item.get('providerPublishTime'))
                try:
                    published=pd.Timestamp(timestamp,unit='s',tz='UTC') if isinstance(timestamp,(int,float)) else pd.Timestamp(timestamp)
                    published=published.tz_convert('UTC') if published.tzinfo else published.tz_localize('UTC')
                    if pd.isna(published):published=None
                except (ValueError,TypeError):published=None
                publisher=item.get('publisher') or item.get('provider',{}).get('displayName')
                link=item.get('link') or item.get('canonicalUrl',{}).get('url') or item.get('clickThroughUrl',{}).get('url')
                score=analyze_text_sentiment(title)['compound'] if _VADER_AVAILABLE else None
                items.append(NewsItem(title=title[:2000],publisher=str(publisher)[:300] if publisher else None,url=safe_url(link),published_at=published,sentiment_score=score,sentiment_label=classify_sentiment(score) if score is not None else None))
            scores=[i.sentiment_score for i in items if i.sentiment_score is not None]
            return NewsResponse(ticker=ticker,retrieved_at=now,status='AVAILABLE' if items else 'UNAVAILABLE',items=items,overall_score=float(np.mean(scores)) if scores else None,error_code=None if items else 'NEWS_UNAVAILABLE')
        except Exception:return NewsResponse(ticker=ticker,retrieved_at=now,status='UNAVAILABLE',items=[],error_code='NEWS_UNAVAILABLE')
    def technicals(self,ticker,group='trend',offset=0,limit=100):
        ticker=valid_ticker(ticker)
        if group not in self.groups or not 1<=limit<=1000 or offset<0:raise ServiceError('INVALID_REQUEST',422)
        snapshot=self.data.snapshot(ticker);frame=add_all_indicators(snapshot.frame)
        columns=[key for key in frame if any(key.startswith(prefix) for prefix in self.groups[group])]
        points=[TechnicalPoint(exchange_session=timestamp.tz_convert(snapshot.metadata.exchange_timezone).date().isoformat(),timestamp=timestamp,values={key:number(row[key]) for key in columns}) for timestamp,row in frame.iterrows()]
        # latest window offset, ascending within window
        total=len(points)
        return TechnicalSeries(ticker=ticker,group=group,items=points[max(0,total-offset-limit):max(0,total-offset)],total=total,offset=offset,limit=limit,exchange_timezone=snapshot.metadata.exchange_timezone)
    def markets(self):
        items=[]
        for source,symbol in self.analytics.context.items():
            category='sector' if source.startswith('sector_') else 'market' if source.startswith('market_') or source in {'nasdaq','nikkei','hang_seng'} else 'volatility' if source in {'vix','india_vix'} else 'fx' if source in {'usd_inr','dxy'} else 'commodity' if source in {'brent','wti','gold'} else 'rates'
            try:
                snapshot=self.data.snapshot(symbol,True);frame=snapshot.frame
                current=number(frame.Close.iloc[-1]);previous=number(frame.Close.iloc[-2]) if len(frame)>1 else None
                if current is None:raise ServiceError('DATA_UNAVAILABLE')
                items.append(MarketItem(source=source,symbol=symbol,category=category,status='AVAILABLE',price=current,previous_close=previous,change=current-previous if previous is not None else None,change_percent=(current/previous-1)*100 if previous else None,exchange_session=frame.index[-1].tz_convert(snapshot.metadata.exchange_timezone).date().isoformat(),timestamp=frame.index[-1],retrieved_at=snapshot.metadata.fetched_at))
            except ServiceError as error:items.append(MarketItem(source=source,symbol=symbol,category=category,status='UNAVAILABLE',error_code=error.code))
        return MarketsResponse(items=items,total=len(items))
    def risk(self,ticker):
        ticker=valid_ticker(ticker);snapshot=self.data.snapshot(ticker);prices=session_prices(snapshot)
        benchmark=self.analytics.context['market_india' if currency(ticker)=='INR' else 'market_us']
        benchmark_status='AVAILABLE';aligned=0
        metrics=report(prices)
        try:
            bench=session_prices(self.data.snapshot(benchmark,True));matched=pd.concat([prices.rename('stock'),bench.rename('benchmark')],axis=1,join='inner').dropna()
            aligned=len(matched)-1
            if aligned<10 or matched.benchmark.pct_change().dropna().std()==0:benchmark_status='INSUFFICIENT_ALIGNED_HISTORY'
            else:metrics.update(report(matched.stock,matched.benchmark))
        except ServiceError:benchmark_status='UNAVAILABLE'
        for key in ('beta','alpha','r_squared','treynor','information_ratio'):metrics.setdefault(key,None)
        drawdown=prices/prices.cummax()-1
        zone=snapshot.metadata.exchange_timezone
        points=[TechnicalPoint(exchange_session=session,timestamp=pd.Timestamp(session,tz=zone),values={'drawdown':float(value)}) for session,value in drawdown.items()]
        return RiskResponse(ticker=ticker,benchmark=benchmark,aligned_observations=max(0,aligned),metrics=metrics,drawdown=points,benchmark_status=benchmark_status)
    def factors(self,ticker):
        ticker=valid_ticker(ticker);snapshot=self.data.snapshot(ticker);prices=session_prices(snapshot);fundamental=self.fundamentals(ticker).metrics
        signals=clean({'momentum_12_1':compute_momentum_signal(prices),'lowvol':compute_lowvol_signal(prices),'value':compute_value_signal(fundamental.pe_ttm),'quality':compute_quality_signal(fundamental.roe,fundamental.gross_margin,fundamental.operating_margin)})
        try:
            factors=self._metadata('factors',ticker).copy(deep=True)
            factors.index=pd.DatetimeIndex(factors.index).tz_localize(None).to_period('M')
            if not factors.index.is_unique or not {'Mkt-RF','SMB','HML','RF'}.issubset(factors.columns):raise ValueError('Factor schema')
            monthly=prices.copy();monthly.index=pd.to_datetime(monthly.index).to_period('M')
            monthly=monthly.groupby(level=0).last().pct_change(fill_method=None)
            # Exclude ongoing month and incomplete last source month even if the
            # provider's factor vintage extends further than the price history.
            today=datetime.now(timezone.utc).astimezone(__import__('zoneinfo').ZoneInfo(snapshot.metadata.exchange_timezone))
            current=pd.Period(today.strftime('%Y-%m'),freq='M')
            monthly=monthly[monthly.index<current]
            last_source_date=pd.Timestamp(prices.index[-1])
            if last_source_date.date()<(last_source_date+pd.offsets.BMonthEnd(0)).date():
                monthly=monthly[monthly.index<last_source_date.to_period('M')]
            aligned=pd.concat([monthly.rename('stock_ret'),factors],axis=1,join='inner').dropna()
            if len(aligned)<12:raise ValueError('Short factors')
            exposure=compute_factor_exposures(aligned.stock_ret,aligned.drop(columns='stock_ret'))
            if 'error' in exposure:raise ValueError('Unavailable regression')
            return FactorResponse(ticker=ticker,status='AVAILABLE',signals=signals,exposures=clean(exposure),aligned_months=aligned.index.astype(str).tolist(),factor_series=[FactorMonth(month=str(month),stock_return=float(row.stock_ret),factors=clean(row.drop('stock_ret').to_dict())) for month,row in aligned.iterrows()])
        except Exception:return FactorResponse(ticker=ticker,status='UNAVAILABLE',signals=signals,error_code='FACTOR_EXPOSURES_UNAVAILABLE')
    def portfolio(self,request):
        holdings={valid_ticker(h.ticker):h.weight for h in request.holdings}
        if len({currency(t) for t in holdings})!=1:raise ServiceError('MIXED_CURRENCY_UNSUPPORTED',422)
        values={};failed={}
        for ticker,weight in holdings.items():
            if weight==0:continue
            try:values[ticker]=session_prices(self.data.snapshot(ticker))
            except ServiceError as error:failed[ticker]=error.code
        if not values:raise ServiceError('DATA_UNAVAILABLE')
        frame=pd.concat(values,axis=1,join='inner').dropna()
        if len(frame)<10:raise ServiceError('INSUFFICIENT_ALIGNED_HISTORY')
        total=sum(holdings[t] for t in values);weights={t:holdings[t]/total for t in values}
        returns=frame.pct_change(fill_method=None).iloc[1:].mul(pd.Series(weights)).sum(axis=1)
        value=pd.concat([pd.Series([request.initial_value],index=frame.index[:1]),request.initial_value*(1+returns).cumprod()])
        return PortfolioResponse(currency=currency(next(iter(values))),weights=weights,series=[ValuePoint(exchange_session=session,value=float(v)) for session,v in value.items()],metrics=report(value),failed_holdings=failed)
    def compare(self,request):
        tickers=[valid_ticker(t) for t in request.tickers]
        frame=pd.concat({t:session_prices(self.data.snapshot(t)) for t in tickers},axis=1,join='inner').dropna()
        if len(frame)<10:raise ServiceError('INSUFFICIENT_ALIGNED_HISTORY')
        normalized=frame/frame.iloc[0]*100;returns=frame.pct_change(fill_method=None).dropna()
        series=[ComparePoint(exchange_session=session,values=clean(row.to_dict())) for session,row in normalized.iterrows()]
        spreads=(normalized[tickers[1:]].subtract(normalized[tickers[0]],axis=0))/100
        rolling={f'{tickers[0]}:{t}':returns[tickers[0]].rolling(30).corr(returns[t]) for t in tickers[1:]}
        rolling_frame=pd.DataFrame(rolling)
        return CompareResponse(series=series,cumulative_return_spread=[ComparePoint(exchange_session=s,values=clean(r.to_dict())) for s,r in spreads.iterrows()],correlation=clean(returns.corr().to_dict()),rolling_correlation=[ComparePoint(exchange_session=s,values=clean(r.to_dict())) for s,r in rolling_frame.iterrows()],metrics={t:report(frame[t]) for t in tickers},currencies={t:currency(t) for t in tickers})
    def watchlist(self,request):
        items=[];totals={}
        for position in request.positions:
            ticker=valid_ticker(position.ticker);curr=currency(ticker)
            try:
                snap=self.data.snapshot(ticker);prices=session_prices(snap,'Close');last=float(prices.iloc[-1]);value=last*position.quantity;cost=position.average_buy*position.quantity
                item=WatchItem(ticker=ticker,currency=curr,status='AVAILABLE',last_price=last,exchange_session=prices.index[-1],value=value,cost=cost,pnl=value-cost,pnl_percent=(value/cost-1)*100,change_percent=(last/float(prices.iloc[-2])-1)*100 if len(prices)>1 else None,target_reached=last>=position.target if position.target else None,stop_reached=last<=position.stop if position.stop else None)
                total=totals.setdefault(curr,{'value':0.,'cost':0.,'pnl':0.});total['value']+=value;total['cost']+=cost;total['pnl']+=value-cost
            except ServiceError as error:item=WatchItem(ticker=ticker,currency=curr,status='UNAVAILABLE',error_code=error.code)
            items.append(item)
        return WatchlistResponse(items=items,totals=totals)
    def screen(self,request):
        items=[]
        for raw in request.tickers:
            ticker=valid_ticker(raw);reasons=[];values={}
            try:
                frame=add_all_indicators(self.data.snapshot(ticker).frame);last=frame.iloc[-1]
                rsi_value=number(last.get('RSI',last.get('RSI_14')));ema=number(last.get('EMA_50'))
                values.update(price=number(last.Close),rsi=rsi_value,ema50=ema)
                if rsi_value is None:reasons.append('RSI_UNAVAILABLE')
                elif not request.rsi_min<=rsi_value<=request.rsi_max:reasons.append('RSI_FILTER')
                if request.trend!='any':
                    if ema is None:reasons.append('EMA_UNAVAILABLE')
                    elif (request.trend=='above_ema50' and last.Close<=ema) or (request.trend=='below_ema50' and last.Close>=ema):reasons.append('TREND_FILTER')
                if any(value is not None for value in (request.pe_min,request.pe_max,request.beta_max,request.dividend_min)):
                    company=self.fundamentals(ticker);metrics=company.metrics
                    values.update(pe_ttm=metrics.pe_ttm,beta=metrics.beta,dividend_yield=metrics.dividend_yield)
                    if company.status=='UNAVAILABLE':reasons.append('FUNDAMENTALS_UNAVAILABLE')
                    if request.dividend_min is not None and company.dividend_yield_source_unit=='unverified':reasons.append('DIVIDEND_UNIT_UNVERIFIED')
                    for active,value,label,condition in [(request.pe_min is not None or request.pe_max is not None,metrics.pe_ttm,'PE',(request.pe_min is None or metrics.pe_ttm is not None and metrics.pe_ttm>=request.pe_min) and (request.pe_max is None or metrics.pe_ttm is not None and metrics.pe_ttm<=request.pe_max)),(request.beta_max is not None,metrics.beta,'BETA',request.beta_max is None or metrics.beta is not None and metrics.beta<=request.beta_max),(request.dividend_min is not None,metrics.dividend_yield,'DIVIDEND',request.dividend_min is None or metrics.dividend_yield is not None and metrics.dividend_yield*100>=request.dividend_min)]:
                        if active and value is None:reasons.append(label+'_UNAVAILABLE')
                        elif active and not condition:reasons.append(label+'_FILTER')
                items.append(ScreenItem(ticker=ticker,status='EXCLUDED' if reasons else 'MATCH',reasons=reasons,values=values))
            except ServiceError as error:items.append(ScreenItem(ticker=ticker,status='UNAVAILABLE',reasons=[error.code],values={}))
        return ScreenResponse(items=items)

def _provider_main():
    kind,ticker,limit=sys.argv[1:];limit=int(limit)
    if kind=='factors':
        from core.data.factors import fetch_fama_french_factors
        frame=fetch_fama_french_factors(start='2019-01-01')
        payload={'columns':frame.columns.tolist(),'index':frame.index.astype(str).tolist(),'data':clean(frame.values.tolist())}
    elif kind in {'fundamentals','news'}:
        from core.data.legacy_market import fetch_fundamentals,fetch_news
        payload=fetch_fundamentals(valid_ticker(ticker)) if kind=='fundamentals' else fetch_news(valid_ticker(ticker),max_items=min(limit,50))
    else:raise ValueError('Unsupported provider')
    print(json.dumps(clean(payload),allow_nan=False))

if __name__=='__main__':_provider_main()
