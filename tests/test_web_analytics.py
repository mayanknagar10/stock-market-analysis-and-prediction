"""Finance services tested with frozen local provider fixtures."""
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pytest
from application.services.prospective import ServiceError

def service():
    from application.services.analytics import AnalyticsService
    from application.services.web_analytics import WebAnalytics
    from test_automation_outcomes import snapshot
    dates=pd.bdate_range('2024-01-01',periods=650).date.astype(str).tolist()
    base=100*np.exp(np.cumsum(.0003+.01*np.sin(np.arange(650))))
    def provider(symbol,reference,timeout): return snapshot(symbol,dates,base**(2 if symbol=='TCS.NS' else 1), '2026-10-09T12:00:00Z',reference=reference)
    def metadata(kind,ticker,limit):
        if kind=='fundamentals':return {'name':'Test company','currency':'INR','pe_ttm':20,'roe':.2,'description':'Test','private_key':'hidden'}
        if kind=='news':return [{'content':{'title':'Company beats earnings','pubDate':'2026-10-08T12:00:00Z','provider':{'displayName':'Publisher'},'canonicalUrl':{'url':'https://example.com/story'}}}]
        if kind=='factors':
            rng=np.random.default_rng(3)
            return pd.DataFrame(rng.normal(0,.02,(34,5)),index=pd.date_range('2024-01-01',periods=34,freq='MS'),columns=['Mkt-RF','SMB','HML','RMW','CMA']).assign(RF=.002)
    return WebAnalytics(AnalyticsService(provider),metadata_provider=metadata)

def test_history_latest_ranges_and_exchange_session_are_not_utc_dates():
    svc=service()
    result=svc.history('TCS.NS',start='2026-01-01',limit=3,latest=True)
    assert len(result.items)==3 and result.items[-1].exchange_session>='2026-01-01'
    assert result.items[-1].exchange_session==pd.Timestamp(result.items[-1].timestamp).tz_convert('Asia/Kolkata').date().isoformat()
    assert result.total>3
    with pytest.raises(ServiceError):svc.history('TCS.NS',start='2026-01-02',end='2026-01-01')

def test_company_and_nested_news_normalized_without_private_fields():
    svc=service()
    company=svc.fundamentals('TCS.NS')
    assert company.company.name=='Test company' and company.metrics.pe_ttm==20
    assert 'private_key' not in company.model_dump_json() and company.vintage=='CURRENT_REVISED'
    feed=svc.news('TCS.NS')
    assert feed.items[0].title=='Company beats earnings' and feed.items[0].published_at.tzinfo
    assert feed.items[0].url=='https://example.com/story'

def test_technical_series_group_and_risk_measured_benchmark():
    svc=service()
    series=svc.technicals('TCS.NS',group='momentum',limit=10)
    assert len(series.items)==10 and any('RSI' in k for k in series.items[-1].values)
    risk=svc.risk('TCS.NS')
    assert risk.benchmark=='^NSEI' and risk.aligned_observations>10
    assert 1.9<risk.metrics['beta']<2.1 and len(risk.drawdown)>10
    assert len(svc.markets().items)==19 and svc.markets().items[0].change_percent is not None

def test_factor_month_alignment_and_completed_months():
    svc=service();result=svc.factors('TCS.NS')
    assert result.status=='AVAILABLE' and result.exposures.n_observations>=12
    assert all(month<'2026-10' for month in result.aligned_months)
    assert result.limitations

def test_portfolio_compare_watchlist_currency_and_alignment():
    from application.web_schemas import PortfolioRequest, CompareRequest, WatchlistRequest
    svc=service()
    result=svc.portfolio(PortfolioRequest(holdings=[{'ticker':'TCS.NS','weight':2},{'ticker':'RELIANCE.NS','weight':1}]))
    assert result.currency=='INR' and result.weights['TCS.NS']==pytest.approx(2/3)
    assert result.series[0].value==100 and result.series[-1].value>0
    with pytest.raises(ServiceError):svc.portfolio(PortfolioRequest(holdings=[{'ticker':'TCS.NS','weight':1},{'ticker':'AAPL','weight':1}]))
    comparison=svc.compare(CompareRequest(tickers=['TCS.NS','RELIANCE.NS']))
    assert comparison.series[0].values=={'TCS.NS':100,'RELIANCE.NS':100}
    watch=svc.watchlist(WatchlistRequest(positions=[{'ticker':'TCS.NS','quantity':2,'average_buy':100,'target':110},{'ticker':'AAPL','quantity':1,'average_buy':100}]))
    assert set(watch.totals)=={'INR','USD'} and watch.items[0].target_reached

def test_provider_errors_are_unavailable_not_neutral_and_screen_retains_failures():
    from application.web_schemas import ScreenRequest
    svc=service()
    def fail(*args):raise RuntimeError('private C:/secret')
    svc.metadata_provider=fail
    assert svc.fundamentals('TCS.NS').status=='UNAVAILABLE'
    assert svc.news('TCS.NS').overall_score is None
    result=svc.screen(ScreenRequest(tickers=['TCS.NS'],pe_max=10))
    assert result.items[0].status=='EXCLUDED' and 'FUNDAMENTALS_UNAVAILABLE' in result.items[0].reasons

def test_complete_indicator_groups_and_partial_final_factor_month():
    svc=service()
    assert 'Stoch_K' in svc.technicals('TCS.NS','momentum').items[-1].values
    assert 'HV_20' in svc.technicals('TCS.NS','volatility').items[-1].values
    assert 'VolRatio' in svc.technicals('TCS.NS','volume').items[-1].values
    result=svc.factors('TCS.NS')
    # 650 business-day observations end mid-month: that month is not a
    # completed monthly return, even though it precedes today's date.
    last=svc.data.snapshot('TCS.NS').frame.index[-1].tz_convert('Asia/Kolkata').strftime('%Y-%m')
    assert last not in result.aligned_months

def test_risk_short_benchmark_is_nullable_not_synthetic_beta():
    svc=service()
    original=svc.data.provider
    from core.data.contracts import DataSnapshot
    def short(symbol,reference,timeout):
        snapshot=original(symbol,reference,timeout)
        return DataSnapshot(snapshot.frame.tail(5),snapshot.metadata) if reference else snapshot
    svc.data.provider=short
    result=svc.risk('TCS.NS')
    assert result.metrics['beta'] is None and result.benchmark_status=='INSUFFICIENT_ALIGNED_HISTORY'

def test_percent_units_and_declared_provider_dividend_units():
    from application.web_schemas import WatchlistRequest,ScreenRequest
    svc=service()
    svc.metadata_provider=lambda *args:{'name':'Company','dividend_yield':2.05,'dividend_yield_unit':'percent'}
    fundamentals=svc.fundamentals('TCS.NS')
    assert fundamentals.metrics.dividend_yield==pytest.approx(.0205)
    assert fundamentals.metrics.dividend_yield_provider_value==2.05
    assert fundamentals.dividend_yield_source_unit=='percent' and fundamentals.units['dividend_yield']=='fraction'
    assert svc.markets().change_percent_units=='percent'
    watch=svc.watchlist(WatchlistRequest(positions=[{'ticker':'TCS.NS','quantity':1,'average_buy':100}]))
    assert watch.percent_units=='percent'
    screen=svc.screen(ScreenRequest(tickers=['TCS.NS'],dividend_min=3))
    assert 'DIVIDEND_FILTER' in screen.items[0].reasons

def test_unknown_dividend_provider_unit_is_unavailable_for_financial_filter():
    from application.web_schemas import ScreenRequest
    svc=service();svc.metadata_provider=lambda *args:{'name':'Company','dividend_yield':2.05}
    result=svc.fundamentals('TCS.NS')
    assert result.metrics.dividend_yield is None and result.metrics.dividend_yield_provider_value==2.05
    assert result.dividend_yield_source_unit=='unverified'
    screen=svc.screen(ScreenRequest(tickers=['TCS.NS'],dividend_min=1))
    assert screen.items[0].status=='EXCLUDED' and 'DIVIDEND_UNIT_UNVERIFIED' in screen.items[0].reasons
