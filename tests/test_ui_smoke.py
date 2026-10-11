"""Baseline page smoke with network boundaries mocked; computations remain real."""
from pathlib import Path
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

PAGES = ['overview','technical_analysis','risk_analysis','portfolio','compare',
         'screener','market_overview','watchlist','backtester','factor_analysis','insights','global_data']

@pytest.fixture
def mock_providers(monkeypatch,candles):
    from core import data_fetcher as feed, external_data as ext
    monkeypatch.setattr(feed,'fetch_ohlcv',lambda *a,**kw:candles.copy())
    monkeypatch.setattr(feed,'validate_ticker',lambda *a,**kw:(True,''))
    monkeypatch.setattr(feed,'fetch_news',lambda *a,**kw:[])
    monkeypatch.setattr(feed,'fetch_fundamentals',lambda ticker:{'name':ticker,'currency':'USD',
        'sector':'Technology','market_cap':1e9,'pe_ttm':20.,'pe_fwd':19.,'eps':5.,
        'beta':1.,'dividend_yield':.01,'week52_high':150.,'week52_low':70.,
        'avg_volume_10d':1e6,'avg_volume_3m':1e6,'description':'Fixture',
        'website':'','employees':100,'exchange':'US','revenue_ttm':1e8,
        'gross_margin':.3,'operating_margin':.2,'roe':.2,'debt_equity':.3,'logo_url':''})
    for name in ['fetch_crypto_price']: monkeypatch.setattr(ext,name,lambda *a,**kw:{})
    for name in ['fetch_crypto_history','fetch_crypto_top_movers','fetch_sec_filings']:
        monkeypatch.setattr(ext,name,lambda *a,**kw:pd.DataFrame())
    for name in ['fetch_fx_history','fetch_macro_indicator']:
        monkeypatch.setattr(ext,name,lambda *a,**kw:pd.Series(dtype=float))
    for name in ['fetch_fx_rate','fetch_sec_cik']:
        monkeypatch.setattr(ext,name,lambda *a,**kw:None)

@pytest.mark.parametrize('page',PAGES)
def test_existing_page_initialization(page,mock_providers):
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root/'pages'/(page+'.py')),default_timeout=30)
    app.run()
    assert not app.exception, [e.message for e in app.exception]

def test_prediction_landing_with_explicit_normalized_checkpoint(mock_providers,monkeypatch):
    from core import models
    from core.validation.v4_adapter import load_frozen_v4
    monkeypatch.setattr(models,'_predictor_singleton',load_frozen_v4(normalize_text=True))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'pages/price_prediction.py'))
    app.run()
    assert not app.exception, [e.message for e in app.exception]
