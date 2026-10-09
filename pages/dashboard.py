"""V5 dashboard: research state, recent outcomes and a single upgrade task board."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from core.forecasting.service import ROOT,active_research
from core.forecasting.ledger import PredictionLedger
from core.data_fetcher import fetch_ohlcv
from utils.ui_v5 import inject_theme,percent

inject_theme()
st.title('StockPro Analytics')
st.caption('Research workbench · Direct forecasts, evidence and risk')
st.warning('V5 is research-only. Historical availability/revision vintages are unverified; production promotion is blocked.')
try:
    pointer=active_research()
    version=pointer['model_version']
except Exception as error:
    version='Unavailable'
    st.error(str(error))
records=PredictionLedger(ROOT/'data/v5/predictions.sqlite').rows()
matured=[r for r in records if r['outcome']]
columns=st.columns(4)
columns[0].metric('Research forecasts',len(records))
columns[1].metric('Matured outcomes',len(matured))
columns[2].metric('Abstained forecasts',sum(r['abstain'] for r in records))
columns[3].metric('Production promotion','Blocked')
st.caption('Active research candidate: '+version)
st.subheader('Research workflow')
left,right=st.columns(2)
with left:
    st.page_link('pages/forecast.py',label='Forecast — distributions, model associations and scenarios')
    st.page_link('pages/validate.py',label='Validate — reliability, coverage and full prediction history')
with right:
    st.page_link('pages/overview.py',label='Research — price, fundamentals and technical context')
    st.page_link('pages/risk_analysis.py',label='Risk — drawdown, tail risk and portfolio exposure')

st.subheader('Markets')
if st.button('Refresh market snapshot'):
    rows=[]
    for name,symbol in [('Nifty 50','^NSEI'),('S&P 500','^GSPC'),('India VIX','^INDIAVIX'),('US VIX','^VIX')]:
        bars=fetch_ohlcv(symbol,'5d','1d')
        if bars.empty or len(bars)<2: rows.append({'Market':name,'Last':'Unavailable','Change':'Unavailable','Source session':'Unavailable'})
        else:
            rows.append({'Market':name,'Last':round(float(bars.Close.iloc[-1]),2),
                'Change':percent(float(bars.Close.iloc[-1]/bars.Close.iloc[-2]-1)),
                'Source session':str(bars.index[-1].date())})
    st.session_state['v5_market_snapshot']=rows
if st.session_state.get('v5_market_snapshot'):
    st.dataframe(pd.DataFrame(st.session_state['v5_market_snapshot']),hide_index=True,use_container_width=True)
    st.caption('Provider observations may be delayed. These quotes are not certified live or exchange-grade.')
else: st.caption('Refresh to inspect a dated market snapshot. Portfolio and watchlist tools remain under Risk & Portfolio.')

st.subheader('Research progress')
with st.expander('One master task board — completed, active, pending and blocked',expanded=True):
    board=ROOT/'docs/NEXT_RESEARCH_TASK_BOARD.md'
    st.markdown((board if board.exists() else ROOT/'docs/V5_TASK_BOARD.md').read_text(encoding='utf-8'))
if records:
    st.subheader('Recent research predictions')
    st.dataframe(pd.DataFrame([{'Ticker':r['ticker'],'Horizon':r['horizon'],'As of':r['forecast_as_of'],
        'Status':r['outcome']['status'] if r['outcome'] else 'pending','Abstained':r['abstain']} for r in records[-10:]]),hide_index=True,use_container_width=True)
