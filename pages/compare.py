"""Page 6 — Compare: side-by-side technical, risk, fundamentals, correlation."""
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data_fetcher  import (fetch_ohlcv, fetch_fundamentals, validate_ticker,
                                 PERIOD_MAP, detect_market, currency_symbol)
from core.indicators    import (rsi, macd, ema, bollinger_bands, historical_volatility,
                                 generate_signals)
from core.risk_metrics  import (full_risk_report, compute_returns, drawdown_series,
                                 annualised_volatility)
from utils.helpers      import (inject_css, section_header, kpi_row, kpi_card,
                                 fmt_price, fmt_pct, fmt_pct_plain, fmt_large, esc, sidebar_brand, footer_bar)
from utils.charts       import T, BASE, COLORS
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats
inject_css()

C1, C2 = "#3FB950", "#58A6FF"

with st.sidebar:
    sidebar_brand()
    st.divider()
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:16px;font-weight:600;color:#3FB950;padding:8px 0 16px;">📈 StockPro<span style="font-size:10px;color:#8B949E;font-weight:400;display:block;letter-spacing:.1em;margin-top:2px;">ANALYTICS TERMINAL</span></div>', unsafe_allow_html=True)
    st.markdown("**Stocks to Compare**")
    ticker1 = st.text_input("Stock A", value="RELIANCE.NS", help="US: AAPL | NSE: RELIANCE.NS").upper().strip()
    ticker2 = st.text_input("Stock B", value="TCS.NS",      help="US: MSFT | NSE: TCS.NS").upper().strip()
    period_label = st.selectbox("Time Period", list(PERIOD_MAP.keys()), index=3)
    period, interval = PERIOD_MAP[period_label]
    st.divider()
    show_norm = st.checkbox("Normalise prices (indexed 100)", value=True)
    show_ema  = st.checkbox("Show EMA 20/50",               value=True)
    st.divider()
    st.caption("Data via Yahoo Finance · Not financial advice")

# Detect markets for flags (before load, based on ticker text)
_f1 = "🇮🇳" if detect_market(ticker1 or "AAPL") in ("NSE","BSE") else "🇺🇸"
_f2 = "🇮🇳" if detect_market(ticker2 or "MSFT") in ("NSE","BSE") else "🇺🇸"

st.markdown(
    f'<div style="font-family:\'IBM Plex Mono\',monospace;padding:10px 0 6px;'
    f'border-bottom:1px solid #30363D;margin-bottom:16px">'
    f'<span style="font-size:20px;font-weight:600;color:{C1}">{_f1} {esc(ticker1 or "A")}</span>'
    f'&nbsp;<span style="font-size:16px;color:#8B949E">vs</span>&nbsp;'
    f'<span style="font-size:20px;font-weight:600;color:{C2}">{_f2} {esc(ticker2 or "B")}</span>'
    f'&nbsp;&nbsp;<span style="font-size:13px;color:#8B949E">Side-by-Side Comparison</span>'
    f'</div>', unsafe_allow_html=True)

if not ticker1 or not ticker2:
    st.info("Enter two ticker symbols in the sidebar.")
    st.stop()
if ticker1 == ticker2:
    st.warning("Enter two different tickers.")
    st.stop()

with st.spinner(f"Loading {ticker1} & {ticker2}…"):
    df1 = fetch_ohlcv(ticker1, period, interval)
    df2 = fetch_ohlcv(ticker2, period, interval)
    info1 = fetch_fundamentals(ticker1)
    info2 = fetch_fundamentals(ticker2)

for t, df in [(ticker1, df1), (ticker2, df2)]:
    if df.empty:
        st.error(f"No data for **{t}**.")
        st.stop()

shared = df1.index.intersection(df2.index)
df1 = df1.loc[shared]; df2 = df2.loc[shared]
ret1 = compute_returns(df1["Close"]); ret2 = compute_returns(df2["Close"])

_sym1 = currency_symbol("INR" if detect_market(ticker1) in ("NSE","BSE") else "USD")
_sym2 = currency_symbol("INR" if detect_market(ticker2) in ("NSE","BSE") else "USD")

# ── Price chart ────────────────────────────────────────────────────────────
section_header("Price Performance")
fig = go.Figure()
y1 = df1["Close"]/df1["Close"].iloc[0]*100 if show_norm else df1["Close"]
y2 = df2["Close"]/df2["Close"].iloc[0]*100 if show_norm else df2["Close"]
fig.add_trace(go.Scatter(x=y1.index,y=y1,name=ticker1,line=dict(color=C1,width=2)))
fig.add_trace(go.Scatter(x=y2.index,y=y2,name=ticker2,line=dict(color=C2,width=2)))
if show_ema and not show_norm:
    for t_,df_,c_ in [(ticker1,df1,C1),(ticker2,df2,C2)]:
        for span_ in [20,50]:
            fig.add_trace(go.Scatter(x=df_.index,y=ema(df_["Close"],span_),name=f"{t_} EMA{span_}",line=dict(color=c_,width=1,dash="dot"),opacity=0.55))
if show_norm:
    fig.add_hline(y=100,line_color=T["dim"],line_dash="dot",line_width=0.8)
fig.update_layout(**{**BASE,"height":400,"title":dict(text=f"{'Normalised' if show_norm else 'Absolute'} Price Comparison",font_size=12)},yaxis_title="Indexed to 100" if show_norm else "Price")
st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

# ── Return spread ──────────────────────────────────────────────────────────
section_header("Return Spread")
cum1=(1+ret1).cumprod()-1; cum2=(1+ret2).cumprod()-1; spread=(cum1-cum2)*100
fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.03,row_heights=[0.55,0.45])
fig.add_trace(go.Scatter(x=cum1.index,y=cum1*100,name=ticker1,line=dict(color=C1,width=1.8)),row=1,col=1)
fig.add_trace(go.Scatter(x=cum2.index,y=cum2*100,name=ticker2,line=dict(color=C2,width=1.8)),row=1,col=1)
fig.add_hline(y=0,line_color=T["dim"],line_dash="dot",line_width=0.8,row=1,col=1)
bar_c=["rgba(63,185,80,0.55)" if v>=0 else "rgba(248,81,73,0.5)" for v in spread]
fig.add_trace(go.Bar(x=spread.index,y=spread,name=f"{ticker1}–{ticker2}",marker_color=bar_c,opacity=0.8),row=2,col=1)
fig.add_hline(y=0,line_color=T["dim"],line_dash="dot",line_width=0.8,row=2,col=1)
fig.update_layout(**{**BASE,"height":440,"title":dict(text="Cumulative Return (%) & Spread",font_size=12)})
fig.update_yaxes(title_text="Cumul. Return (%)",row=1,col=1,gridcolor=T["grid"])
fig.update_yaxes(title_text="Spread (%)",row=2,col=1,gridcolor=T["grid"])
st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

# ── Indicator panels ────────────────────────────────────────────────────────
section_header("Technical Indicators")
tabs=st.tabs(["  RSI  ","  MACD  ","  Volatility  ","  Volume  "])
L={**BASE,"height":300,"margin":dict(l=8,r=8,t=36,b=8)}

with tabs[0]:
    fig=go.Figure()
    fig.add_hrect(y0=70,y1=100,fillcolor="rgba(248,81,73,0.05)",line_width=0)
    fig.add_hrect(y0=0,y1=30,fillcolor="rgba(63,185,80,0.05)",line_width=0)
    fig.add_trace(go.Scatter(x=rsi(df1["Close"]).index,y=rsi(df1["Close"]),name=f"{ticker1} RSI",line=dict(color=C1,width=1.8)))
    fig.add_trace(go.Scatter(x=rsi(df2["Close"]).index,y=rsi(df2["Close"]),name=f"{ticker2} RSI",line=dict(color=C2,width=1.8)))
    for y_,c_ in [(70,T["red"]),(30,T["green"]),(50,T["dim"])]:
        fig.add_hline(y=y_,line_color=c_,line_dash="dot",line_width=1)
    fig.update_yaxes(range=[0,100]); fig.update_layout(**{**L,"title":dict(text="RSI (14) Comparison",font_size=12)})
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

with tabs[1]:
    fig=make_subplots(rows=1,cols=2,subplot_titles=(f"{ticker1} MACD",f"{ticker2} MACD"))
    for ci,(t_,df_,c_) in enumerate([(ticker1,df1,C1),(ticker2,df2,C2)],1):
        md=macd(df_["Close"])
        fig.add_trace(go.Scatter(x=md.index,y=md["MACD"],name=f"{t_} MACD",line=dict(color=c_,width=1.8)),row=1,col=ci)
        fig.add_trace(go.Scatter(x=md.index,y=md["Signal"],name=f"{t_} Sig",line=dict(color=T["amber"],width=1.5)),row=1,col=ci)
        hc=["rgba(63,185,80,0.6)" if v>=0 else "rgba(248,81,73,0.6)" for v in md["Hist"]]
        fig.add_trace(go.Bar(x=md.index,y=md["Hist"],marker_color=hc,opacity=0.7,showlegend=False),row=1,col=ci)
    fig.update_layout(**{**L,"title":dict(text="MACD (12,26,9) Comparison",font_size=12)})
    for ax in ["xaxis","xaxis2","yaxis","yaxis2"]: fig.update_layout(**{ax:dict(gridcolor=T["grid"])})
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

with tabs[2]:
    hv1=historical_volatility(df1["Close"],20)*100; hv2=historical_volatility(df2["Close"],20)*100
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=hv1.index,y=hv1,name=f"{ticker1} HV20",line=dict(color=C1,width=1.8),fill="tozeroy",fillcolor="rgba(63,185,80,0.05)"))
    fig.add_trace(go.Scatter(x=hv2.index,y=hv2,name=f"{ticker2} HV20",line=dict(color=C2,width=1.8),fill="tozeroy",fillcolor="rgba(88,166,255,0.05)"))
    fig.update_layout(**{**L,"title":dict(text="Historical Volatility 20D (Ann. %)",font_size=12)},yaxis_title="Vol (%)")
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

with tabs[3]:
    fig=make_subplots(rows=1,cols=2,subplot_titles=(f"{ticker1} Volume",f"{ticker2} Volume"))
    for ci,(t_,df_,c_) in enumerate([(ticker1,df1,C1),(ticker2,df2,C2)],1):
        vc=["rgba(63,185,80,0.6)" if cl>=op else "rgba(248,81,73,0.6)" for cl,op in zip(df_["Close"],df_["Open"])]
        fig.add_trace(go.Bar(x=df_.index,y=df_["Volume"],marker_color=vc,opacity=0.65,name=t_),row=1,col=ci)
        fig.add_trace(go.Scatter(x=df_.index,y=df_["Volume"].rolling(20).mean(),name=f"{t_} 20D MA",line=dict(color=T["amber"],width=1.5)),row=1,col=ci)
    fig.update_layout(**{**L,"title":dict(text="Volume Comparison",font_size=12)})
    for ax in ["xaxis","xaxis2","yaxis","yaxis2"]: fig.update_layout(**{ax:dict(gridcolor=T["grid"])})
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

# ── Risk head to head ───────────────────────────────────────────────────────
section_header("Risk Metrics — Head to Head")
r1=full_risk_report(df1["Close"]); r2=full_risk_report(df2["Close"])
metrics=[
    ("Ann. Return",     fmt_pct(r1["annualised_return"]),    fmt_pct(r2["annualised_return"]),    "annualised_return",True),
    ("Ann. Volatility", f"{r1['annualised_volatility']*100:.1f}%", f"{r2['annualised_volatility']*100:.1f}%","annualised_volatility",False),
    ("Sharpe Ratio",    f"{r1['sharpe_ratio']:.3f}",         f"{r2['sharpe_ratio']:.3f}",         "sharpe_ratio",True),
    ("Sortino Ratio",   f"{r1['sortino_ratio']:.3f}",        f"{r2['sortino_ratio']:.3f}",        "sortino_ratio",True),
    ("Max Drawdown",    f"{r1['max_drawdown']*100:.1f}%",    f"{r2['max_drawdown']*100:.1f}%",    "max_drawdown",True),
    ("VaR 95%",         f"{r1['var_95_historical']*100:.2f}%",f"{r2['var_95_historical']*100:.2f}%","var_95_historical",False),
    ("CVaR 95%",        f"{r1['cvar_95']*100:.2f}%",         f"{r2['cvar_95']*100:.2f}%",         "cvar_95",False),
    ("Skewness",        f"{r1['skewness']:.3f}",             f"{r2['skewness']:.3f}",             "skewness",True),
    ("DD Duration",     f"{r1['max_drawdown_duration']}d",  f"{r2['max_drawdown_duration']}d",   "max_drawdown_duration",False),
]
rows_h=""
td="padding:8px 14px;font-family:'IBM Plex Mono',monospace"
for lbl,v1,v2,key,hb in metrics:
    w1=w2="#C9D1D9"
    if key:
        raw1=r1.get(key,0) or 0; raw2=r2.get(key,0) or 0
        if hb:
            if raw1>raw2: w1=C1
            elif raw2>raw1: w2=C2
        else:
            if raw1<raw2: w1=C1
            elif raw2<raw1: w2=C2
    rows_h+=f'<tr style="border-bottom:1px solid #21262D"><td style="{td};font-size:11px;color:#8B949E;text-transform:uppercase;text-align:center">{lbl}</td><td style="{td};font-size:14px;font-weight:600;color:{w1};text-align:center">{v1}</td><td style="{td};font-size:14px;font-weight:600;color:{w2};text-align:center">{v2}</td></tr>'
st.markdown(f'<table style="width:100%;border-collapse:collapse;background:#161B22;border:1px solid #30363D;border-radius:8px;overflow:hidden"><thead><tr style="background:#21262D"><th style="{td};font-size:10px;color:#8B949E;text-transform:uppercase;text-align:center">Metric</th><th style="{td};font-size:13px;color:{C1};font-weight:600;text-align:center">{ticker1}</th><th style="{td};font-size:13px;color:{C2};font-weight:600;text-align:center">{ticker2}</th></tr></thead><tbody>{rows_h}</tbody></table>',unsafe_allow_html=True)

# ── Fundamentals ────────────────────────────────────────────────────────────
section_header("Fundamentals — Head to Head")
fund=[
    ("Market Cap",    fmt_large(info1.get("market_cap")),    fmt_large(info2.get("market_cap"))),
    ("P/E (TTM)",     f'{info1["pe_ttm"]:.1f}' if info1.get("pe_ttm") else "—", f'{info2["pe_ttm"]:.1f}' if info2.get("pe_ttm") else "—"),
    ("EPS (TTM)",     fmt_price(info1.get("eps"),currency=_sym1), fmt_price(info2.get("eps"),currency=_sym2)),
    ("Beta",          f'{info1["beta"]:.2f}' if info1.get("beta") else "—", f'{info2["beta"]:.2f}' if info2.get("beta") else "—"),
    ("Div. Yield",    fmt_pct_plain(info1.get("dividend_yield",0)), fmt_pct_plain(info2.get("dividend_yield",0))),
    ("Revenue TTM",   fmt_large(info1.get("revenue_ttm")), fmt_large(info2.get("revenue_ttm"))),
    ("Gross Margin",  fmt_pct_plain(info1.get("gross_margin",0)), fmt_pct_plain(info2.get("gross_margin",0))),
    ("ROE",           fmt_pct_plain(info1.get("roe",0)), fmt_pct_plain(info2.get("roe",0))),
    ("Sector",        esc(info1.get("sector","—")), esc(info2.get("sector","—"))),
]
rows_f="".join(f'<tr style="border-bottom:1px solid #21262D"><td style="{td};font-size:10px;color:#8B949E;text-transform:uppercase;text-align:center">{k}</td><td style="{td};font-size:12px;color:#C9D1D9;text-align:center">{v1}</td><td style="{td};font-size:12px;color:#C9D1D9;text-align:center">{v2}</td></tr>' for k,v1,v2 in fund)
st.markdown(f'<table style="width:100%;border-collapse:collapse;background:#161B22;border:1px solid #30363D;border-radius:8px;overflow:hidden"><thead><tr style="background:#21262D"><th style="{td};font-size:10px;color:#8B949E;text-transform:uppercase;text-align:center">Metric</th><th style="{td};font-size:13px;color:{C1};font-weight:600;text-align:center">{ticker1}<br><span style="font-size:10px;color:#8B949E;font-weight:400">{esc(info1.get("name",ticker1)[:28])}</span></th><th style="{td};font-size:13px;color:{C2};font-weight:600;text-align:center">{ticker2}<br><span style="font-size:10px;color:#8B949E;font-weight:400">{esc(info2.get("name",ticker2)[:28])}</span></th></tr></thead><tbody>{rows_f}</tbody></table>',unsafe_allow_html=True)

# ── Drawdown + Correlation ──────────────────────────────────────────────────
col_dd, col_corr = st.columns([3,2])
with col_dd:
    section_header("Drawdown Comparison")
    dd1=drawdown_series(df1["Close"])*100; dd2=drawdown_series(df2["Close"])*100
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=dd1.index,y=dd1,name=ticker1,fill="tozeroy",fillcolor="rgba(63,185,80,0.12)",line=dict(color=C1,width=1.5)))
    fig.add_trace(go.Scatter(x=dd2.index,y=dd2,name=ticker2,fill="tozeroy",fillcolor="rgba(88,166,255,0.10)",line=dict(color=C2,width=1.5)))
    fig.update_layout(**{**BASE,"height":280,"title":dict(text="Drawdown Comparison (%)",font_size=12)},yaxis_title="Drawdown (%)")
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

with col_corr:
    section_header("Return Correlation")
    roll_c=ret1.rolling(30).corr(ret2); overall=float(ret1.corr(ret2))
    bar_c_=["rgba(63,185,80,0.6)" if v>=0 else "rgba(248,81,73,0.5)" for v in roll_c.fillna(0)]
    fig=go.Figure()
    fig.add_trace(go.Bar(x=roll_c.index,y=roll_c,marker_color=bar_c_,opacity=0.8,name="30D Corr"))
    fig.add_hline(y=overall,line_color=T["amber"],line_dash="dash",line_width=1.5,annotation_text=f" r={overall:.3f}",annotation_font_color=T["amber"])
    fig.add_hline(y=0,line_color=T["dim"],line_dash="dot",line_width=0.8)
    fig.update_layout(**{**BASE,"height":280,"title":dict(text=f"30-Day Rolling Correlation",font_size=12)},yaxis_range=[-1.1,1.1])
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

footer_bar()
