"""Page 4 — Portfolio Tracker: multi-stock P&L, correlation, risk table, allocation."""
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="Portfolio · StockPro", page_icon="💼", layout="wide")

from core.data_fetcher  import fetch_ohlcv, PERIOD_MAP, detect_market, currency_symbol
from core.risk_metrics  import (compute_returns, annualised_return, annualised_volatility,
                                 sharpe_ratio, sortino_ratio, drawdown_analysis, var_historical)
from utils.helpers      import (inject_css, section_header, kpi_row, kpi_card,
                                 fmt_pct, fmt_large, esc)
from utils.charts       import correlation_heatmap, T, BASE, COLORS
import plotly.graph_objects as go
inject_css()

DEFAULT_TICKERS = "RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS"
DEFAULT_WEIGHTS = "20, 20, 20, 20, 20"

with st.sidebar:
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:16px;font-weight:600;color:#3FB950;padding:8px 0 16px;">📈 StockPro<span style="font-size:10px;color:#8B949E;font-weight:400;display:block;letter-spacing:.1em;margin-top:2px;">ANALYTICS TERMINAL</span></div>',unsafe_allow_html=True)
    raw_tickers = st.text_area("Portfolio Tickers", value=DEFAULT_TICKERS, help="NSE: add .NS  |  US: plain symbol", height=90)
    raw_weights = st.text_area("Weights (%)", value=DEFAULT_WEIGHTS, height=68)
    period_label = st.selectbox("Time Period", list(PERIOD_MAP.keys()), index=3)
    period, interval = PERIOD_MAP[period_label]
    # Auto benchmark
    tickers_raw = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()][:10]
    _first = tickers_raw[0] if tickers_raw else "AAPL"
    _auto_b = {"NSE":"^NSEI","BSE":"^BSESN"}.get(detect_market(_first),"^GSPC")
    benchmark_t = st.text_input("Benchmark", value=_auto_b, help="Auto-detected from first ticker")
    init_invest = st.number_input("Initial Investment", value=100_000, min_value=1_000, step=10_000)
    st.divider()
    st.caption("Data via Yahoo Finance · Not financial advice")

# ── Parse inputs ───────────────────────────────────────────────────────────
try:
    weights_raw = [float(w.strip()) for w in raw_weights.split(",") if w.strip()]
except:
    weights_raw = [100.0/len(tickers_raw)]*len(tickers_raw)

n = min(len(tickers_raw), len(weights_raw))
tickers_raw  = tickers_raw[:n]
weights_raw  = weights_raw[:n]
total_w      = sum(weights_raw)
weights_norm = [w/total_w for w in weights_raw]

if not tickers_raw:
    st.info("Enter at least one ticker."); st.stop()

# ── Load ───────────────────────────────────────────────────────────────────
with st.spinner("Loading portfolio data…"):
    prices_dict = {}
    failed = []
    for t in tickers_raw:
        df_t = fetch_ohlcv(t, period, interval)
        if not df_t.empty: prices_dict[t] = df_t["Close"]
        else: failed.append(t)

if failed: st.warning(f"Could not load: {', '.join(failed)}")
tickers = list(prices_dict.keys())
if not tickers: st.error("No valid tickers loaded."); st.stop()

weights_norm = [weights_norm[tickers_raw.index(t)] for t in tickers]
total_w2 = sum(weights_norm); weights_norm = [w/total_w2 for w in weights_norm]

prices_df  = pd.DataFrame(prices_dict).dropna()
returns_df = prices_df.pct_change().dropna()
port_ret   = (returns_df * weights_norm).sum(axis=1)
bench_df   = fetch_ohlcv(benchmark_t.upper().strip(), period, interval)
bench_ret  = compute_returns(bench_df["Close"]) if not bench_df.empty else None

# Currency
_mkts = {}
for t in tickers: m=detect_market(t); _mkts[m]=_mkts.get(m,0)+1
_dom = max(_mkts,key=_mkts.get)
_sym = currency_symbol("INR" if _dom in ("NSE","BSE") else "USD")
_flag = "🇮🇳" if _dom in ("NSE","BSE") else "🇺🇸"

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="font-family:\'IBM Plex Mono\',monospace;padding:10px 0 6px;'
    f'border-bottom:1px solid #30363D;margin-bottom:16px">'
    f'<span style="font-size:20px;font-weight:600;color:#C9D1D9">Portfolio Tracker</span>'
    f'&nbsp;<span style="font-size:11px;color:#E3B341">{_flag} {_dom}</span>'
    f'&nbsp;&nbsp;<span style="font-size:13px;color:#8B949E">'
    f'{len(tickers)} positions · {period_label} · {_sym}{init_invest:,.0f} invested</span></div>',
    unsafe_allow_html=True)

# ── Portfolio KPIs ─────────────────────────────────────────────────────────
section_header("Portfolio Overview")
p_ann = annualised_return(port_ret); p_vol = annualised_volatility(port_ret)
p_sh  = sharpe_ratio(port_ret);      p_so  = sortino_ratio(port_ret)
_,p_mdd,_ = drawdown_analysis((1+port_ret).cumprod()*init_invest)
p_cum = float((1+port_ret).prod()-1); p_var = var_historical(port_ret,0.95)
port_val = init_invest*(1+p_cum)
kpi_row([
    kpi_card("Portfolio Value",  f"{_sym}{port_val:,.0f}",  f"Started {_sym}{init_invest:,.0f}"),
    kpi_card("Total Return",     fmt_pct(p_cum),             period_label, "pos" if p_cum>=0 else "neg"),
    kpi_card("Ann. Return",      fmt_pct(p_ann),             "", "pos" if p_ann>=0 else "neg"),
    kpi_card("Ann. Volatility",  f"{p_vol*100:.1f}%",        ""),
    kpi_card("Sharpe Ratio",     f"{p_sh:.2f}",              ">1 = good", "pos" if p_sh>=1 else ""),
    kpi_card("Sortino Ratio",    f"{p_so:.2f}",              "Downside adj.", "pos" if p_so>=1 else ""),
    kpi_card("Max Drawdown",     f"{p_mdd*100:.1f}%",        "", "neg"),
    kpi_card("VaR (95%)",        f"{p_var*100:.2f}%",        f"{_sym}{p_var*port_val:,.0f}", "neg"),
])

# ── Tabs ───────────────────────────────────────────────────────────────────
tabs = st.tabs(["  Performance  ","  Allocation  ","  Correlation  ","  Risk Table  ","  Individual  "])

with tabs[0]:
    section_header("Cumulative Performance (Indexed to 100)")
    norm_p = prices_df.divide(prices_df.iloc[0])*100
    port_idx = (1+port_ret).cumprod()*100; port_idx.name = "Portfolio"
    plot_df = pd.concat([norm_p, port_idx], axis=1)
    if bench_ret is not None:
        bi = (1+bench_ret.reindex(port_idx.index).fillna(0)).cumprod()*100; bi.name=benchmark_t
        plot_df = pd.concat([plot_df, bi], axis=1)
    fig = go.Figure()
    for i,col in enumerate(plot_df.columns):
        is_p=(col=="Portfolio"); is_b=(col==benchmark_t)
        c_=("#FFFFFF" if is_p else ("#8B949E" if is_b else COLORS[i%len(COLORS)]))
        w_=(2.5 if is_p else (1.5 if is_b else 1.2)); d_=("dot" if not(is_p or is_b) else "solid")
        fig.add_trace(go.Scatter(x=plot_df.index,y=plot_df[col],name=col,line=dict(color=c_,width=w_,dash=d_),opacity=0.9 if(is_p or is_b) else 0.75))
    fig.add_hline(y=100,line_color=T["dim"],line_dash="dot",line_width=0.8)
    fig.update_layout(**{**BASE,"height":460,"title":dict(text="Normalised Performance (100 = start)",font_size=12)},yaxis_title="Indexed")
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

with tabs[1]:
    col1,col2 = st.columns([1,1])
    with col1:
        fig = go.Figure(go.Pie(labels=tickers,values=[w*100 for w in weights_norm],hole=0.55,
            marker=dict(colors=COLORS[:len(tickers)],line=dict(color="#0D1117",width=2)),
            textinfo="label+percent",textfont=dict(family="IBM Plex Mono, monospace",size=10)))
        fig.update_layout(**{**BASE,"height":380,"title":dict(text="Allocation (%)",font_size=12)},showlegend=False,
            annotations=[dict(text="Portfolio",x=0.5,y=0.5,showarrow=False,font_size=13,font_color="#C9D1D9")])
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    with col2:
        section_header("Position Summary")
        td="padding:7px 10px;border-bottom:1px solid #21262D;font-family:'IBM Plex Mono',monospace"
        rows=""
        for t,w in zip(tickers,weights_norm):
            lp=float(prices_df[t].iloc[-1]); fp=float(prices_df[t].iloc[0])
            tr=(lp/fp-1)*100; tv=w*init_invest*(1+tr/100)
            c_="#3FB950" if tr>=0 else "#F85149"; sgn="+" if tr>=0 else ""
            rows+=f'<tr><td style="{td};font-size:12px;font-weight:600;color:#C9D1D9">{esc(t)}</td><td style="{td};font-size:11px;color:#8B949E">{w*100:.1f}%</td><td style="{td};font-size:12px;color:#C9D1D9">{_sym}{lp:,.2f}</td><td style="{td};font-size:12px;font-weight:600;color:{c_}">{sgn}{tr:.2f}%</td><td style="{td};font-size:11px;color:#8B949E">{_sym}{tv:,.0f}</td></tr>'
        st.markdown(f'<table style="width:100%;border-collapse:collapse;background:#161B22;border:1px solid #30363D;border-radius:6px;overflow:hidden"><thead><tr style="background:#21262D"><th style="{td};font-size:9px;color:#8B949E;text-transform:uppercase">Ticker</th><th style="{td};font-size:9px;color:#8B949E;text-transform:uppercase">Wt</th><th style="{td};font-size:9px;color:#8B949E;text-transform:uppercase">Price</th><th style="{td};font-size:9px;color:#8B949E;text-transform:uppercase">Return</th><th style="{td};font-size:9px;color:#8B949E;text-transform:uppercase">Value</th></tr></thead><tbody>{rows}</tbody></table>',unsafe_allow_html=True)

with tabs[2]:
    section_header("Return Correlation Matrix")
    if len(tickers)>=2:
        st.plotly_chart(correlation_heatmap(returns_df,height=440),use_container_width=True,config={"displayModeBar":False})
    else:
        st.info("Add at least 2 tickers.")

with tabs[3]:
    section_header("Risk-Adjusted Performance Table")
    rows_d = []
    for t,w in zip(tickers,weights_norm):
        r_s=returns_df[t]; p_s=prices_df[t]
        rows_d.append({"Ticker":t,"Weight (%)":round(w*100,1),
            "Cumul. Return":round((float(p_s.iloc[-1]/p_s.iloc[0])-1)*100,2),
            "Ann. Return":round(annualised_return(r_s)*100,2),
            "Ann. Vol (%)":round(annualised_volatility(r_s)*100,2),
            "Sharpe":round(sharpe_ratio(r_s),3),"Sortino":round(sortino_ratio(r_s),3),
            "Max DD (%)":round(drawdown_analysis(p_s)[1]*100,2),
            "VaR 95% (%)":round(var_historical(r_s,0.95)*100,2)})
    risk_df=pd.DataFrame(rows_d).set_index("Ticker")
    st.dataframe(risk_df.style.format({"Weight (%)":"{:.1f}%","Cumul. Return":"{:+.2f}%","Ann. Return":"{:+.2f}%","Ann. Vol (%)":"{:.2f}%","Sharpe":"{:.3f}","Sortino":"{:.3f}","Max DD (%)":"{:.2f}%","VaR 95% (%)":"{:.2f}%"}).background_gradient(subset=["Sharpe"],cmap="RdYlGn",vmin=-1,vmax=2).background_gradient(subset=["Max DD (%)"],cmap="RdYlGn_r",vmin=-50,vmax=0),use_container_width=True)
    section_header("Risk / Return Map")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=[risk_df.loc[t,"Ann. Vol (%)"] for t in tickers],y=[risk_df.loc[t,"Ann. Return"] for t in tickers],mode="markers+text",text=tickers,textposition="top center",marker=dict(color=[risk_df.loc[t,"Sharpe"] for t in tickers],colorscale=[[0,T["red"]],[0.5,T["amber"]],[1,T["green"]]],size=14,opacity=0.9,colorbar=dict(title="Sharpe",tickfont_size=9),showscale=True),textfont=dict(family="IBM Plex Mono, monospace",size=9,color="#C9D1D9")))
    fig.add_trace(go.Scatter(x=[annualised_volatility(port_ret)*100],y=[annualised_return(port_ret)*100],mode="markers+text",text=["Portfolio"],textposition="top center",marker=dict(color="#FFFFFF",size=16,symbol="star"),textfont=dict(family="IBM Plex Mono, monospace",size=11,color="#FFFFFF"),name="Portfolio"))
    fig.add_hline(y=0,line_color=T["dim"],line_dash="dot",line_width=0.8)
    fig.update_layout(**{**BASE,"height":420,"title":dict(text="Risk / Return (colour = Sharpe)",font_size=12)},xaxis_title="Ann. Volatility (%)",yaxis_title="Ann. Return (%)",showlegend=False)
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

with tabs[4]:
    section_header("Individual Stock Charts")
    sel=st.selectbox("Select Ticker",tickers)
    df_sel=fetch_ohlcv(sel,period,interval)
    if not df_sel.empty:
        from core.indicators import ema as _ema
        fig=go.Figure()
        fig.add_trace(go.Candlestick(x=df_sel.index,open=df_sel["Open"],high=df_sel["High"],low=df_sel["Low"],close=df_sel["Close"],name=sel,increasing_line_color=T["green"],decreasing_line_color=T["red"],increasing_fillcolor=T["green"],decreasing_fillcolor=T["red"]))
        for span_,col_ in [(20,T["amber"]),(50,T["purple"])]:
            fig.add_trace(go.Scatter(x=df_sel.index,y=_ema(df_sel["Close"],span_),name=f"EMA {span_}",line=dict(color=col_,width=1.5)))
        fig.update_layout(**{**BASE,"height":420,"title":dict(text=f"{sel} — OHLCV + EMA 20/50",font_size=12)})
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        _sel_sym=currency_symbol("INR" if detect_market(sel) in ("NSE","BSE") else "USD")
        _last=float(df_sel["Close"].iloc[-1]); _first=float(df_sel["Close"].iloc[0])
        _chg=(_last/_first-1)*100
        kpi_row([kpi_card("Last Close",f"{_sel_sym}{_last:,.2f}",""),kpi_card("Period Return",f"{_chg:+.2f}%","","pos" if _chg>=0 else "neg"),kpi_card("Sharpe",f"{sharpe_ratio(compute_returns(df_sel['Close'])):.2f}","","pos" if sharpe_ratio(compute_returns(df_sel['Close']))>=1 else ""),kpi_card("Max DD",f"{drawdown_analysis(df_sel['Close'])[1]*100:.1f}%","","neg")])
