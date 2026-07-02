"""Page 3 — Risk Analysis: VaR/CVaR, Monte Carlo, CAPM, drawdown, distributions."""
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data_fetcher  import fetch_ohlcv, validate_ticker, PERIOD_MAP, detect_market, currency_symbol
from core.risk_metrics  import (full_risk_report, monte_carlo, compute_returns,
                                 var_historical, annualised_return, annualised_volatility,
                                 drawdown_series)
from utils.helpers      import (inject_css, section_header, kpi_row, kpi_card,
                                 fmt_pct, fmt_pct_plain, esc, footer_bar, sidebar_brand)
from utils.charts       import (returns_distribution, drawdown_chart, monte_carlo_chart, T, BASE)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
inject_css()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    sidebar_brand()
    st.divider()
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:16px;font-weight:600;color:#3FB950;padding:8px 0 16px;">📈 StockPro<span style="font-size:10px;color:#8B949E;font-weight:400;display:block;letter-spacing:.1em;margin-top:2px;">ANALYTICS TERMINAL</span></div>', unsafe_allow_html=True)
    ticker = st.text_input("Ticker Symbol", value="AAPL", placeholder="AAPL · RELIANCE.NS").upper().strip()
    period_label = st.selectbox("Time Period", list(PERIOD_MAP.keys()), index=3)
    period, interval = PERIOD_MAP[period_label]
    st.divider()
    st.markdown("**VaR Settings**")
    confidence   = st.select_slider("Confidence Level", [0.90, 0.95, 0.99], value=0.95, format_func=lambda x: f"{int(x*100)}%")
    portfolio_val= st.number_input("Portfolio Value (local currency)", value=100_000, min_value=1_000, step=10_000)
    st.divider()
    st.markdown("**Monte Carlo**")
    mc_sims  = st.slider("Simulations",  100, 1000, 500, 100)
    mc_days  = st.slider("Horizon (days)", 5, 60, 30)
    # Auto-detect benchmark
    _mkt0 = detect_market(ticker) if ticker else "US"
    _auto_bench = {"NSE": "^NSEI", "BSE": "^BSESN"}.get(_mkt0, "^GSPC")
    _bench_help  = {"NSE": "Nifty=^NSEI · BankNifty=^NSEBANK", "BSE": "Sensex=^BSESN"}.get(_mkt0, "S&P=^GSPC · Nasdaq=^IXIC")
    benchmark_input = st.text_input("Benchmark Ticker", value=_auto_bench, help=_bench_help)
    st.divider()
    st.caption("Data via Yahoo Finance · Not financial advice")

# ── Load ───────────────────────────────────────────────────────────────────
if not ticker:
    st.info("Enter a ticker symbol in the sidebar.")
    st.stop()

with st.spinner(f"Loading {ticker}…"):
    valid, err = validate_ticker(ticker)
    if not valid: st.error(f"**{ticker}** — {err}"); st.stop()
    df    = fetch_ohlcv(ticker, period, interval)
    bench = fetch_ohlcv(benchmark_input.upper().strip() or "^GSPC", period, interval)

if df.empty or len(df) < 30:
    st.error("Not enough data. Try a longer period.")
    st.stop()

with st.spinner("Computing risk metrics…"):
    report = full_risk_report(df["Close"], bench["Close"] if not bench.empty else None)
    returns = compute_returns(df["Close"])

mkt  = detect_market(ticker)
_sym = currency_symbol("INR" if mkt in ("NSE","BSE") else "USD")
flag = "🇮🇳" if mkt in ("NSE","BSE") else "🇺🇸"

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="font-family:\'IBM Plex Mono\',monospace;padding:10px 0 6px;'
    f'border-bottom:1px solid #30363D;margin-bottom:16px">'
    f'<span style="font-size:20px;font-weight:600;color:#C9D1D9">{esc(ticker)}</span>'
    f'&nbsp;<span style="font-size:11px;color:#E3B341">{flag} {mkt}</span>'
    f'&nbsp;&nbsp;<span style="font-size:13px;color:#8B949E">Risk Analysis</span>'
    f'<span style="float:right;font-size:12px;color:#3FB950">'
    f'{len(df)} sessions · {df.index[0].strftime("%d %b %Y")} → {df.index[-1].strftime("%d %b %Y")}'
    f'</span></div>',
    unsafe_allow_html=True,
)

# ── KPIs ───────────────────────────────────────────────────────────────────
section_header("Risk Overview")
var95  = report["var_95_historical"]
cvar95 = report["cvar_95"]
ann_ret= report["annualised_return"]
ann_vol= report["annualised_volatility"]
sharpe = report.get("sharpe_ratio", 0)
sortino= report.get("sortino_ratio", 0)
mdd    = report["max_drawdown"]
beta   = report.get("beta")
alpha  = report.get("alpha")
kpi_row([
    kpi_card("Ann. Return",     fmt_pct(ann_ret),        period_label, "pos" if ann_ret>=0 else "neg"),
    kpi_card("Ann. Volatility", f"{ann_vol*100:.1f}%",   "σ × √252"),
    kpi_card("Sharpe Ratio",    f"{sharpe:.2f}",         ">1 = good", "pos" if sharpe>=1 else ("neg" if sharpe<0 else "")),
    kpi_card("Sortino Ratio",   f"{sortino:.2f}",        "Downside adj.", "pos" if sortino>=1 else ("neg" if sortino<0 else "")),
    kpi_card("Max Drawdown",    f"{mdd*100:.1f}%",       f"{report['max_drawdown_duration']}d", "neg"),
    kpi_card(f"VaR {int(confidence*100)}%", f"{var95*100:.2f}%", f"{_sym}{var95*portfolio_val:,.0f}", "neg"),
    kpi_card(f"CVaR {int(confidence*100)}%",f"{cvar95*100:.2f}%",f"{_sym}{cvar95*portfolio_val:,.0f}", "neg"),
    kpi_card("Beta",            f"{beta:.2f}" if beta else "—", f"vs {benchmark_input}"),
    kpi_card("Alpha (Ann.)",    f"{alpha*100:.2f}%" if alpha else "—", "Jensen's", "pos" if (alpha and alpha>0) else ("neg" if alpha else "")),
    kpi_card("Skewness",        f"{report['skewness']:.3f}", "Neg = fat left tail", "pos" if report['skewness']>0 else "neg"),
    kpi_card("Kurtosis",        f"{report['kurtosis']:.2f}", ">0 = fat tails"),
    kpi_card("Calmar Ratio",    f"{report.get('calmar_ratio',0):.2f}", "Return/MaxDD", "pos" if report.get('calmar_ratio',0)>0 else "neg"),
])

# ── Tabs ───────────────────────────────────────────────────────────────────
tabs = st.tabs(["  VaR & CVaR  ","  Drawdown  ","  Distribution  ","  Monte Carlo  ","  CAPM & Beta  "])

L = {**BASE, "height": 300, "margin": dict(l=8,r=8,t=36,b=8)}

# TAB 1: VaR
with tabs[0]:
    section_header(f"Value at Risk — {int(confidence*100)}% Confidence")
    c1, c2, c3 = st.columns(3)
    for col, lbl, val, note in [
        (c1, "Historical Simulation", report["var_95_historical"], "No distribution assumption"),
        (c2, "Parametric (Gaussian)",  report["var_95_parametric"], "Assumes normal distribution"),
        (c3, "Cornish-Fisher",         report["var_95_cf"],         "Adjusts for skew & kurtosis"),
    ]:
        with col:
            st.markdown(
                f'<div style="background:#161B22;border:1px solid #30363D;border-radius:8px;'
                f'padding:16px;text-align:center">'
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#8B949E;'
                f'text-transform:uppercase;margin-bottom:8px">{lbl}</div>'
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:28px;'
                f'font-weight:600;color:#F85149">{val*100:.2f}%</div>'
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;'
                f'color:#F85149;margin:4px 0">{_sym}{val*portfolio_val:,.0f}</div>'
                f'<div style="font-size:11px;color:#8B949E;margin-top:8px">{note}</div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col4, col5 = st.columns(2)
    with col4:
        section_header("CVaR at 95% and 99%")
        for conf_l, var_k, cvar_k in [(95,"var_95_historical","cvar_95"),(99,"var_99_historical","cvar_99")]:
            v_val = report[var_k]; cv_val = report[cvar_k]
            st.markdown(
                f'<div style="background:#161B22;border:1px solid #30363D;border-radius:6px;'
                f'padding:12px 16px;margin-bottom:8px;display:flex;justify-content:space-between">'
                f'<div><div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#8B949E">{conf_l}% VaR</div>'
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:18px;font-weight:600;color:#F85149">{v_val*100:.2f}%</div>'
                f'<div style="font-size:11px;color:#8B949E">{_sym}{v_val*portfolio_val:,.0f}</div></div>'
                f'<div style="text-align:right"><div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#8B949E">{conf_l}% CVaR/ES</div>'
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:18px;font-weight:600;color:#BC8CFF">{cv_val*100:.2f}%</div>'
                f'<div style="font-size:11px;color:#8B949E">{_sym}{cv_val*portfolio_val:,.0f}</div></div>'
                f'</div>', unsafe_allow_html=True)
    with col5:
        section_header("Rolling 21-Day VaR")
        roll_var = returns.rolling(21).apply(lambda x: -np.percentile(x,(1-confidence)*100),raw=True)*100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roll_var.index,y=roll_var,name=f"Rolling VaR {int(confidence*100)}%",
            line=dict(color=T["red"],width=1.5),fill="tozeroy",fillcolor="rgba(248,81,73,0.08)"))
        fig.update_layout(**{**L,"title":dict(text="Rolling 21-Day VaR (%)",font_size=12)})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# TAB 2: Drawdown
with tabs[1]:
    section_header("Drawdown Analysis")
    dd = report["drawdown_series"]
    col1, col2 = st.columns([2,1])
    with col1:
        st.plotly_chart(drawdown_chart(dd, ticker, height=340), use_container_width=True, config={"displayModeBar":False})
    with col2:
        items = [("Max Drawdown",f"{report['max_drawdown']*100:.2f}%","#F85149"),
                 ("DD Duration",f"{report['max_drawdown_duration']} days","#E3B341"),
                 ("Current DD",f"{dd.iloc[-1]*100:.2f}%","#F85149" if dd.iloc[-1]<-0.05 else "#8B949E"),
                 ("Sessions in DD",str((dd<0).sum()),"#8B949E"),
                 ("Avg DD Depth",f"{dd[dd<0].mean()*100:.2f}%","#FFA657")]
        st.markdown(
            '<div style="display:flex;flex-direction:column;gap:10px;padding-top:8px">'
            + "".join(f'<div style="background:#161B22;border:1px solid #30363D;border-radius:6px;padding:12px 14px">'
                      f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#8B949E;text-transform:uppercase">{k}</div>'
                      f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:18px;font-weight:600;color:{c}">{v}</div></div>'
                      for k,v,c in items)
            + '</div>', unsafe_allow_html=True)

    section_header("Equity Curve & Drawdown")
    cum_ret = (1+returns).cumprod()*100
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.02,row_heights=[0.65,0.35])
    fig.add_trace(go.Scatter(x=cum_ret.index,y=cum_ret,name="Equity",line=dict(color=T["green"],width=1.8)),row=1,col=1)
    fig.add_trace(go.Scatter(x=dd.index,y=dd*100,fill="tozeroy",fillcolor="rgba(248,81,73,0.2)",
        line=dict(color=T["red"],width=1),name="Drawdown %"),row=2,col=1)
    fig.update_layout(**{**BASE,"height":420,"title":dict(text="Equity Curve & Underwater Chart",font_size=12)})
    fig.update_yaxes(title_text="Indexed (100=start)",row=1,col=1,gridcolor=T["grid"])
    fig.update_yaxes(title_text="Drawdown (%)",row=2,col=1,gridcolor=T["grid"])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

# TAB 3: Distribution
with tabs[2]:
    section_header("Return Distribution")
    col1, col2 = st.columns([3,2])
    with col1:
        st.plotly_chart(returns_distribution(returns, ticker, height=380), use_container_width=True, config={"displayModeBar":False})
    with col2:
        sorted_r = np.sort(returns.dropna())
        n = len(sorted_r)
        theoretical = scipy_stats.norm.ppf(np.linspace(0.01,0.99,n))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=theoretical,y=sorted_r*100,mode="markers",name="Returns",
            marker=dict(color=T["blue"],size=3,opacity=0.6)))
        mn = min(theoretical.min(),(sorted_r*100).min()); mx = max(theoretical.max(),(sorted_r*100).max())
        fig.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode="lines",name="Normal",
            line=dict(color=T["amber"],width=1.5,dash="dot")))
        fig.update_layout(**{**BASE,"height":380,"title":dict(text="Q-Q Plot vs Normal",font_size=12)},
            xaxis_title="Theoretical Quantiles",yaxis_title="Sample Quantiles (%)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    section_header("Monthly Returns Heatmap")
    monthly = df["Close"].resample("ME").last().pct_change().dropna()*100
    m_df = pd.DataFrame({"Year":monthly.index.year,"Month":monthly.index.month,"R":monthly.values})
    m_pivot = m_df.pivot_table(index="Year",columns="Month",values="R")
    m_pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(m_pivot.columns)]
    fig = go.Figure(go.Heatmap(z=m_pivot.values,x=m_pivot.columns,y=m_pivot.index,
        colorscale=[[0,T["red"]],[0.5,T["card"]],[1,T["green"]]],zmid=0,
        text=np.round(m_pivot.values,1),texttemplate="%{text}%",textfont_size=9,
        colorbar=dict(tickfont_size=9)))
    from utils.charts import safe_layout
    fig.update_layout(**safe_layout(
        {"xaxis": dict(gridcolor=T["grid"]),
         "yaxis": dict(gridcolor=T["grid"], autorange="reversed")},
        height=320, title="Monthly Returns (%)"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

# TAB 4: Monte Carlo
with tabs[3]:
    section_header(f"Monte Carlo — {mc_sims} paths · {mc_days}-day horizon")
    with st.spinner("Running simulation…"):
        sim_df = monte_carlo(df["Close"], n_simulations=mc_sims, n_days=mc_days)
    st.plotly_chart(monte_carlo_chart(df["Close"], sim_df, ticker, height=460), use_container_width=True, config={"displayModeBar":False})

    section_header("Outcome Distribution")
    last_price = float(df["Close"].iloc[-1])
    final_ret  = (sim_df.iloc[-1]/last_price-1)*100
    col1, col2 = st.columns([2,1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=final_ret,nbinsx=60,name=f"Day {mc_days} Returns",marker_color=T["blue"],opacity=0.75))
        for pct, col_, lbl in [(5,T["red"],"5th"),(95,T["green"],"95th")]:
            fig.add_vline(x=float(final_ret.quantile(pct/100)),line_color=col_,line_dash="dash",line_width=1.5,annotation_text=f" {lbl}",annotation_font_color=col_)
        fig.update_layout(**{**BASE,"height":300,"title":dict(text=f"Day {mc_days} Return Distribution",font_size=12)},xaxis_title="Return (%)",yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    with col2:
        p5,p25,p50,p75,p95=[float(final_ret.quantile(q)) for q in [.05,.25,.50,.75,.95]]
        prob_pos = float((final_ret>0).mean())*100
        items = [("Expected Return",f"{float(final_ret.mean()):+.2f}%","#3FB950" if final_ret.mean()>0 else "#F85149"),
                 ("5th Percentile",f"{p5:+.2f}%","#F85149"),("Median",f"{p50:+.2f}%","#E3B341"),
                 ("95th Percentile",f"{p95:+.2f}%","#3FB950"),("Prob. Gain",f"{prob_pos:.1f}%","#3FB950" if prob_pos>50 else "#F85149")]
        st.markdown('<div style="display:flex;flex-direction:column;gap:8px;padding-top:8px">'
            + "".join(f'<div style="background:#161B22;border:1px solid #30363D;border-radius:6px;padding:10px 14px;display:flex;justify-content:space-between"><span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#8B949E;text-transform:uppercase">{k}</span><span style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;font-weight:600;color:{c}">{v}</span></div>'
                      for k,v,c in items)
            + '</div>', unsafe_allow_html=True)

# TAB 5: CAPM
with tabs[4]:
    section_header(f"CAPM Analysis vs {benchmark_input}")
    if bench.empty:
        st.warning(f"Could not load **{benchmark_input}**.")
    else:
        bench_ret = compute_returns(bench["Close"])
        aligned   = pd.concat([returns,bench_ret],axis=1,join="inner").dropna()
        aligned.columns = [ticker, benchmark_input]
        col1, col2 = st.columns([3,2])
        with col1:
            x_v=aligned[benchmark_input].values; y_v=aligned[ticker].values
            slope,intercept,r_val,_,_ = scipy_stats.linregress(x_v,y_v)
            x_l=np.linspace(x_v.min(),x_v.max(),200)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_v*100,y=y_v*100,mode="markers",name="Daily Returns",marker=dict(color=T["blue"],size=3,opacity=0.5)))
            fig.add_trace(go.Scatter(x=x_l*100,y=(slope*x_l+intercept)*100,mode="lines",name=f"β = {slope:.3f}",line=dict(color=T["amber"],width=2)))
            fig.add_hline(y=0,line_color=T["dim"],line_dash="dot",line_width=0.8)
            fig.add_vline(x=0,line_color=T["dim"],line_dash="dot",line_width=0.8)
            fig.update_layout(**{**BASE,"height":400,"title":dict(text=f"{ticker} vs {benchmark_input} — β={slope:.3f}  R²={r_val**2:.3f}",font_size=12)},xaxis_title=f"{benchmark_input} Daily Return (%)",yaxis_title=f"{ticker} Daily Return (%)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        with col2:
            capm = report
            td = "padding:7px 10px;border-bottom:1px solid #21262D;font-family:'IBM Plex Mono',monospace"
            rows = "".join(f'<tr><td style="{td};font-size:10px;color:#8B949E;text-transform:uppercase">{k}</td><td style="{td};font-size:13px;font-weight:600;color:#C9D1D9">{v}</td><td style="{td};font-size:11px;color:#8B949E">{n}</td></tr>'
                for k,v,n in [("Beta",f"{capm.get('beta',0):.4f}","Systematic risk"),("Alpha (Ann.)",f"{capm.get('alpha',0)*100:.3f}%","Excess return"),("R²",f"{capm.get('r_squared',0):.4f}","Explained var."),("Treynor",f"{capm.get('treynor',0):.4f}","Return per β"),("Sharpe",f"{capm.get('sharpe_ratio',0):.4f}","Risk-adj."),("Sortino",f"{capm.get('sortino_ratio',0):.4f}","Downside"),("Calmar",f"{capm.get('calmar_ratio',0):.4f}","Return/MaxDD")])
            section_header("CAPM Statistics")
            st.markdown(f'<table style="width:100%;border-collapse:collapse;background:#161B22;border:1px solid #30363D;border-radius:6px;overflow:hidden"><thead><tr style="background:#21262D"><th style="{td};font-size:9px;color:#8B949E;text-transform:uppercase">Metric</th><th style="{td};font-size:9px;color:#8B949E;text-transform:uppercase">Value</th><th style="{td};font-size:9px;color:#8B949E;text-transform:uppercase">Note</th></tr></thead><tbody>{rows}</tbody></table>',unsafe_allow_html=True)

        section_header("Rolling 63-Day Beta")
        rc = aligned[ticker].rolling(63).cov(aligned[benchmark_input])
        rv = aligned[benchmark_input].rolling(63).var()
        roll_beta = rc/rv
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roll_beta.index,y=roll_beta,name="Rolling Beta",line=dict(color=T["amber"],width=1.8),fill="tozeroy",fillcolor="rgba(227,179,65,0.07)"))
        fig.add_hline(y=1,line_color=T["dim"],line_dash="dot",annotation_text=" β=1 (market)")
        fig.add_hline(y=0,line_color=T["dim"],line_dash="dot",line_width=0.5)
        fig.update_layout(**{**BASE,"height":280,"title":dict(text=f"Rolling Beta vs {benchmark_input} (63-Day)",font_size=12)})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

footer_bar()
