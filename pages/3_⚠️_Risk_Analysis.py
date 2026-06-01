"""
Page 3 — Risk Analysis
VaR / CVaR (3 methods), Monte Carlo simulation, CAPM beta/alpha,
drawdown analysis, return distribution, and correlation vs benchmark.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="Risk Analysis · StockPro", page_icon="⚠️",
                   layout="wide", initial_sidebar_state="expanded")

from core.data_fetcher import fetch_ohlcv, fetch_benchmark, validate_ticker, PERIOD_MAP, detect_market, currency_symbol
from core.risk_metrics import (
    full_risk_report, monte_carlo, compute_returns,
    annualised_return, annualised_volatility
)
from utils.helpers     import (inject_css, section_header, kpi_row, kpi_card,
                               fmt_pct, fmt_pct_plain, fmt_price)
from utils.charts      import (returns_distribution, drawdown_chart,
                               monte_carlo_chart, THEME)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

inject_css()

LAYOUT_BASE = dict(
    plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
    font=dict(family="IBM Plex Mono, monospace", color="#C9D1D9", size=11),
    margin=dict(l=12, r=12, t=36, b=12),
    xaxis=dict(gridcolor="#21262D", zeroline=False),
    yaxis=dict(gridcolor="#21262D", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:16px;
    font-weight:600;color:#3FB950;padding:8px 0 16px;">📈 StockPro
    <span style="font-size:10px;color:#8B949E;font-weight:400;display:block;
    letter-spacing:.1em;margin-top:2px;">ANALYTICS TERMINAL</span></div>""",
    unsafe_allow_html=True)

    ticker = st.text_input(
        "Ticker Symbol", value="AAPL",
        placeholder="AAPL · RELIANCE.NS · TCS.NS",
        help="US: AAPL  |  NSE: RELIANCE.NS  |  BSE: RELIANCE.BO"
    ).upper().strip()

    # Auto-suggest benchmark based on market
    _auto_bench = {"NSE": "^NSEI", "BSE": "^BSESN"}.get(detect_market(ticker), "^GSPC")
    _bench_hint = {"NSE": "Nifty 50 = ^NSEI · Bank Nifty = ^NSEBANK",
                   "BSE": "Sensex = ^BSESN"}.get(detect_market(ticker),
                   "S&P 500 = ^GSPC · Nasdaq = ^IXIC")
    period_label = st.selectbox("Time Period", list(PERIOD_MAP.keys()), index=3)
    period, interval = PERIOD_MAP[period_label]

    st.divider()
    st.markdown("**VaR Settings**")
    confidence = st.select_slider("Confidence Level", [0.90, 0.95, 0.99], value=0.95,
                                  format_func=lambda x: f"{int(x*100)}%")
    portfolio_val = st.number_input("Portfolio Value (local currency)", value=100_000,
                                    min_value=1_000, step=10_000,
                                    help="Used to compute dollar VaR / CVaR")

    st.divider()
    st.markdown("**Monte Carlo**")
    mc_sims  = st.slider("Simulations",  100, 1000, 500, 100)
    mc_days  = st.slider("Horizon (days)", 5, 60, 30)

    st.divider()
    benchmark_input = st.text_input("Benchmark Ticker", value=_auto_bench,
                                    help=_bench_hint)
    st.caption("Data via Yahoo Finance · Not financial advice")

# ─── LOAD DATA ──────────────────────────────────────────────────────────────
if not ticker:
    st.info("Enter a ticker symbol in the sidebar.")
    st.stop()

with st.spinner(f"Loading {ticker}…"):
    valid, err = validate_ticker(ticker)
    if not valid:
        st.error(f"**{ticker}** — {err}")
        st.stop()
    df    = fetch_ohlcv(ticker, period, interval)
    bench = fetch_ohlcv(benchmark_input.upper().strip() or "^GSPC", period, interval)

if df.empty or len(df) < 30:
    st.error("Not enough data. Try a longer period.")
    st.stop()

# ─── COMPUTE RISK METRICS ───────────────────────────────────────────────────
with st.spinner("Computing risk metrics…"):
    report = full_risk_report(
        df["Close"],
        bench["Close"] if not bench.empty else None
    )
    returns = compute_returns(df["Close"])

# ─── HEADER ─────────────────────────────────────────────────────────────────
_mkt  = detect_market(ticker)
_curr = "INR" if _mkt in ("NSE","BSE") else "USD"
_sym  = currency_symbol(_curr)
_flag = "🇮🇳" if _mkt in ("NSE","BSE") else "🇺🇸"

st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;padding:10px 0 6px;
border-bottom:1px solid #30363D;margin-bottom:16px;">
<span style="font-size:20px;font-weight:600;color:#C9D1D9">{ticker}</span>&nbsp;&nbsp;
<span style="font-size:11px;color:#E3B341;font-family:'IBM Plex Mono',monospace">{_flag} {_mkt}</span>&nbsp;&nbsp;
<span style="font-size:13px;color:#8B949E">Risk Analysis</span>
<span style="float:right;font-size:12px;color:#3FB950">
  {len(df)} sessions · {df.index[0].strftime('%d %b %Y')} → {df.index[-1].strftime('%d %b %Y')}
</span></div>""", unsafe_allow_html=True)

# ─── KPI ROW ────────────────────────────────────────────────────────────────
section_header("Risk Overview")
ann_ret  = report["annualised_return"]
ann_vol  = report["annualised_volatility"]
sharpe   = report.get("sharpe_ratio", 0)
sortino  = report.get("sortino_ratio", 0)
mdd      = report["max_drawdown"]
var95    = report["var_95_historical"]
cvar95   = report["cvar_95"]
beta     = report.get("beta")
alpha    = report.get("alpha")

kpi_row([
    kpi_card("Ann. Return",    fmt_pct(ann_ret),  period_label,
             "pos" if ann_ret >= 0 else "neg"),
    kpi_card("Ann. Volatility",f"{ann_vol*100:.1f}%", "Std dev × √252"),
    kpi_card("Sharpe Ratio",   f"{sharpe:.2f}",   ">1 = good · >2 = excellent",
             "pos" if sharpe >= 1 else ("neg" if sharpe < 0 else "")),
    kpi_card("Sortino Ratio",  f"{sortino:.2f}",  "Downside-adj. Sharpe",
             "pos" if sortino >= 1 else ("neg" if sortino < 0 else "")),
    kpi_card("Max Drawdown",   f"{mdd*100:.1f}%", f"{report['max_drawdown_duration']}d duration",
             "neg"),
    kpi_card(f"VaR ({int(confidence*100)}%)", f"{var95*100:.2f}%",
             f"{_sym}{var95*portfolio_val:,.0f} on {_sym}{portfolio_val:,}", "neg"),
    kpi_card(f"CVaR ({int(confidence*100)}%)", f"{cvar95*100:.2f}%",
             f"{_sym}{cvar95*portfolio_val:,.0f} expected loss", "neg"),
    kpi_card("Beta",           f"{beta:.2f}" if beta else "—",
             f"vs {benchmark_input}"),
    kpi_card("Alpha (Ann.)",   f"{alpha*100:.2f}%" if alpha else "—",
             "Jensen's Alpha",
             "pos" if (alpha and alpha > 0) else ("neg" if alpha else "")),
    kpi_card("Skewness",       f"{report['skewness']:.3f}",
             "Negative = fat left tail",
             "pos" if report["skewness"] > 0 else "neg"),
    kpi_card("Kurtosis",       f"{report['kurtosis']:.2f}",
             ">0 = fat tails"),
    kpi_card("Calmar Ratio",   f"{report.get('calmar_ratio',0):.2f}",
             "Return / Max DD",
             "pos" if report.get("calmar_ratio",0) > 0 else "neg"),
])

# ─── TABS ───────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "  VaR & CVaR  ",
    "  Drawdown  ",
    "  Distribution  ",
    "  Monte Carlo  ",
    "  CAPM & Beta  ",
])

# ════════════════════════════════════════════
# TAB 1 — VaR & CVaR
# ════════════════════════════════════════════
with tabs[0]:
    section_header(f"Value at Risk & Expected Shortfall — {int(confidence*100)}% Confidence")

    # Three-method comparison
    col1, col2, col3 = st.columns(3)
    var_hist  = report["var_95_historical"]
    var_param = report["var_95_parametric"]
    var_cf    = report["var_95_cf"]
    cvar_v    = report["cvar_95"]

    for col, label, var_v, note in [
        (col1, "Historical Simulation", var_hist,
         "No distribution assumption.\nDirectly uses past returns."),
        (col2, "Parametric (Gaussian)", var_param,
         "Assumes normal distribution.\nFast but underestimates fat tails."),
        (col3, "Cornish-Fisher", var_cf,
         "Adjusts for skew & kurtosis.\nBest for non-normal returns."),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#161B22;border:1px solid #30363D;border-radius:8px;
            padding:16px;text-align:center;">
              <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                          color:#8B949E;text-transform:uppercase;letter-spacing:.08em;
                          margin-bottom:8px">{label}</div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:28px;
                          font-weight:600;color:#F85149">{var_v*100:.2f}%</div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;
                          color:#F85149;margin:4px 0">{_sym}{var_v*portfolio_val:,.0f}</div>
              <div style="font-size:11px;color:#8B949E;margin-top:8px;
                          line-height:1.5">{note}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # CVaR comparison at 95% and 99%
    col4, col5 = st.columns(2)
    with col4:
        section_header("Expected Shortfall (CVaR)")
        for conf_level, var_key, cvar_key in [
            (95, "var_95_historical", "cvar_95"),
            (99, "var_99_historical", "cvar_99"),
        ]:
            v = report[var_key]
            cv = report[cvar_key]
            st.markdown(f"""
            <div style="background:#161B22;border:1px solid #30363D;border-radius:6px;
            padding:12px 16px;margin-bottom:8px;display:flex;justify-content:space-between;
            align-items:center;">
              <div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                            color:#8B949E;text-transform:uppercase">{conf_level}% VaR</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:18px;
                            font-weight:600;color:#F85149">{v*100:.2f}%</div>
                <div style="font-size:11px;color:#8B949E">{_sym}{v*portfolio_val:,.0f}</div>
              </div>
              <div style="text-align:right;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                            color:#8B949E;text-transform:uppercase">{conf_level}% CVaR / ES</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:18px;
                            font-weight:600;color:#BC8CFF">{cv*100:.2f}%</div>
                <div style="font-size:11px;color:#8B949E">{_sym}{cv*portfolio_val:,.0f}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    with col5:
        # VaR over rolling windows
        section_header("Rolling 1-Month VaR (Historical)")
        roll_var = returns.rolling(21).apply(
            lambda x: -np.percentile(x, (1 - confidence) * 100), raw=True) * 100
        fig_rvar = go.Figure()
        fig_rvar.add_trace(go.Scatter(x=roll_var.index, y=roll_var,
            name=f"Rolling VaR {int(confidence*100)}%",
            line=dict(color="#F85149", width=1.5),
            fill="tozeroy", fillcolor="rgba(248,81,73,0.08)"))
        fig_rvar.update_layout(**{**LAYOUT_BASE, "height": 250,
            "title": dict(text="Rolling 21-Day VaR (%)", font_size=12)})
        st.plotly_chart(fig_rvar, use_container_width=True,
                        config={"displayModeBar": False})

# ════════════════════════════════════════════
# TAB 2 — DRAWDOWN
# ════════════════════════════════════════════
with tabs[1]:
    section_header("Drawdown Analysis")

    dd = report["drawdown_series"]
    mdd_val = report["max_drawdown"]
    mdd_dur = report["max_drawdown_duration"]

    col1, col2 = st.columns([2, 1])
    with col1:
        fig_dd = drawdown_chart(dd, ticker, height=340)
        st.plotly_chart(fig_dd, use_container_width=True,
                        config={"displayModeBar": False})

    with col2:
        # Drawdown stats
        dd_pct = dd * 100
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;gap:10px;padding-top:8px">
          {"".join(f'''<div style="background:#161B22;border:1px solid #30363D;border-radius:6px;
            padding:12px 14px;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                        color:#8B949E;text-transform:uppercase;letter-spacing:.06em">{k}</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:18px;
                        font-weight:600;color:{vc}">{v}</div></div>'''
            for k, v, vc in [
                ("Max Drawdown",     f"{mdd_val*100:.2f}%",    "#F85149"),
                ("DD Duration",      f"{mdd_dur} days",         "#E3B341"),
                ("Current DD",       f"{dd.iloc[-1]*100:.2f}%", "#F85149" if dd.iloc[-1] < -0.05 else "#8B949E"),
                ("Sessions in DD",   f"{(dd < 0).sum()}",       "#8B949E"),
                ("Avg DD Depth",     f"{dd[dd<0].mean()*100:.2f}%", "#FFA657"),
                ("DD Skew",          f"{dd[dd<0].skew():.2f}",  "#8B949E"),
            ])}
        </div>""", unsafe_allow_html=True)

    # Underwater equity curve
    section_header("Cumulative Return & Drawdown (Overlaid)")
    cum_ret = (1 + returns).cumprod() * 100
    fig_uw = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.02, row_heights=[0.65, 0.35])
    fig_uw.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret, name="Equity Curve",
        line=dict(color="#3FB950", width=1.8)), row=1, col=1)
    fig_uw.add_trace(go.Scatter(x=dd.index, y=dd * 100,
        fill="tozeroy", fillcolor="rgba(248,81,73,0.2)",
        line=dict(color="#F85149", width=1), name="Drawdown (%)"), row=2, col=1)
    fig_uw.update_layout(
        **{**LAYOUT_BASE, "height": 440,
           "title": dict(text="Equity Curve & Underwater Chart", font_size=12)})
    fig_uw.update_yaxes(title_text="Indexed (100=start)", row=1, col=1,
                        gridcolor="#21262D")
    fig_uw.update_yaxes(title_text="Drawdown (%)", row=2, col=1,
                        gridcolor="#21262D")
    st.plotly_chart(fig_uw, use_container_width=True,
                    config={"displayModeBar": False})

# ════════════════════════════════════════════
# TAB 3 — DISTRIBUTION
# ════════════════════════════════════════════
with tabs[2]:
    section_header("Return Distribution Analysis")

    col1, col2 = st.columns([3, 2])
    with col1:
        fig_rd = returns_distribution(returns, ticker, height=380)
        st.plotly_chart(fig_rd, use_container_width=True,
                        config={"displayModeBar": False})

    with col2:
        # QQ plot
        from scipy import stats
        sorted_returns = np.sort(returns.dropna())
        n = len(sorted_returns)
        theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, n))
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=theoretical, y=sorted_returns * 100,
            mode="markers", name="Returns",
            marker=dict(color="#58A6FF", size=3, opacity=0.6)))
        # 45-degree reference line
        mn = min(theoretical.min(), (sorted_returns*100).min())
        mx = max(theoretical.max(), (sorted_returns*100).max())
        fig_qq.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx],
            mode="lines", name="Normal",
            line=dict(color="#E3B341", width=1.5, dash="dot")))
        fig_qq.update_layout(
            **{**LAYOUT_BASE, "height": 380,
               "title": dict(text="Q-Q Plot vs Normal", font_size=12)},
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles (%)",
        )
        st.plotly_chart(fig_qq, use_container_width=True,
                        config={"displayModeBar": False})

    # Monthly returns heatmap
    section_header("Monthly Returns Heatmap")
    df_ret = df["Close"].resample("ME").last().pct_change().dropna()
    monthly = df_ret.copy()
    monthly.index = pd.to_datetime(monthly.index)
    m_df = pd.DataFrame({
        "Year":  monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values * 100,
    }).pivot_table(index="Year", columns="Month", values="Return")
    m_df.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig_mh = go.Figure(go.Heatmap(
        z=m_df.values,
        x=m_df.columns,
        y=m_df.index,
        colorscale=[[0.0,"#F85149"],[0.5,"#21262D"],[1.0,"#3FB950"]],
        zmid=0,
        text=np.round(m_df.values, 1),
        texttemplate="%{text}%",
        textfont_size=9,
        colorbar=dict(title="%", tickfont_size=9),
    ))
    fig_mh.update_layout(
        **{**LAYOUT_BASE, "height": 320,
           "title": dict(text="Monthly Returns (%)", font_size=12)},
        xaxis=dict(gridcolor="#21262D"),
        yaxis=dict(gridcolor="#21262D", autorange="reversed"),
    )
    st.plotly_chart(fig_mh, use_container_width=True,
                    config={"displayModeBar": False})

    # Rolling annualised stats
    section_header("Rolling Risk Metrics")
    col3, col4 = st.columns(2)
    with col3:
        roll_ann_ret = returns.rolling(63).apply(
            lambda x: (1+x).prod()**(252/len(x))-1, raw=True) * 100
        roll_ann_vol = returns.rolling(63).std() * np.sqrt(252) * 100
        fig_rv = go.Figure()
        fig_rv.add_trace(go.Scatter(x=roll_ann_ret.index, y=roll_ann_ret,
            name="Ann. Return", line=dict(color="#3FB950", width=1.5)))
        fig_rv.add_trace(go.Scatter(x=roll_ann_vol.index, y=roll_ann_vol,
            name="Ann. Volatility", line=dict(color="#F85149", width=1.5)))
        fig_rv.add_hline(y=0, line_color="#8B949E", line_dash="dot", line_width=1)
        fig_rv.update_layout(**{**LAYOUT_BASE, "height": 280,
            "title": dict(text="63-Day Rolling Return & Volatility (%)", font_size=12)})
        st.plotly_chart(fig_rv, use_container_width=True,
                        config={"displayModeBar": False})
    with col4:
        roll_sharpe = (returns.rolling(63).mean() /
                       returns.rolling(63).std() * np.sqrt(252))
        fig_rsh = go.Figure()
        colours_sh = ["#3FB950" if v >= 0 else "#F85149" for v in roll_sharpe.fillna(0)]
        fig_rsh.add_trace(go.Bar(x=roll_sharpe.index, y=roll_sharpe,
            name="Rolling Sharpe", marker_color=colours_sh, opacity=0.75))
        fig_rsh.add_hline(y=1,  line_color="#E3B341", line_dash="dot", line_width=1)
        fig_rsh.add_hline(y=0,  line_color="#8B949E", line_dash="dot", line_width=1)
        fig_rsh.add_hline(y=-1, line_color="#F85149", line_dash="dot", line_width=1)
        fig_rsh.update_layout(**{**LAYOUT_BASE, "height": 280,
            "title": dict(text="63-Day Rolling Sharpe Ratio", font_size=12)})
        st.plotly_chart(fig_rsh, use_container_width=True,
                        config={"displayModeBar": False})

# ════════════════════════════════════════════
# TAB 4 — MONTE CARLO
# ════════════════════════════════════════════
with tabs[3]:
    section_header(f"Monte Carlo Simulation — {mc_sims} Paths · {mc_days}-Day Horizon")

    with st.spinner("Running simulation…"):
        sim_df = monte_carlo(df["Close"], n_simulations=mc_sims, n_days=mc_days)

    fig_mc = monte_carlo_chart(df["Close"], sim_df, ticker, height=460)
    st.plotly_chart(fig_mc, use_container_width=True,
                    config={"displayModeBar": False})

    # Outcome distribution
    section_header("Simulated Outcome Distribution")
    final_prices = sim_df.iloc[-1]
    last_price   = float(df["Close"].iloc[-1])
    final_returns = (final_prices / last_price - 1) * 100

    col1, col2 = st.columns([2, 1])
    with col1:
        fig_fo = go.Figure()
        fig_fo.add_trace(go.Histogram(x=final_returns, nbinsx=60,
            name=f"Day {mc_days} Returns",
            marker_color="#58A6FF", opacity=0.75))
        fig_fo.add_vline(x=0, line_color="#8B949E", line_dash="dot", line_width=1)
        fig_fo.add_vline(x=float(final_returns.quantile(0.05)),
                         line_color="#F85149", line_dash="dash", line_width=1.5,
                         annotation_text=" 5th pct", annotation_font_color="#F85149")
        fig_fo.add_vline(x=float(final_returns.quantile(0.95)),
                         line_color="#3FB950", line_dash="dash", line_width=1.5,
                         annotation_text=" 95th pct", annotation_font_color="#3FB950")
        fig_fo.update_layout(
            **{**LAYOUT_BASE, "height": 300,
               "title": dict(text=f"Distribution of Day {mc_days} Outcomes", font_size=12)},
            xaxis_title="Return (%)", yaxis_title="Frequency",
        )
        st.plotly_chart(fig_fo, use_container_width=True,
                        config={"displayModeBar": False})

    with col2:
        p5, p25, p50, p75, p95 = [float(final_returns.quantile(q))
                                   for q in [0.05, 0.25, 0.50, 0.75, 0.95]]
        prob_pos = float((final_returns > 0).mean()) * 100
        exp_ret  = float(final_returns.mean())
        exp_loss = float(final_returns[final_returns < 0].mean()) if (final_returns < 0).any() else 0

        st.markdown(f"""
        <div style="display:flex;flex-direction:column;gap:8px;padding-top:8px">
          {"".join(f'''<div style="background:#161B22;border:1px solid #30363D;border-radius:6px;
          padding:10px 14px;display:flex;justify-content:space-between;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                         color:#8B949E;text-transform:uppercase">{k}</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:13px;
                         font-weight:600;color:{vc}">{v}</span></div>'''
          for k, v, vc in [
              ("Expected Return",    f"{exp_ret:+.2f}%",   "#3FB950" if exp_ret>0 else "#F85149"),
              ("5th Percentile",     f"{p5:+.2f}%",        "#F85149"),
              ("25th Percentile",    f"{p25:+.2f}%",       "#FFA657"),
              ("Median",             f"{p50:+.2f}%",       "#E3B341"),
              ("75th Percentile",    f"{p75:+.2f}%",       "#58A6FF"),
              ("95th Percentile",    f"{p95:+.2f}%",       "#3FB950"),
              ("Prob. of Gain",      f"{prob_pos:.1f}%",   "#3FB950" if prob_pos>50 else "#F85149"),
              ("Avg Loss (tail)",    f"{exp_loss:+.2f}%",  "#F85149"),
          ])}
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 5 — CAPM & BETA
# ════════════════════════════════════════════
with tabs[4]:
    section_header(f"CAPM Analysis vs {benchmark_input}")

    if bench.empty:
        st.warning(f"Could not load benchmark data for **{benchmark_input}**.")
    else:
        bench_ret = compute_returns(bench["Close"])
        stock_ret = returns

        aligned = pd.concat([stock_ret, bench_ret], axis=1, join="inner").dropna()
        aligned.columns = [ticker, benchmark_input]

        # Scatter + regression
        col1, col2 = st.columns([3, 2])
        with col1:
            from scipy import stats as scipy_stats
            x_vals = aligned[benchmark_input].values
            y_vals = aligned[ticker].values
            slope, intercept, r_val, _, stderr = scipy_stats.linregress(x_vals, y_vals)

            x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
            y_line = slope * x_line + intercept

            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(x=x_vals*100, y=y_vals*100,
                mode="markers", name="Daily Returns",
                marker=dict(color="#58A6FF", size=3, opacity=0.5)))
            fig_sc.add_trace(go.Scatter(x=x_line*100, y=y_line*100,
                mode="lines", name=f"β = {slope:.3f}",
                line=dict(color="#E3B341", width=2)))
            fig_sc.add_hline(y=0, line_color="#8B949E", line_dash="dot", line_width=0.8)
            fig_sc.add_vline(x=0, line_color="#8B949E", line_dash="dot", line_width=0.8)
            fig_sc.update_layout(
                **{**LAYOUT_BASE, "height": 400,
                   "title": dict(
                       text=f"{ticker} vs {benchmark_input} — β = {slope:.3f}  R² = {r_val**2:.3f}",
                       font_size=12)},
                xaxis_title=f"{benchmark_input} Daily Return (%)",
                yaxis_title=f"{ticker} Daily Return (%)",
            )
            st.plotly_chart(fig_sc, use_container_width=True,
                            config={"displayModeBar": False})

        with col2:
            section_header("CAPM Statistics")
            capm = report
            items = [
                ("Beta",          f"{capm.get('beta',0):.4f}",  "Systematic risk"),
                ("Alpha (Ann.)",  f"{capm.get('alpha',0)*100:.3f}%", "Excess return"),
                ("R²",            f"{capm.get('r_squared',0):.4f}", "Explained variance"),
                ("Treynor Ratio", f"{capm.get('treynor',0):.4f}", "Return per unit β"),
                ("Info Ratio",    f"{capm.get('information_ratio',0):.4f}", "Active return / TE"),
                ("Sharpe",        f"{capm.get('sharpe_ratio',0):.4f}", "Risk-adj. return"),
                ("Sortino",       f"{capm.get('sortino_ratio',0):.4f}", "Downside-adj."),
                ("Calmar",        f"{capm.get('calmar_ratio',0):.4f}", "Return / MaxDD"),
            ]
            rows = "".join(f"""<tr>
              <td style="padding:7px 10px;border-bottom:1px solid #30363D;
                         font-family:'IBM Plex Mono',monospace;font-size:11px;
                         color:#8B949E;text-transform:uppercase">{k}</td>
              <td style="padding:7px 10px;border-bottom:1px solid #30363D;
                         font-family:'IBM Plex Mono',monospace;font-size:13px;
                         font-weight:600;color:#C9D1D9">{v}</td>
              <td style="padding:7px 10px;border-bottom:1px solid #30363D;
                         font-size:11px;color:#8B949E">{n}</td>
            </tr>""" for k, v, n in items)
            st.markdown(f"""<table style="width:100%;border-collapse:collapse;
              background:#161B22;border:1px solid #30363D;border-radius:6px;overflow:hidden">
              <thead><tr style="background:#21262D">
                <th style="padding:8px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;
                   font-size:10px;color:#8B949E;text-transform:uppercase">Metric</th>
                <th style="padding:8px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;
                   font-size:10px;color:#8B949E;text-transform:uppercase">Value</th>
                <th style="padding:8px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;
                   font-size:10px;color:#8B949E;text-transform:uppercase">Note</th>
              </tr></thead><tbody>{rows}</tbody></table>""", unsafe_allow_html=True)

        # Rolling Beta
        section_header("Rolling 63-Day Beta")
        roll_cov  = aligned[ticker].rolling(63).cov(aligned[benchmark_input])
        roll_var  = aligned[benchmark_input].rolling(63).var()
        roll_beta = roll_cov / roll_var

        fig_rb = go.Figure()
        fig_rb.add_trace(go.Scatter(x=roll_beta.index, y=roll_beta,
            name="Rolling Beta",
            line=dict(color="#E3B341", width=1.8),
            fill="tozeroy", fillcolor="rgba(227,179,65,0.07)"))
        fig_rb.add_hline(y=1, line_color="#8B949E", line_dash="dot",
                         annotation_text=" β = 1 (market)")
        fig_rb.add_hline(y=0, line_color="#8B949E", line_dash="dot", line_width=0.5)
        fig_rb.update_layout(
            **{**LAYOUT_BASE, "height": 280,
               "title": dict(text=f"Rolling Beta vs {benchmark_input} (63-Day)", font_size=12)})
        st.plotly_chart(fig_rb, use_container_width=True,
                        config={"displayModeBar": False})
