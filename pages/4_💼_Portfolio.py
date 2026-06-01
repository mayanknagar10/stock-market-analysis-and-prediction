"""
Page 4 — Portfolio Tracker
Multi-stock comparison, correlation heatmap, portfolio P&L tracking,
allocation summary, and risk-adjusted performance table.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="Portfolio · StockPro", page_icon="💼",
                   layout="wide", initial_sidebar_state="expanded")

from core.data_fetcher import fetch_ohlcv, validate_ticker, PERIOD_MAP, detect_market, currency_symbol
from core.risk_metrics import (compute_returns, annualised_return,
                               annualised_volatility, sharpe_ratio,
                               sortino_ratio, drawdown_analysis, var_historical)
from utils.helpers     import (inject_css, section_header, kpi_row, kpi_card,
                               fmt_pct, fmt_price, fmt_large)
from utils.charts      import (correlation_heatmap, portfolio_performance_chart, THEME)
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

DEFAULT_TICKERS  = "RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS"
DEFAULT_WEIGHTS  = "20, 20, 20, 20, 20"

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:16px;
    font-weight:600;color:#3FB950;padding:8px 0 16px;">📈 StockPro
    <span style="font-size:10px;color:#8B949E;font-weight:400;display:block;
    letter-spacing:.1em;margin-top:2px;">ANALYTICS TERMINAL</span></div>""",
    unsafe_allow_html=True)

    raw_tickers = st.text_area(
        "Portfolio Tickers",
        value=DEFAULT_TICKERS,
        help="NSE: RELIANCE.NS  |  BSE: RELIANCE.BO  |  US: AAPL  (max 10)",
        height=90,
    )
    raw_weights = st.text_area(
        "Portfolio Weights (%)",
        value=DEFAULT_WEIGHTS,
        help="Comma-separated weights that sum to 100",
        height=68,
    )
    period_label = st.selectbox("Time Period", list(PERIOD_MAP.keys()), index=3)
    period, interval = PERIOD_MAP[period_label]

    # Auto-detect benchmark from first valid ticker
    _first_t = tickers_raw[0] if tickers_raw else "AAPL"
    _auto_bench = {"NSE": "^NSEI", "BSE": "^BSESN"}.get(detect_market(_first_t), "^GSPC")
    benchmark_t = st.text_input("Benchmark", value=_auto_bench,
                                help="Auto-detected · NSE→^NSEI  BSE→^BSESN  US→^GSPC")
    init_invest  = st.number_input("Initial Investment ($)", value=100_000,
                                   min_value=1_000, step=10_000)

    st.divider()
    st.caption("Data via Yahoo Finance · Not financial advice")

# ─── PARSE INPUTS ───────────────────────────────────────────────────────────
tickers_raw = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()][:10]
try:
    weights_raw = [float(w.strip()) for w in raw_weights.split(",") if w.strip()]
except ValueError:
    weights_raw = [100.0 / len(tickers_raw)] * len(tickers_raw)

# Align lengths
n = min(len(tickers_raw), len(weights_raw))
tickers_raw  = tickers_raw[:n]
weights_raw  = weights_raw[:n]
total_w      = sum(weights_raw)
weights_norm = [w / total_w for w in weights_raw]

if not tickers_raw:
    st.info("Enter at least one ticker symbol in the sidebar.")
    st.stop()

# ─── LOAD DATA ──────────────────────────────────────────────────────────────
with st.spinner("Loading portfolio data…"):
    prices_dict = {}
    failed = []
    for t in tickers_raw:
        df_t = fetch_ohlcv(t, period, interval)
        if not df_t.empty:
            prices_dict[t] = df_t["Close"]
        else:
            failed.append(t)

if failed:
    st.warning(f"Could not load data for: {', '.join(failed)}")

tickers = list(prices_dict.keys())
if not tickers:
    st.error("No valid tickers loaded.")
    st.stop()

# Align weights after failures
weights_norm = [weights_norm[tickers_raw.index(t)] for t in tickers]
total_w2 = sum(weights_norm)
weights_norm = [w / total_w2 for w in weights_norm]

# Build aligned price DataFrame
prices_df = pd.DataFrame(prices_dict).dropna()
returns_df = prices_df.pct_change().dropna()

# Portfolio returns
port_returns = (returns_df * weights_norm).sum(axis=1)

# Benchmark
bench_df = fetch_ohlcv(benchmark_t.upper().strip() or "^GSPC", period, interval)
bench_ret = compute_returns(bench_df["Close"]) if not bench_df.empty else None

# ─── CURRENCY DETECTION ─────────────────────────────────────────────────────
# Use first ticker to decide currency (mixed portfolios use the majority market)
_mkt_counts = {}
for t in tickers:
    m = detect_market(t)
    _mkt_counts[m] = _mkt_counts.get(m, 0) + 1
_dominant_mkt = max(_mkt_counts, key=_mkt_counts.get)
_curr = "INR" if _dominant_mkt in ("NSE","BSE") else "USD"
_sym  = currency_symbol(_curr)
_flag = "🇮🇳" if _dominant_mkt in ("NSE","BSE") else "🇺🇸"

# ─── HEADER ─────────────────────────────────────────────────────────────────
st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;padding:10px 0 6px;
border-bottom:1px solid #30363D;margin-bottom:16px;">
<span style="font-size:20px;font-weight:600;color:#C9D1D9">Portfolio Tracker</span>&nbsp;&nbsp;
<span style="font-size:11px;color:#E3B341;font-family:'IBM Plex Mono',monospace">{_flag} {_dominant_mkt}</span>&nbsp;&nbsp;
<span style="font-size:13px;color:#8B949E">
  {len(tickers)} positions · {period_label} · {_sym}{init_invest:,.0f} invested
</span></div>""", unsafe_allow_html=True)

# ─── KPI ROW ────────────────────────────────────────────────────────────────
section_header("Portfolio Overview")
p_ann_ret = annualised_return(port_returns)
p_ann_vol = annualised_volatility(port_returns)
p_sharpe  = sharpe_ratio(port_returns)
p_sortino = sortino_ratio(port_returns)
_, p_mdd, _ = drawdown_analysis((1 + port_returns).cumprod() * init_invest)
p_cum_ret = float((1 + port_returns).prod() - 1)
p_var95   = var_historical(port_returns, 0.95)
port_val  = init_invest * (1 + p_cum_ret)

kpi_row([
    kpi_card("Portfolio Value",  f"{_sym}{port_val:,.0f}",      f"Started {_sym}{init_invest:,.0f}"),
    kpi_card("Total Return",     fmt_pct(p_cum_ret),        period_label,
             "pos" if p_cum_ret >= 0 else "neg"),
    kpi_card("Ann. Return",      fmt_pct(p_ann_ret),        "",
             "pos" if p_ann_ret >= 0 else "neg"),
    kpi_card("Ann. Volatility",  f"{p_ann_vol*100:.1f}%",  ""),
    kpi_card("Sharpe Ratio",     f"{p_sharpe:.2f}",         ">1 = good",
             "pos" if p_sharpe >= 1 else ""),
    kpi_card("Sortino Ratio",    f"{p_sortino:.2f}",        "Downside adj.",
             "pos" if p_sortino >= 1 else ""),
    kpi_card("Max Drawdown",     f"{p_mdd*100:.1f}%",      "", "neg"),
    kpi_card("VaR (95%)",        f"{p_var95*100:.2f}%",    f"{_sym}{p_var95*port_val:,.0f}", "neg"),
])

# ─── CHARTS TABS ────────────────────────────────────────────────────────────
tabs = st.tabs([
    "  Performance  ", "  Allocation  ",
    "  Correlation  ", "  Risk Table  ", "  Individual  "
])

# ════════════════════════════════════════════
# TAB 1 — PERFORMANCE
# ════════════════════════════════════════════
with tabs[0]:
    section_header("Cumulative Performance (Indexed to 100)")

    # Build normalised price DataFrame including portfolio and benchmark
    norm_prices = prices_df.divide(prices_df.iloc[0]) * 100
    port_idx    = (1 + port_returns).cumprod() * 100
    port_idx.name = "Portfolio"

    plot_df = pd.concat([norm_prices, port_idx], axis=1)
    if bench_ret is not None:
        bench_idx = (1 + bench_ret.reindex(port_idx.index).fillna(0)).cumprod() * 100
        bench_idx.name = benchmark_t
        plot_df = pd.concat([plot_df, bench_idx], axis=1)

    colours = ["#3FB950","#58A6FF","#E3B341","#BC8CFF","#FFA657",
               "#79C0FF","#F85149","#3FB950","#58A6FF","#E3B341",
               "#FFFFFF", "#8B949E"]

    fig_perf = go.Figure()
    for i, col in enumerate(plot_df.columns):
        is_port  = col == "Portfolio"
        is_bench = col == benchmark_t
        width  = 2.5 if is_port else (1.5 if is_bench else 1.2)
        dash   = "solid" if (is_port or is_bench) else "dot"
        colour = "#FFFFFF" if is_port else ("#8B949E" if is_bench else colours[i % len(colours)])
        fig_perf.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df[col], name=col,
            line=dict(color=colour, width=width, dash=dash),
            opacity=0.9 if (is_port or is_bench) else 0.75,
        ))

    fig_perf.add_hline(y=100, line_color="#8B949E", line_dash="dot", line_width=0.8)
    fig_perf.update_layout(
        **{**LAYOUT_BASE, "height": 460,
           "title": dict(text="Normalised Performance (100 = start)", font_size=12)},
        yaxis_title="Indexed Value",
    )
    st.plotly_chart(fig_perf, use_container_width=True,
                    config={"displayModeBar": False})

    # Absolute P&L
    section_header("Absolute P&L ($)")
    pnl_df = (prices_df.divide(prices_df.iloc[0]) - 1)
    for t, w in zip(tickers, weights_norm):
        pnl_df[t] = pnl_df[t] * w * init_invest

    port_pnl = port_returns.cumsum() * init_invest
    fig_pnl = go.Figure()
    for i, t in enumerate(tickers):
        pnl_series = (prices_df[t] / prices_df[t].iloc[0] - 1) * weights_norm[i] * init_invest
        fig_pnl.add_trace(go.Bar(x=pnl_series.index, y=pnl_series,
            name=t, marker_color=colours[i % len(colours)], opacity=0.75))
    fig_pnl.update_layout(
        **{**LAYOUT_BASE, "height": 320,
           "title": dict(text="Contribution to P&L by Position ($)", font_size=12)},
        barmode="stack", yaxis_title="P&L ($)",
    )
    st.plotly_chart(fig_pnl, use_container_width=True,
                    config={"displayModeBar": False})

# ════════════════════════════════════════════
# TAB 2 — ALLOCATION
# ════════════════════════════════════════════
with tabs[1]:
    section_header("Portfolio Allocation")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Donut chart
        fig_pie = go.Figure(go.Pie(
            labels=tickers,
            values=[w * 100 for w in weights_norm],
            hole=0.55,
            marker=dict(colors=colours[:len(tickers)],
                        line=dict(color="#0D1117", width=2)),
            textinfo="label+percent",
            textfont=dict(family="IBM Plex Mono, monospace", size=11),
        ))
        fig_pie.update_layout(
            plot_bgcolor="#0D1117", paper_bgcolor="#0D1117", height=380,
            font=dict(family="IBM Plex Mono, monospace", color="#C9D1D9"),
            margin=dict(l=12, r=12, t=36, b=12),
            title=dict(text="Current Allocation (%)", font_size=12),
            legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
            annotations=[dict(text="Portfolio", x=0.5, y=0.5, showarrow=False,
                              font_size=13, font_color="#C9D1D9")],
        )
        st.plotly_chart(fig_pie, use_container_width=True,
                        config={"displayModeBar": False})

    with col2:
        section_header("Position Summary")
        rows = ""
        for t, w in zip(tickers, weights_norm):
            last_p  = float(prices_df[t].iloc[-1])
            first_p = float(prices_df[t].iloc[0])
            t_ret   = (last_p / first_p - 1) * 100
            t_val   = w * init_invest * (1 + t_ret / 100)
            colour  = "#3FB950" if t_ret >= 0 else "#F85149"
            sign    = "+" if t_ret >= 0 else ""
            rows += f"""<tr>
              <td style="padding:7px 10px;border-bottom:1px solid #30363D;
                         font-family:'IBM Plex Mono',monospace;font-size:12px;
                         font-weight:600;color:#C9D1D9">{t}</td>
              <td style="padding:7px 10px;border-bottom:1px solid #30363D;
                         font-family:'IBM Plex Mono',monospace;font-size:12px;
                         color:#8B949E">{w*100:.1f}%</td>
              <td style="padding:7px 10px;border-bottom:1px solid #30363D;
                         font-family:'IBM Plex Mono',monospace;font-size:12px;
                         color:#C9D1D9">${last_p:,.2f}</td>
              <td style="padding:7px 10px;border-bottom:1px solid #30363D;
                         font-family:'IBM Plex Mono',monospace;font-size:12px;
                         color:{colour}">{sign}{t_ret:.2f}%</td>
              <td style="padding:7px 10px;border-bottom:1px solid #30363D;
                         font-family:'IBM Plex Mono',monospace;font-size:12px;
                         color:#C9D1D9">${t_val:,.0f}</td>
            </tr>"""
        st.markdown(f"""<table style="width:100%;border-collapse:collapse;
          background:#161B22;border:1px solid #30363D;border-radius:6px;overflow:hidden">
          <thead><tr style="background:#21262D">
            {"".join(f'<th style="padding:8px 10px;text-align:left;font-family:'
                     + "'IBM Plex Mono',monospace;"
                     + f'font-size:10px;color:#8B949E;text-transform:uppercase">{h}</th>'
                     for h in ["Ticker","Weight","Last Price","Return","Value"])}
          </tr></thead><tbody>{rows}</tbody></table>""", unsafe_allow_html=True)

    # Sector allocation (if available)
    section_header("Weight Distribution")
    fig_wt = go.Figure(go.Bar(
        x=tickers, y=[w * 100 for w in weights_norm],
        marker_color=colours[:len(tickers)], opacity=0.85,
        text=[f"{w*100:.1f}%" for w in weights_norm],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono, monospace", size=10, color="#C9D1D9"),
    ))
    fig_wt.update_layout(
        **{**LAYOUT_BASE, "height": 280,
           "title": dict(text="Portfolio Weight by Ticker (%)", font_size=12)},
        yaxis_title="Weight (%)", yaxis_range=[0, max(weights_norm)*130],
    )
    st.plotly_chart(fig_wt, use_container_width=True,
                    config={"displayModeBar": False})

# ════════════════════════════════════════════
# TAB 3 — CORRELATION
# ════════════════════════════════════════════
with tabs[2]:
    section_header("Return Correlation Matrix")

    if len(tickers) < 2:
        st.info("Add at least 2 tickers to see the correlation matrix.")
    else:
        fig_corr = correlation_heatmap(returns_df, height=440)
        st.plotly_chart(fig_corr, use_container_width=True,
                        config={"displayModeBar": False})

        section_header("Rolling 30-Day Pairwise Correlations")
        if len(tickers) == 2:
            roll_corr = returns_df.iloc[:,0].rolling(30).corr(returns_df.iloc[:,1])
            fig_rc = go.Figure()
            colours_rc = ["#3FB950" if v >= 0 else "#F85149" for v in roll_corr.fillna(0)]
            fig_rc.add_trace(go.Bar(x=roll_corr.index, y=roll_corr,
                marker_color=colours_rc, opacity=0.75,
                name=f"{tickers[0]} vs {tickers[1]}"))
            fig_rc.add_hline(y=0, line_color="#8B949E", line_dash="dot", line_width=1)
            fig_rc.update_layout(
                **{**LAYOUT_BASE, "height": 280,
                   "title": dict(
                       text=f"Rolling 30D Correlation: {tickers[0]} vs {tickers[1]}",
                       font_size=12)},
                yaxis_range=[-1.1, 1.1],
            )
            st.plotly_chart(fig_rc, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            # Show first pair + average cross-correlation
            pairs = [(tickers[i], tickers[j])
                     for i in range(len(tickers))
                     for j in range(i+1, len(tickers))][:4]

            cols = st.columns(min(len(pairs), 2))
            pair_colours = ["#3FB950","#58A6FF","#E3B341","#BC8CFF"]
            for idx, (t1, t2) in enumerate(pairs):
                rc = returns_df[t1].rolling(30).corr(returns_df[t2])
                with cols[idx % 2]:
                    fig_rc2 = go.Figure()
                    fig_rc2.add_trace(go.Scatter(x=rc.index, y=rc,
                        name=f"{t1} vs {t2}",
                        line=dict(color=pair_colours[idx], width=1.5),
                        fill="tozeroy", fillcolor=f"rgba(63,185,80,0.05)"))
                    fig_rc2.add_hline(y=0, line_color="#8B949E",
                                      line_dash="dot", line_width=1)
                    fig_rc2.update_layout(
                        **{**LAYOUT_BASE, "height": 220,
                           "title": dict(text=f"Corr: {t1} vs {t2}", font_size=11)},
                        yaxis_range=[-1.1, 1.1], showlegend=False,
                    )
                    st.plotly_chart(fig_rc2, use_container_width=True,
                                    config={"displayModeBar": False})

# ════════════════════════════════════════════
# TAB 4 — RISK TABLE
# ════════════════════════════════════════════
with tabs[3]:
    section_header("Risk-Adjusted Performance Table")

    rows_data = []
    for t, w in zip(tickers, weights_norm):
        ret_s = returns_df[t]
        p_s   = prices_df[t]
        ann_r = annualised_return(ret_s)
        ann_v = annualised_volatility(ret_s)
        sh    = sharpe_ratio(ret_s)
        so    = sortino_ratio(ret_s)
        _, mdd, mdd_d = drawdown_analysis(p_s)
        var95_ = var_historical(ret_s, 0.95)
        cum_r  = float(p_s.iloc[-1] / p_s.iloc[0] - 1)

        rows_data.append({
            "Ticker":        t,
            "Weight (%)":    round(w * 100, 1),
            "Cumul. Return": round(cum_r * 100, 2),
            "Ann. Return":   round(ann_r * 100, 2),
            "Ann. Vol (%)":  round(ann_v * 100, 2),
            "Sharpe":        round(sh, 3),
            "Sortino":       round(so, 3),
            "Max DD (%)":    round(mdd * 100, 2),
            "DD Dur (days)": mdd_d,
            "VaR 95% (%)":   round(var95_ * 100, 2),
        })

    risk_df = pd.DataFrame(rows_data).set_index("Ticker")

    def colour_pos_neg(val):
        if isinstance(val, (int, float)):
            color = "#3FB950" if val > 0 else ("#F85149" if val < 0 else "#C9D1D9")
            return f"color: {color}; font-family: IBM Plex Mono, monospace; font-size: 12px"
        return ""

    styled = (
        risk_df.style
        .format({
            "Weight (%)":    "{:.1f}%",
            "Cumul. Return": "{:+.2f}%",
            "Ann. Return":   "{:+.2f}%",
            "Ann. Vol (%)":  "{:.2f}%",
            "Sharpe":        "{:.3f}",
            "Sortino":       "{:.3f}",
            "Max DD (%)":    "{:.2f}%",
            "VaR 95% (%)":   "{:.2f}%",
        })
        .applymap(colour_pos_neg, subset=["Cumul. Return","Ann. Return","Sharpe","Sortino"])
        .background_gradient(subset=["Sharpe"], cmap="RdYlGn", vmin=-1, vmax=2)
        .background_gradient(subset=["Max DD (%)"], cmap="RdYlGn_r", vmin=-50, vmax=0)
    )
    st.dataframe(styled, use_container_width=True, height=380)

    # Risk / Return scatter
    section_header("Risk / Return Map")
    x_vals = [risk_df.loc[t, "Ann. Vol (%)"]  for t in tickers]
    y_vals = [risk_df.loc[t, "Ann. Return"] for t in tickers]
    fig_rr = go.Figure()
    fig_rr.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode="markers+text",
        text=tickers, textposition="top center",
        marker=dict(
            color=[risk_df.loc[t,"Sharpe"] for t in tickers],
            colorscale=[[0,"#F85149"],[0.5,"#E3B341"],[1,"#3FB950"]],
            size=14, opacity=0.9,
            colorbar=dict(title="Sharpe", tickfont_size=9),
            showscale=True,
        ),
        textfont=dict(family="IBM Plex Mono, monospace", size=10, color="#C9D1D9"),
    ))
    # Add portfolio point
    fig_rr.add_trace(go.Scatter(
        x=[annualised_volatility(port_returns)*100],
        y=[annualised_return(port_returns)*100],
        mode="markers+text", text=["Portfolio"],
        textposition="top center",
        marker=dict(color="#FFFFFF", size=16, symbol="star", opacity=1),
        textfont=dict(family="IBM Plex Mono, monospace", size=11, color="#FFFFFF"),
        name="Portfolio",
    ))
    fig_rr.add_hline(y=0, line_color="#8B949E", line_dash="dot", line_width=0.8)
    fig_rr.update_layout(
        **{**LAYOUT_BASE, "height": 420,
           "title": dict(text="Risk / Return Scatter (colour = Sharpe)", font_size=12)},
        xaxis_title="Annualised Volatility (%)",
        yaxis_title="Annualised Return (%)",
        showlegend=False,
    )
    st.plotly_chart(fig_rr, use_container_width=True,
                    config={"displayModeBar": False})

# ════════════════════════════════════════════
# TAB 5 — INDIVIDUAL STOCKS
# ════════════════════════════════════════════
with tabs[4]:
    section_header("Individual Stock Charts")

    selected = st.selectbox("Select Ticker", tickers)
    df_sel   = fetch_ohlcv(selected, period, interval)

    if not df_sel.empty:
        # Mini candlestick
        fig_ind = go.Figure()
        fig_ind.add_trace(go.Candlestick(
            x=df_sel.index, open=df_sel["Open"], high=df_sel["High"],
            low=df_sel["Low"], close=df_sel["Close"], name=selected,
            increasing_line_color="#3FB950", decreasing_line_color="#F85149",
            increasing_fillcolor="#3FB950", decreasing_fillcolor="#F85149",
        ))
        # EMA overlay
        from core.indicators import ema as ema_fn
        fig_ind.add_trace(go.Scatter(x=df_sel.index, y=ema_fn(df_sel["Close"], 20),
            name="EMA 20", line=dict(color="#E3B341", width=1.5)))
        fig_ind.add_trace(go.Scatter(x=df_sel.index, y=ema_fn(df_sel["Close"], 50),
            name="EMA 50", line=dict(color="#BC8CFF", width=1.5)))
        fig_ind.update_layout(
            **{**LAYOUT_BASE, "height": 420,
               "title": dict(text=f"{selected} — OHLCV + EMA 20/50", font_size=12)},
        )
        fig_ind.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig_ind, use_container_width=True,
                        config={"displayModeBar": False})

        # Volume
        v_colours = ["#3FB950" if c >= o else "#F85149"
                     for c, o in zip(df_sel["Close"], df_sel["Open"])]
        fig_vol = go.Figure(go.Bar(
            x=df_sel.index, y=df_sel["Volume"],
            marker_color=v_colours, opacity=0.65, name="Volume",
        ))
        fig_vol.add_trace(go.Scatter(
            x=df_sel.index, y=df_sel["Volume"].rolling(20).mean(),
            name="20D MA Vol", line=dict(color="#E3B341", width=1.5),
        ))
        fig_vol.update_layout(
            **{**LAYOUT_BASE, "height": 220,
               "title": dict(text=f"{selected} — Volume", font_size=12)},
        )
        st.plotly_chart(fig_vol, use_container_width=True,
                        config={"displayModeBar": False})

        # Stats for this stock
        col1, col2, col3, col4 = st.columns(4)
        ret_sel = compute_returns(df_sel["Close"])
        last_px = float(df_sel["Close"].iloc[-1])
        first_px = float(df_sel["Close"].iloc[0])
        chg_pct = (last_px / first_px - 1) * 100
        colour  = "pos" if chg_pct >= 0 else "neg"
        _sel_sym = currency_symbol("INR" if detect_market(selected) in ("NSE","BSE") else "USD")
        with col1:
            kpi_row([kpi_card("Last Close", f"{_sel_sym}{last_px:,.2f}", "")])
        with col2:
            kpi_row([kpi_card("Period Return", f"{chg_pct:+.2f}%", "", colour)])
        with col3:
            sh_sel = sharpe_ratio(ret_sel)
            kpi_row([kpi_card("Sharpe", f"{sh_sel:.2f}", "",
                              "pos" if sh_sel >= 1 else "")])
        with col4:
            _, mdd_sel, _ = drawdown_analysis(df_sel["Close"])
            kpi_row([kpi_card("Max Drawdown", f"{mdd_sel*100:.1f}%", "", "neg")])
